"""
utonia_segment_exporter.py
==========================
Reads a Utonia PCA-colored point cloud (.ply, float RGB in [0,1]),
groups points by gradient-color region (pink, yellow, orange, etc.),
saves one .ply per region, and prints AABB height / width / depth.

Color encoding strategy
-----------------------
Utonia's get_pca_color() (app.py) produces float RGB via min-max
normalised PCA projection.  Semantically related points share a
similar *hue* in HSV space.

The previous (cos H·S, sin H·S, V) 3-D feature caused yellow/orange
bleed because:
  • cos(60°) ≈ 0.50  vs  cos(30°) ≈ 0.87  — only 0.37 apart
  • sin(60°) ≈ 0.87  vs  sin(30°) ≈ 0.50  — only 0.37 apart
  → 2-D distance ≈ 0.52, easily collapsed by K-Means or DBSCAN

Fix: replace the 2-D circular encoding with a **soft hue histogram**
(N_BINS Gaussian-weighted bins covering 0-360°).  Yellow lands heavily
in bin ~60° while orange lands heavily in bin ~30°, giving them
a much larger feature-space separation.  Saturation and value are
appended as separate channels so grey/dark regions don't misfire.

Usage
-----
    python utonia_segment_exporter.py --input scene_pca.ply
    python utonia_segment_exporter.py --input scene_pca.ply --n_segments 10
    python utonia_segment_exporter.py --input scene_pca.ply --method dbscan --eps 0.25
    python utonia_segment_exporter.py --input scene_pca.ply --merge_deg 12
    python utonia_segment_exporter.py --input scene_pca.ply --hue_bins 36 --hue_sigma 8

Requirements
------------
    pip install open3d numpy scikit-learn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d is not installed.  Run:  pip install open3d")

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import normalize
except ImportError:
    sys.exit("scikit-learn is not installed.  Run:  pip install scikit-learn")


# ══════════════════════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR      = "segments_out"
DEFAULT_METHOD          = "kmeans"
DEFAULT_N_SEGMENTS      = None   # None → auto-estimate via elbow
DEFAULT_DBSCAN_EPS      = 0.25   # in soft-hue-histogram feature space
DEFAULT_DBSCAN_MIN_PTS  = 50
DEFAULT_MIN_POINTS      = 10

DEFAULT_HUE_BINS        = 36     # one bin per 10° → fine enough to separate yellow/orange
DEFAULT_HUE_SIGMA_DEG   = 8.0    # Gaussian width per bin (degrees)
DEFAULT_HUE_WEIGHT      = 3.0    # scale applied to hue histogram channels vs S/V
DEFAULT_MERGE_DEG       = 0.0    # 0 = no post-merge


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_pointcloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz float64, colors float64 [0,1])."""
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        sys.exit(f"ERROR: Point cloud is empty: {path}")
    if not pcd.has_colors():
        sys.exit(
            "ERROR: No color data found.\n"
            "Please supply the Utonia PCA-colored .ply (float RGB)."
        )
    xyz    = np.asarray(pcd.points,  dtype=np.float64)
    colors = np.asarray(pcd.colors,  dtype=np.float64)
    print(f"Loaded  : {path.name}")
    print(f"Points  : {len(xyz):,}")
    print(
        f"XYZ range  "
        f"X=[{xyz[:,0].min():.3f}, {xyz[:,0].max():.3f}]  "
        f"Y=[{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}]  "
        f"Z=[{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]"
    )
    return xyz, colors


# ══════════════════════════════════════════════════════════════════════════════
# RGB → HSV  (vectorised, no OpenCV)
# ══════════════════════════════════════════════════════════════════════════════

def rgb_to_hsv(colors: np.ndarray) -> np.ndarray:
    """
    (N,3) float64 RGB [0,1] → (N,3) HSV
      H : [0, 360)  degrees
      S : [0, 1]
      V : [0, 1]
    """
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    v       = np.maximum.reduce([r, g, b])
    mn      = np.minimum.reduce([r, g, b])
    diff    = np.clip(v - mn, 1e-8, None)
    s       = np.where(v > 1e-8, diff / np.clip(v, 1e-8, None), 0.0)

    h = np.select(
        [v == r,      v == g],
        [(g - b) / diff % 6,
         (b - r) / diff + 2],
        default=(r - g) / diff + 4,
    )
    h = (h / 6.0 * 360.0) % 360.0   # degrees [0, 360)
    return np.stack([h, s, v], axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Soft hue-histogram feature  (the core improvement)
# ══════════════════════════════════════════════════════════════════════════════

def build_soft_hue_features(
    colors: np.ndarray,
    n_bins:     int   = DEFAULT_HUE_BINS,
    sigma_deg:  float = DEFAULT_HUE_SIGMA_DEG,
    hue_weight: float = DEFAULT_HUE_WEIGHT,
) -> np.ndarray:
    """
    Build a (N, n_bins + 2) feature matrix for clustering.

    Channels
    --------
    0 … n_bins-1 : soft hue histogram
        Each bin centre is at  k * (360 / n_bins)  degrees.
        Contribution of a point with hue H to bin k:

            w_k = exp( -0.5 * circular_dist(H, centre_k)² / sigma² )

        The row is then multiplied by S (saturation) so that near-grey
        points (S ≈ 0, hue undefined) produce a near-zero histogram and
        don't pollute hue-based clusters.  Rows are L2-normalised.

        With n_bins=36 (10°/bin) and sigma=8°:
          • Yellow (H≈60°) activates bins 5-7 strongly
          • Orange (H≈30°) activates bins 2-4 strongly
          → ~3 bins of separation → clear cluster boundary

    n_bins     : saturation  S  (scalar)
    n_bins + 1 : value        V  (scalar)

    The hue channels are scaled by `hue_weight` before clustering so
    that colour differences dominate over brightness differences.

    Parameters
    ----------
    colors     : (N, 3) float64 RGB in [0, 1]
    n_bins     : number of hue bins  (36 → 10°/bin, 72 → 5°/bin)
    sigma_deg  : Gaussian half-width per bin in degrees
    hue_weight : multiplier for hue channels vs S/V

    Returns
    -------
    features : (N, n_bins + 2) float32
    """
    hsv = rgb_to_hsv(colors)                   # (N, 3)
    H   = hsv[:, 0]                            # degrees [0, 360)
    S   = hsv[:, 1]
    V   = hsv[:, 2]

    # Bin centres: 0°, 360/n_bins°, 2·360/n_bins°, …
    centres = np.linspace(0.0, 360.0, n_bins, endpoint=False)   # (n_bins,)

    # Circular distance: (N, n_bins)
    diff_deg = H[:, None] - centres[None, :]           # (N, n_bins)
    diff_deg = (diff_deg + 180.0) % 360.0 - 180.0      # wrap to [-180, 180)

    # Soft Gaussian activation per bin
    hist = np.exp(-0.5 * (diff_deg / sigma_deg) ** 2)  # (N, n_bins)

    # Weight by saturation: grey points → near-zero histogram
    hist = hist * S[:, None]                            # (N, n_bins)

    # L2-normalise each row (so that overall brightness doesn't dominate)
    row_norms = np.linalg.norm(hist, axis=1, keepdims=True)
    hist      = hist / np.where(row_norms > 1e-8, row_norms, 1.0)

    # Scale hue channels relative to S and V
    hist_weighted = hist * hue_weight                   # (N, n_bins)

    # Append S and V as scalar channels
    features = np.concatenate(
        [hist_weighted, S[:, None], V[:, None]], axis=1
    ).astype(np.float32)                               # (N, n_bins+2)

    return features


# ══════════════════════════════════════════════════════════════════════════════
# Cluster merging by hue proximity
# ═══���══════════════════════════════════════════════════════════════════════════

def _mean_hue_deg(colors_subset: np.ndarray) -> float:
    """Circular mean hue (degrees) of a subset of float-RGB points."""
    h_rad = np.deg2rad(rgb_to_hsv(colors_subset)[:, 0])
    return float(np.rad2deg(np.arctan2(np.sin(h_rad).mean(),
                                       np.cos(h_rad).mean())) % 360.0)


def merge_close_hue_clusters(
    labels: np.ndarray,
    colors: np.ndarray,
    merge_deg: float,
) -> np.ndarray:
    """
    Greedy single-linkage merge of clusters whose circular mean hue
    difference is less than `merge_deg` degrees.
    """
    if merge_deg <= 0.0:
        return labels

    unique = [l for l in np.unique(labels) if l != -1]
    mean_h = {l: _mean_hue_deg(colors[labels == l]) for l in unique}

    # Union-Find
    parent = {l: l for l in unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i, la in enumerate(unique):
        for lb in unique[i + 1:]:
            d = abs(mean_h[la] - mean_h[lb])
            d = min(d, 360.0 - d)
            if d < merge_deg:
                union(la, lb)

    root_to_new: dict[int, int] = {}
    new_id = 0
    new_labels = labels.copy()
    for l in unique:
        root = find(l)
        if root not in root_to_new:
            root_to_new[root] = new_id
            new_id += 1
        new_labels[labels == l] = root_to_new[root]

    n_before = len(unique)
    n_after  = len(set(root_to_new.values()))
    if n_after < n_before:
        print(f"Merged {n_before} → {n_after} clusters "
              f"(hue threshold {merge_deg:.1f}°)")
    return new_labels


# ══════════════════════════════════════════════════════════════════════════════
# K estimation (elbow on soft-hue features)
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_k(
    features: np.ndarray,
    k_min: int = 2,
    k_max: int = 20,
) -> int:
    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(features), size=min(50_000, len(features)), replace=False)
    sample = features[idx]
    inertias = []
    ks       = list(range(k_min, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
        km.fit(sample)
        inertias.append(km.inertia_)
    if len(inertias) < 3:
        return k_min
    d2     = np.diff(np.diff(inertias))
    best_k = ks[int(np.argmax(d2)) + 1]
    print(f"Auto-estimated K = {best_k}")
    return best_k


# ══════════════════════════════════════════════════════════════════════════════
# Clustering
# ══════════════════════════════════════════════════════════════════════════════

def _filter_noise(labels: np.ndarray, min_points: int) -> np.ndarray:
    """Mark clusters smaller than min_points as noise (-1)."""
    labels = labels.copy()
    for lbl in np.unique(labels):
        if lbl != -1 and (labels == lbl).sum() < min_points:
            labels[labels == lbl] = -1
    return labels


def segment_by_kmeans(
    colors: np.ndarray,
    n_segments: int | None,
    min_points: int,
    n_bins: int,
    sigma_deg: float,
    hue_weight: float,
) -> np.ndarray:
    features = build_soft_hue_features(colors, n_bins, sigma_deg, hue_weight)
    k        = n_segments if n_segments is not None else _estimate_k(features)
    print(f"Running K-Means (K={k}, hue_bins={n_bins}, σ={sigma_deg}°) …")
    km     = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(features).astype(np.int32)
    labels = _filter_noise(labels, min_points)
    n_valid = len(np.unique(labels[labels != -1]))
    print(f"K-Means gradient-color segments (after noise filter): {n_valid}")
    return labels


def segment_by_dbscan(
    colors: np.ndarray,
    eps: float,
    min_pts: int,
    min_points: int,
    n_bins: int,
    sigma_deg: float,
    hue_weight: float,
) -> np.ndarray:
    features = build_soft_hue_features(colors, n_bins, sigma_deg, hue_weight)
    print(f"Running DBSCAN (eps={eps}, min_samples={min_pts}, "
          f"hue_bins={n_bins}, σ={sigma_deg}°) …")
    db     = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1)
    labels = db.fit_predict(features).astype(np.int32)
    labels = _filter_noise(labels, min_points)
    n_valid = len(np.unique(labels[labels != -1]))
    print(f"DBSCAN gradient-color segments: {n_valid}  "
          f"(noise pts: {(labels == -1).sum():,})")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════

# Hue name table (degrees)
_HUE_NAMES = [
    (  0,  15, "red"),
    ( 15,  45, "orange"),
    ( 45,  70, "yellow"),
    ( 70, 150, "green"),
    (150, 190, "cyan"),
    (190, 260, "blue"),
    (260, 290, "indigo"),
    (290, 330, "pink/magenta"),
    (330, 360, "red"),
]

def _hue_name(hue_deg: float) -> str:
    for lo, hi, name in _HUE_NAMES:
        if lo <= hue_deg < hi:
            return name
    return "unknown"


def save_segments(
    xyz: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> tuple[list[Path], list[int]]:
    """Write one .ply per gradient-color region. Filename encodes mean hue."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    unique_labels = sorted(l for l in np.unique(labels) if l != -1)

    for seg_id, lbl in enumerate(unique_labels):
        mask       = labels == lbl
        seg_xyz    = xyz[mask]
        seg_colors = colors[mask]
        hue_deg    = int(_mean_hue_deg(seg_colors))
        hue_name   = _hue_name(hue_deg).replace("/", "-")

        seg_pcd = o3d.geometry.PointCloud()
        seg_pcd.points = o3d.utility.Vector3dVector(seg_xyz)
        seg_pcd.colors = o3d.utility.Vector3dVector(seg_colors)

        out_file = output_dir / f"segment_{seg_id:04d}_{hue_name}_hue{hue_deg:03d}deg.ply"
        o3d.io.write_point_cloud(str(out_file), seg_pcd)
        saved.append(out_file)

    print(f"\nSaved {len(saved)} segment file(s) → '{output_dir.resolve()}'")
    return saved, unique_labels


# ══════════════════════════════════════════════════════════════════════════════
# Dimension reporting
# ══════════════════════════════════════════════════════════════════════════════

def print_dimensions(
    xyz: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    unique_labels: list[int],
    saved_paths: list[Path],
) -> None:
    col_w = 92
    print("\n" + "═" * col_w)
    print(f"  GRADIENT-COLOR SEGMENT DIMENSIONS  ({len(unique_labels)} region(s))")
    print("═" * col_w)
    print(
        f"{'ID':>4}  {'Color name':>14}  {'Hue':>7}  {'Pts':>8}  "
        f"{'Width(X)':>9}  {'Depth(Y)':>9}  {'Height(Z)':>10}"
    )
    print("─" * col_w)

    for seg_id, lbl in enumerate(unique_labels):
        mask       = labels == lbl
        seg_xyz    = xyz[mask]
        seg_colors = colors[mask]

        x_min, x_max = float(seg_xyz[:,0].min()), float(seg_xyz[:,0].max())
        y_min, y_max = float(seg_xyz[:,1].min()), float(seg_xyz[:,1].max())
        z_min, z_max = float(seg_xyz[:,2].min()), float(seg_xyz[:,2].max())

        hue_deg = _mean_hue_deg(seg_colors)
        print(
            f"{seg_id:>4}  {_hue_name(hue_deg):>14}  {hue_deg:>6.1f}°  "
            f"{mask.sum():>8,}  "
            f"{x_max-x_min:>9.4f}  {y_max-y_min:>9.4f}  {z_max-z_min:>10.4f}"
        )
        print(f"      ↳ {saved_paths[seg_id].name}")

    print("═" * col_w)
    print("  height=Z-extent · width=X-extent · depth=Y-extent")
    print("═" * col_w)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract gradient-color regions from a Utonia PCA-colored point cloud.\n"
            "Uses a soft hue-histogram feature to separate adjacent hues\n"
            "(yellow vs orange, cyan vs green, etc.).\n\n"
            "Quick start:\n"
            "  python utonia_segment_exporter.py --input scene.ply\n"
            "  python utonia_segment_exporter.py --input scene.ply --n_segments 10\n"
            "  python utonia_segment_exporter.py --input scene.ply --method dbscan --eps 0.25\n"
            "  python utonia_segment_exporter.py --input scene.ply --merge_deg 12\n"
            "\nTuning tips:\n"
            "  --hue_bins 72       finer bins (5°/bin) for very similar colors\n"
            "  --hue_sigma 5       narrower Gaussians for sharper separation\n"
            "  --hue_weight 5      emphasise hue over brightness even more\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True,
                   help="Utonia PCA-colored .ply file.")
    p.add_argument("--output_dir", "-o", default=DEFAULT_OUTPUT_DIR,
                   help="Output directory for per-segment .ply files.")
    p.add_argument("--method", "-M", default=DEFAULT_METHOD,
                   choices=["kmeans", "dbscan"],
                   help="Clustering method (default: kmeans).")

    # K-Means
    p.add_argument("--n_segments", "-k", type=int, default=DEFAULT_N_SEGMENTS,
                   help="[K-Means] K — auto-estimated if omitted.")
    # DBSCAN
    p.add_argument("--eps", "-e", type=float, default=DEFAULT_DBSCAN_EPS,
                   help="[DBSCAN] Neighborhood radius in soft-hue feature space.")
    p.add_argument("--min_pts", type=int, default=DEFAULT_DBSCAN_MIN_PTS,
                   help="[DBSCAN] min_samples.")

    # Hue feature knobs
    p.add_argument("--hue_bins", type=int, default=DEFAULT_HUE_BINS,
                   help=f"Number of soft hue bins (default {DEFAULT_HUE_BINS} → 10°/bin). "
                        "Increase to 72 for 5°/bin when yellow/orange still blend.")
    p.add_argument("--hue_sigma", type=float, default=DEFAULT_HUE_SIGMA_DEG,
                   help=f"Gaussian σ per hue bin in degrees (default {DEFAULT_HUE_SIGMA_DEG}). "
                        "Decrease for sharper hue separation.")
    p.add_argument("--hue_weight", type=float, default=DEFAULT_HUE_WEIGHT,
                   help=f"Scale factor for hue channels vs S/V (default {DEFAULT_HUE_WEIGHT}). "
                        "Increase to make hue dominate over brightness.")

    # Common
    p.add_argument("--min_points", "-m", type=int, default=DEFAULT_MIN_POINTS,
                   help="Min points per segment; smaller → noise.")
    p.add_argument("--merge_deg", type=float, default=DEFAULT_MERGE_DEG,
                   help="Post-merge clusters within this hue distance (degrees). "
                        "0 = no merging.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: File not found: {input_path.resolve()}")

    # 1. Load
    xyz, colors = load_pointcloud(input_path)

    # 2. Cluster in soft-hue-histogram feature space
    if args.method == "kmeans":
        labels = segment_by_kmeans(
            colors, args.n_segments, args.min_points,
            args.hue_bins, args.hue_sigma, args.hue_weight,
        )
    else:
        labels = segment_by_dbscan(
            colors, args.eps, args.min_pts, args.min_points,
            args.hue_bins, args.hue_sigma, args.hue_weight,
        )

    # 3. Optional: merge near-hue clusters
    if args.merge_deg > 0:
        labels = merge_close_hue_clusters(labels, colors, args.merge_deg)

    if len(np.unique(labels[labels != -1])) == 0:
        sys.exit(
            "No valid segments found.\n"
            "  K-Means : try a different --n_segments\n"
            "  DBSCAN  : try a smaller --eps or --min_pts\n"
            "  Both    : try lowering --min_points"
        )

    # 4. Save
    saved_paths, unique_labels = save_segments(xyz, colors, labels, Path(args.output_dir))

    # 5. Report
    print_dimensions(xyz, colors, labels, unique_labels, saved_paths)


if __name__ == "__main__":
    main()
