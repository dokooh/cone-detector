"""
utonia_segment_exporter.py
==========================
Reads a color-segmented point cloud produced by Utonia's PCA pipeline
(.ply with per-point float RGB in [0,1]), recovers segments by grouping
points that share the same **hue band** in HSV space, writes ONE .ply
file per gradient-color region, and prints the height / width of every
segmented object.

Why HSV hue-band clustering?
-----------------------------
Utonia's get_pca_color (app.py) produces colors via:

    u, s, v  = torch.pca_lowrank(feat, center=True, q=3*(start+1))
    proj     = feat @ v                       # (N, 3)  PCA projection
    proj     = minmax_normalize(proj)         # → float RGB in [0, 1]
    color    = clamp(proj * brightness, 0, 1)

Points belonging to the same semantic region share similar PCA
projection values → similar **hue** when viewed in HSV.
"Pink", "yellow", "cyan" etc. are hue bands — clustering in circular
hue space is therefore the most faithful way to recover PCA segments.

Clustering pipeline
-------------------
1. Convert float RGB [0,1] → HSV.
2. Build a (cos H, sin H, S, V) feature for circular-safe clustering.
3. K-Means (default, fast) or DBSCAN (no-K fallback) on that feature.
4. Optional: merge any two clusters whose mean hue differs < --merge_deg.
5. Save one .ply per cluster, print AABB dimensions.
6. Post-process each saved .ply:
     a. Statistical outlier removal (SOR).
     b. Extract the largest connected component (radius neighbour graph).
     c. Overwrite the file with the cleaned result.
     d. Re-print dimensions of the cleaned clouds.

Usage
-----
    # Auto-estimate K (elbow method):
    python utonia_segment_exporter.py --input scene_pca.ply

    # Explicit K:
    python utonia_segment_exporter.py --input scene_pca.ply --n_segments 8

    # DBSCAN (no K needed):
    python utonia_segment_exporter.py --input scene_pca.ply --method dbscan --eps 0.12

    # Merge clusters whose hues are within 15 degrees of each other:
    python utonia_segment_exporter.py --input scene_pca.ply --merge_deg 15

    # Tune post-processing:
    python utonia_segment_exporter.py --input scene_pca.ply \\
        --sor_neighbors 30 --sor_std_ratio 1.5 --cc_radius 0.05

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
except ImportError:
    sys.exit("scikit-learn is not installed.  Run:  pip install scikit-learn")


# ══════════════════════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR     = "segments_out"
DEFAULT_METHOD         = "kmeans"
DEFAULT_N_SEGMENTS     = None   # None → auto-estimate via elbow
DEFAULT_DBSCAN_EPS     = 0.12   # in HSV feature space
DEFAULT_DBSCAN_MIN_PTS = 50
DEFAULT_MIN_POINTS     = 10     # clusters smaller than this = noise
DEFAULT_MERGE_DEG      = 0.0    # 0 = no merging

# Post-processing defaults
DEFAULT_SOR_NEIGHBORS  = 20     # Statistical Outlier Removal: kNN neighbours
DEFAULT_SOR_STD_RATIO  = 2.0    # Statistical Outlier Removal: std-dev multiplier
DEFAULT_CC_RADIUS      = 0.05   # Connected-component graph edge radius (scene units)


# ═════════════════════════════════════════════════════��════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_pointcloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a .ply and return:
      xyz    : (N, 3) float64 — XYZ coordinates
      colors : (N, 3) float64 — float RGB in [0, 1]  (Utonia PCA colors)
    """
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        sys.exit(f"ERROR: Point cloud is empty: {path}")
    if not pcd.has_colors():
        sys.exit(
            "ERROR: The point cloud has no color information.\n"
            "Please supply the Utonia PCA-colored output (.ply with float RGB)."
        )

    xyz    = np.asarray(pcd.points,  dtype=np.float64)
    colors = np.asarray(pcd.colors,  dtype=np.float64)   # [0, 1]

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
# RGB → HSV  (vectorised, no OpenCV dependency)
# ══════════════════════════════════════════════════════════════════════════════

def rgb_to_hsv(colors: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) float RGB in [0,1] → (N, 3) HSV.
      H : [0, 2π)   (radians, circular)
      S : [0, 1]
      V : [0, 1]
    """
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]

    v   = np.maximum.reduce([r, g, b])          # value
    s   = np.where(v > 1e-8,
                   (v - np.minimum.reduce([r, g, b])) / np.clip(v, 1e-8, None),
                   0.0)                          # saturation

    diff = np.clip(v - np.minimum.reduce([r, g, b]), 1e-8, None)

    # Hue in [0, 6)
    h = np.where(
        v == r, (g - b) / diff % 6,
        np.where(
            v == g, (b - r) / diff + 2,
                    (r - g) / diff + 4,
        )
    )
    h = h / 6.0 * 2.0 * np.pi   # → radians [0, 2π)

    return np.stack([h, s, v], axis=-1)          # (N, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Build clustering feature vector
# ══════════════════════════════════════════════════════════════════════════════

def build_hsv_features(colors: np.ndarray, hsv_weight: float = 1.0) -> np.ndarray:
    """
    Build a clustering feature that captures gradient-color regions.

    Feature = [cos(H)*S,  sin(H)*S,  V]

    • (cos H * S, sin H * S)  encodes hue circularly, weighted by
      saturation so that near-grey points (S≈0, hue undefined) don't
      pollute hue-based clusters.
    • V (brightness) is included to separate dark from bright regions.

    Parameters
    ----------
    colors     : (N, 3) float64 RGB in [0, 1]
    hsv_weight : scale for the hue/saturation channels (default 1.0)

    Returns
    -------
    features : (N, 3) float64
    """
    hsv = rgb_to_hsv(colors)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    cos_h = np.cos(h) * s * hsv_weight
    sin_h = np.sin(h) * s * hsv_weight
    val   = v

    return np.stack([cos_h, sin_h, val], axis=-1)   # (N, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Cluster merging by hue proximity
# ══════════════════════════════════════════════════════════════════════════════

def _mean_hue(colors_subset: np.ndarray) -> float:
    """Circular mean hue (radians) of a color subset."""
    hsv = rgb_to_hsv(colors_subset)
    h   = hsv[:, 0]
    return float(np.arctan2(np.sin(h).mean(), np.cos(h).mean()) % (2 * np.pi))


def merge_close_hue_clusters(
    labels: np.ndarray,
    colors: np.ndarray,
    merge_deg: float,
) -> np.ndarray:
    """
    Merge any two clusters whose circular mean hue difference is less
    than `merge_deg` degrees.  Uses greedy single-linkage on hue distance.

    Parameters
    ----------
    labels    : (N,) int — cluster labels (-1 = noise)
    colors    : (N, 3) float64 RGB in [0, 1]
    merge_deg : angular threshold in degrees

    Returns
    -------
    new_labels : (N,) int — merged labels
    """
    if merge_deg <= 0:
        return labels

    merge_rad = np.deg2rad(merge_deg)
    unique = [lbl for lbl in np.unique(labels) if lbl != -1]
    mean_hues = {lbl: _mean_hue(colors[labels == lbl]) for lbl in unique}

    # Union-Find
    parent = {lbl: lbl for lbl in unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i, la in enumerate(unique):
        for lb in unique[i + 1:]:
            diff = abs(mean_hues[la] - mean_hues[lb])
            diff = min(diff, 2 * np.pi - diff)   # circular distance
            if diff < merge_rad:
                union(la, lb)

    # Re-map labels through union-find
    root_to_new: dict[int, int] = {}
    new_id = 0
    new_labels = labels.copy()
    for lbl in unique:
        root = find(lbl)
        if root not in root_to_new:
            root_to_new[root] = new_id
            new_id += 1
        new_labels[labels == lbl] = root_to_new[root]

    n_before = len(unique)
    n_after  = len(set(root_to_new.values()))
    if n_after < n_before:
        print(f"Merged {n_before} → {n_after} clusters "
              f"(hue threshold {merge_deg:.1f}°)")
    return new_labels


# ══════════════════════════════════════════════════════════════════════════════
# Clustering
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_k(features: np.ndarray, k_min: int = 2, k_max: int = 20) -> int:
    """Estimate K via the elbow method on a 50 k-point sub-sample."""
    rng     = np.random.default_rng(42)
    idx     = rng.choice(len(features), size=min(50_000, len(features)), replace=False)
    sample  = features[idx]
    inertias = []
    ks       = list(range(k_min, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
        km.fit(sample)
        inertias.append(km.inertia_)
    if len(inertias) < 3:
        return k_min
    d2    = np.diff(np.diff(inertias))
    best_k = ks[int(np.argmax(d2)) + 1]
    print(f"Auto-estimated K = {best_k}")
    return best_k


def segment_by_kmeans(
    colors: np.ndarray,
    n_segments: int | None = None,
    min_points: int = DEFAULT_MIN_POINTS,
) -> np.ndarray:
    """
    Cluster gradient-color regions with K-Means in HSV feature space.

    Parameters
    ----------
    colors     : (N, 3) float64 RGB in [0, 1]
    n_segments : K; auto-estimated if None
    min_points : clusters smaller than this become noise (-1)

    Returns
    -------
    labels : (N,) int
    """
    features = build_hsv_features(colors)
    k = n_segments if n_segments is not None else _estimate_k(features)

    print(f"Running K-Means (K={k}) in HSV feature space …")
    km     = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(features).astype(np.int32)

    # Mark tiny clusters as noise
    for lbl in np.unique(labels):
        if (labels == lbl).sum() < min_points:
            labels[labels == lbl] = -1

    n_valid = len(np.unique(labels[labels != -1]))
    print(f"K-Means gradient-color segments (after noise filter): {n_valid}")
    return labels


def segment_by_dbscan(
    colors: np.ndarray,
    eps: float = DEFAULT_DBSCAN_EPS,
    min_pts: int = DEFAULT_DBSCAN_MIN_PTS,
    min_points: int = DEFAULT_MIN_POINTS,
) -> np.ndarray:
    """
    Cluster gradient-color regions with DBSCAN in HSV feature space.

    Parameters
    ----------
    colors     : (N, 3) float64 RGB in [0, 1]
    eps        : neighborhood radius in HSV feature space
    min_pts    : DBSCAN min_samples
    min_points : clusters smaller than this become noise (-1)

    Returns
    -------
    labels : (N,) int
    """
    features = build_hsv_features(colors)

    print(f"Running DBSCAN (eps={eps}, min_samples={min_pts}) in HSV feature space …")
    db     = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1)
    labels = db.fit_predict(features).astype(np.int32)

    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        if (labels == lbl).sum() < min_points:
            labels[labels == lbl] = -1

    n_valid = len(np.unique(labels[labels != -1]))
    print(f"DBSCAN gradient-color segments (after noise filter): {n_valid}  "
          f"(noise pts: {(labels == -1).sum():,})")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════

def save_segments(
    xyz: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> tuple[list[Path], list[int]]:
    """
    Write one .ply per gradient-color segment (noise label -1 is skipped).

    File naming: segment_<id>_hue<mean_hue_deg>.ply
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    unique_labels = sorted(lbl for lbl in np.unique(labels) if lbl != -1)

    for seg_id, lbl in enumerate(unique_labels):
        mask       = labels == lbl
        seg_xyz    = xyz[mask]
        seg_colors = colors[mask]

        # Compute mean hue for the filename (human-readable)
        mean_hue_deg = int(np.rad2deg(_mean_hue(seg_colors)) % 360)

        seg_pcd = o3d.geometry.PointCloud()
        seg_pcd.points = o3d.utility.Vector3dVector(seg_xyz)
        seg_pcd.colors = o3d.utility.Vector3dVector(seg_colors)

        out_file = output_dir / f"segment_{seg_id:04d}_hue{mean_hue_deg:03d}deg.ply"
        o3d.io.write_point_cloud(str(out_file), seg_pcd)
        saved.append(out_file)

    print(f"\nSaved {len(saved)} segment file(s) → '{output_dir.resolve()}'")
    return saved, unique_labels


# ══════════════════════════════════════════════════════════════════════════════
# Dimension reporting
# ══════════════════════════════════════════════════════════════════════════════

# Approximate hue → color name mapping (HSV hue in degrees)
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


def print_dimensions(
    xyz: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    unique_labels: list[int],
    saved_paths: list[Path],
    header: str = "GRADIENT-COLOR SEGMENT DIMENSIONS",
) -> None:
    """
    Print AABB height / width / depth and dominant hue for every segment.

    height : Z-axis extent  (z_max − z_min)
    width  : X-axis extent  (x_max − x_min)
    depth  : Y-axis extent  (y_max − y_min)
    """
    col_w = 86
    print("\n" + "═" * col_w)
    print(f"  {header}  ({len(unique_labels)} region(s))")
    print("═" * col_w)
    print(
        f"{'ID':>4}  {'Hue (name)':>16}  {'Pts':>8}  "
        f"{'Width(X)':>9}  {'Depth(Y)':>9}  {'Height(Z)':>10}  File"
    )
    print("─" * col_w)

    for seg_id, lbl in enumerate(unique_labels):
        mask       = labels == lbl
        seg_xyz    = xyz[mask]
        seg_colors = colors[mask]

        x_min, x_max = float(seg_xyz[:,0].min()), float(seg_xyz[:,0].max())
        y_min, y_max = float(seg_xyz[:,1].min()), float(seg_xyz[:,1].max())
        z_min, z_max = float(seg_xyz[:,2].min()), float(seg_xyz[:,2].max())

        width  = x_max - x_min
        depth  = y_max - y_min
        height = z_max - z_min

        hue_deg  = np.rad2deg(_mean_hue(seg_colors)) % 360
        hue_str  = f"{hue_deg:5.1f}° ({_hue_name(hue_deg)})"

        print(
            f"{seg_id:>4}  {hue_str:>16}  {mask.sum():>8,}  "
            f"{width:>9.4f}  {depth:>9.4f}  {height:>10.4f}  "
            f"{saved_paths[seg_id].name}"
        )

    print("═" * col_w)
    print("  Units match the coordinate units of the input point cloud.")
    print("═" * col_w)


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Post-processing: outlier removal + largest connected component
# ══════════════════════════════════════════════════════════════════════════════

def remove_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int,
    std_ratio: float,
) -> tuple[o3d.geometry.PointCloud, int]:
    """
    Statistical Outlier Removal (SOR).

    For every point the mean distance to its `nb_neighbors` nearest
    neighbours is computed.  Points whose mean distance exceeds
    (global_mean + std_ratio * global_std) are removed.

    Parameters
    ----------
    pcd          : input point cloud
    nb_neighbors : number of neighbours to consider per point
    std_ratio    : standard-deviation multiplier for the threshold

    Returns
    -------
    clean_pcd    : inlier point cloud
    n_removed    : number of points removed
    """
    n_before = len(pcd.points)
    clean_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    n_removed = n_before - len(clean_pcd.points)
    return clean_pcd, n_removed


def largest_connected_component(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    min_points: int,
) -> tuple[o3d.geometry.PointCloud, int, int]:
    """
    Extract the largest connected component from a point cloud using a
    radius-based neighbour graph.

    Algorithm
    ---------
    Open3D's ``cluster_dbscan`` with ``eps=radius`` and
    ``min_points=1`` groups every point that is within *radius* of at
    least one other point in the same component — this is equivalent to
    single-linkage / connected-components on a radius graph and requires
    **no prior knowledge of K**.

    The label with the highest point count (excluding noise label -1) is
    kept.  If the winning component has fewer than *min_points* the
    function returns the original cloud unchanged and emits a warning.

    Parameters
    ----------
    pcd        : input point cloud (after outlier removal)
    radius     : edge-connection radius in scene units (same as --cc_radius)
    min_points : if the largest component is smaller than this, skip filtering

    Returns
    -------
    largest_pcd  : point cloud of the largest connected component
    n_kept       : number of points kept
    n_components : total number of components found (noise excluded)
    """
    if len(pcd.points) == 0:
        return pcd, 0, 0

    # cluster_dbscan with min_points=1 → every point belongs to some
    # cluster; the labelling is equivalent to connected components on the
    # radius graph.
    cc_labels = np.asarray(
        pcd.cluster_dbscan(eps=radius, min_points=1, print_progress=False)
    )

    unique_cc, counts = np.unique(cc_labels[cc_labels >= 0], return_counts=True)
    n_components = len(unique_cc)

    if n_components == 0:
        # All points were noise (shouldn't happen with min_points=1 unless
        # the cloud is a single isolated point)
        return pcd, len(pcd.points), 0

    best_label = unique_cc[np.argmax(counts)]
    best_count = counts[np.argmax(counts)]

    if best_count < min_points:
        print(
            f"    ⚠  Largest component has only {best_count} pts "
            f"(< min_points={min_points}); skipping CC extraction."
        )
        return pcd, len(pcd.points), n_components

    mask        = cc_labels == best_label
    largest_pcd = pcd.select_by_index(np.where(mask)[0])
    return largest_pcd, int(best_count), n_components


def postprocess_segments(
    saved_paths: list[Path],
    sor_neighbors: int,
    sor_std_ratio: float,
    cc_radius: float,
    min_points: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """
    For every saved segment .ply file (Step 5 output):
      1. Load it.
      2. Statistical Outlier Removal.
      3. Extract the largest connected component (radius graph CC).
      4. Overwrite the file with the cleaned result.

    Returns arrays suitable for re-running ``print_dimensions``:
      all_xyz    : list of (M_i, 3) XYZ arrays
      all_colors : list of (M_i, 3) color arrays
      new_labels : flat label array (0 … N-1), one per point
                   (recycled from segment index so print_dimensions works)
    """
    col_w = 86
    print("\n" + "═" * col_w)
    print("  STEP 6 — POST-PROCESSING: OUTLIER REMOVAL + LARGEST CONNECTED COMPONENT")
    print("═" * col_w)
    print(
        f"  SOR  : nb_neighbors={sor_neighbors}, std_ratio={sor_std_ratio}\n"
        f"  CC   : radius={cc_radius}, min_points={min_points}"
    )
    print("─" * col_w)
    print(
        f"{'ID':>4}  {'File':<45}  {'Before':>8}  "
        f"{'−Outliers':>10}  {'−CC noise':>10}  {'After':>8}"
    )
    print("─" * col_w)

    all_xyz: list[np.ndarray]    = []
    all_colors: list[np.ndarray] = []

    for seg_id, ply_path in enumerate(saved_paths):
        pcd = o3d.io.read_point_cloud(str(ply_path))
        n_before = len(pcd.points)

        # ── 1. Statistical Outlier Removal ───────────────────────────────────
        pcd_sor, n_sor_removed = remove_outliers(pcd, sor_neighbors, sor_std_ratio)
        n_after_sor = len(pcd_sor.points)

        # ── 2. Largest Connected Component ───────────────────────────────────
        pcd_cc, n_kept, n_components = largest_connected_component(
            pcd_sor, cc_radius, min_points
        )
        n_cc_removed = n_after_sor - n_kept

        # ── 3. Overwrite the .ply file ────────────────────────────────────────
        o3d.io.write_point_cloud(str(ply_path), pcd_cc)

        # ── 4. Collect arrays for dimension re-printing ───────────────────────
        xyz_clean    = np.asarray(pcd_cc.points, dtype=np.float64)
        colors_clean = np.asarray(pcd_cc.colors, dtype=np.float64)
        all_xyz.append(xyz_clean)
        all_colors.append(colors_clean)

        print(
            f"{seg_id:>4}  {ply_path.name:<45}  {n_before:>8,}  "
            f"{n_sor_removed:>10,}  {n_cc_removed:>10,}  {n_kept:>8,}"
            + (f"  [{n_components} CC(s)]" if n_components > 1 else "")
        )

    print("═" * col_w)
    print(f"  Cleaned files overwritten in place.\n")

    # Build a flat label array (0 … len(saved_paths)-1) for print_dimensions
    combined_xyz    = np.concatenate(all_xyz,    axis=0)
    combined_colors = np.concatenate(all_colors, axis=0)
    flat_labels     = np.concatenate(
        [np.full(len(x), i, dtype=np.int32) for i, x in enumerate(all_xyz)]
    )

    return combined_xyz, combined_colors, flat_labels


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export gradient-color regions from a Utonia PCA-colored point cloud.\n"
            "Segments are recovered by clustering in HSV hue space — the same\n"
            "color space that get_pca_color() in app.py produces.\n\n"
            "Examples:\n"
            "  python utonia_segment_exporter.py --input scene.ply\n"
            "  python utonia_segment_exporter.py --input scene.ply --n_segments 8\n"
            "  python utonia_segment_exporter.py --input scene.ply --method dbscan --eps 0.1\n"
            "  python utonia_segment_exporter.py --input scene.ply --merge_deg 15\n"
            "  python utonia_segment_exporter.py --input scene.ply \\\n"
            "      --sor_neighbors 30 --sor_std_ratio 1.5 --cc_radius 0.05\n"
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
                   help="[K-Means] Number of gradient-color segments K. "
                        "Auto-estimated if omitted.")

    # DBSCAN
    p.add_argument("--eps", "-e", type=float, default=DEFAULT_DBSCAN_EPS,
                   help="[DBSCAN] Neighborhood radius in HSV feature space.")
    p.add_argument("--min_pts", type=int, default=DEFAULT_DBSCAN_MIN_PTS,
                   help="[DBSCAN] min_samples parameter.")

    # Common
    p.add_argument("--min_points", "-m", type=int, default=DEFAULT_MIN_POINTS,
                   help="Min points per segment; smaller clusters = noise.")
    p.add_argument("--merge_deg", type=float, default=DEFAULT_MERGE_DEG,
                   help="Merge clusters whose mean hue differs by less than "
                        "this many degrees (0 = no merging). "
                        "E.g. --merge_deg 20 collapses near-pink and pink together.")

    # ── Post-processing (Step 6) ──────────────────────────────────────────────
    post = p.add_argument_group("post-processing (Step 6)")
    post.add_argument("--sor_neighbors", type=int, default=DEFAULT_SOR_NEIGHBORS,
                      help="[SOR] Number of nearest neighbours for statistical "
                           "outlier removal (default: %(default)s).")
    post.add_argument("--sor_std_ratio", type=float, default=DEFAULT_SOR_STD_RATIO,
                      help="[SOR] Std-dev multiplier: points beyond "
                           "mean + ratio*std are removed (default: %(default)s). "
                           "Lower = more aggressive.")
    post.add_argument("--cc_radius", type=float, default=DEFAULT_CC_RADIUS,
                      help="[CC] Edge-connection radius for the spatial neighbour "
                           "graph used to find connected components "
                           "(scene units, default: %(default)s). "
                           "Increase for sparser clouds.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: File not found: {input_path.resolve()}")

    # 1. Load
    xyz, colors = load_pointcloud(input_path)

    # 2. Cluster by gradient-color region (HSV hue space)
    if args.method == "kmeans":
        labels = segment_by_kmeans(colors, args.n_segments, args.min_points)
    else:
        labels = segment_by_dbscan(colors, args.eps, args.min_pts, args.min_points)

    # 3. Optionally merge clusters with similar hues
    if args.merge_deg > 0:
        labels = merge_close_hue_clusters(labels, colors, args.merge_deg)

    n_valid = len(np.unique(labels[labels != -1]))
    if n_valid == 0:
        sys.exit(
            "No valid segments found.\n"
            "  • K-Means : try a different --n_segments value.\n"
            "  • DBSCAN  : try a smaller --eps or --min_pts value.\n"
            "  • Both    : try lowering --min_points."
        )

    # 4. Save — one .ply per gradient-color region
    saved_paths, unique_labels = save_segments(xyz, colors, labels, Path(args.output_dir))

    # 5. Print dimension table with hue names (raw, pre-cleaning)
    print_dimensions(xyz, colors, labels, unique_labels, saved_paths,
                     header="RAW SEGMENT DIMENSIONS (before post-processing)")

    # 6. Post-process: outlier removal + largest connected component
    clean_xyz, clean_colors, clean_labels = postprocess_segments(
        saved_paths,
        sor_neighbors=args.sor_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        cc_radius=args.cc_radius,
        min_points=args.min_points,
    )

    # 7. Re-print dimensions of the cleaned segments
    clean_unique = sorted(set(clean_labels))
    print_dimensions(
        clean_xyz, clean_colors, clean_labels,
        clean_unique, saved_paths,
        header="CLEANED SEGMENT DIMENSIONS (after outlier removal + CC extraction)",
    )


if __name__ == "__main__":
    main()
