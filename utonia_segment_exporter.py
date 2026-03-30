"""
utonia_segment_exporter.py
==========================
Reads a color-segmented point cloud produced by Utonia's PCA pipeline
(.ply with per-point float RGB in [0,1]), recovers segments by clustering
the continuous PCA colors, writes ONE .ply file per segment, and prints
the height / width of every segmented object.

How Utonia colors work (from app.py / get_pca_color)
-----------------------------------------------------
Utonia runs torch.pca_lowrank on its feature tensor, then min-max
normalises the 3-component projection into [0, 1] float RGB.
Points belonging to the same semantic region share *similar* (not identical)
RGB values — it is a smooth gradient, not a flat per-segment color.

Segmentation strategy
---------------------
We therefore cluster the float RGB vectors directly:
  • Primary  : K-Means  (fast, works well when you know / estimate K)
  • Fallback : DBSCAN   (no K needed, but slower on large clouds)

Both are implemented in pure numpy / scipy / sklearn — no torch required.

Usage
-----
    # K-Means with automatic K estimation:
    python utonia_segment_exporter.py --input scene_pca.ply

    # K-Means with explicit K:
    python utonia_segment_exporter.py --input scene_pca.ply --n_segments 12

    # DBSCAN (no K):
    python utonia_segment_exporter.py --input scene_pca.ply --method dbscan --eps 0.05

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
# Configuration defaults
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR    = "segments_out"
DEFAULT_N_SEGMENTS    = None   # None → auto-estimate via elbow / gap
DEFAULT_METHOD        = "kmeans"
DEFAULT_DBSCAN_EPS    = 0.05   # in [0,1] RGB space
DEFAULT_DBSCAN_MIN_PTS = 50
DEFAULT_MIN_POINTS    = 10     # segments smaller than this are noise


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_pointcloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a .ply and return (xyz, colors).
      xyz    : (N, 3) float64
      colors : (N, 3) float64 in [0, 1]  ← Utonia PCA colors
    """
    pcd = o3d.io.read_point_cloud(str(path))

    if len(pcd.points) == 0:
        sys.exit(f"ERROR: Point cloud is empty: {path}")
    if not pcd.has_colors():
        sys.exit(
            "ERROR: The point cloud has no color information.\n"
            "Please supply the Utonia PCA-colored output (.ply with float RGB)."
        )

    xyz    = np.asarray(pcd.points,  dtype=np.float64)   # (N, 3)
    colors = np.asarray(pcd.colors,  dtype=np.float64)   # (N, 3) in [0,1]

    print(f"Loaded  : {path.name}")
    print(f"Points  : {len(xyz):,}")
    print(
        f"XYZ range  "
        f"X=[{xyz[:,0].min():.3f}, {xyz[:,0].max():.3f}]  "
        f"Y=[{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}]  "
        f"Z=[{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]"
    )
    print(
        f"RGB range  "
        f"R=[{colors[:,0].min():.3f}, {colors[:,0].max():.3f}]  "
        f"G=[{colors[:,1].min():.3f}, {colors[:,1].max():.3f}]  "
        f"B=[{colors[:,2].min():.3f}, {colors[:,2].max():.3f}]"
    )
    return xyz, colors


# ══════════════════════════════════════════════════════════════════════════════
# PCA color → segment clustering
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_k(colors: np.ndarray, k_min: int = 2, k_max: int = 20) -> int:
    """
    Estimate K for K-Means via the elbow method (inertia drop ratio).
    Runs K-Means for k in [k_min, k_max] on a 50 k-point sub-sample.
    """
    # Sub-sample for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(len(colors), size=min(50_000, len(colors)), replace=False)
    sample = colors[idx]

    inertias = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
        km.fit(sample)
        inertias.append(km.inertia_)

    # Elbow: largest second-derivative of inertia
    if len(inertias) < 3:
        return k_min
    d2 = np.diff(np.diff(inertias))
    best_k = ks[int(np.argmax(d2)) + 1]   # +1 to compensate double diff offset
    print(f"Auto-estimated K = {best_k}")
    return best_k


def _pca_project(colors: np.ndarray) -> np.ndarray:
    """
    Re-apply the same PCA projection that Utonia uses to build the color:
      1. Center the colors.
      2. Compute top-3 principal components (numpy SVD).
      3. Project → (N, 3) float64.

    Clustering in this PCA space is more faithful to the original
    feature space than clustering raw RGB values directly.
    """
    centered = colors - colors.mean(axis=0)
    # Economy SVD: V has shape (3, 3) for 3-channel input
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt.T          # (N, 3)  — same as feat @ v in torch
    return projected


def segment_by_kmeans(
    colors: np.ndarray,
    n_segments: int | None = None,
    min_points: int = DEFAULT_MIN_POINTS,
) -> np.ndarray:
    """
    Cluster PCA colors with K-Means.

    Parameters
    ----------
    colors      : (N, 3) float64 in [0, 1]
    n_segments  : K; auto-estimated if None
    min_points  : clusters smaller than this become label -1 (noise)

    Returns
    -------
    labels : (N,) int  — cluster id per point, -1 = noise
    """
    # Project into the same PCA feature space used to generate the colors
    projected = _pca_project(colors)

    k = n_segments if n_segments is not None else _estimate_k(projected)

    print(f"Running K-Means with K={k} …")
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(projected).astype(np.int32)

    # Mark tiny clusters as noise
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        if (labels == lbl).sum() < min_points:
            labels[labels == lbl] = -1

    n_valid = len(np.unique(labels[labels != -1]))
    print(f"K-Means segments (after noise filter): {n_valid}")
    return labels


def segment_by_dbscan(
    colors: np.ndarray,
    eps: float = DEFAULT_DBSCAN_EPS,
    min_pts: int = DEFAULT_DBSCAN_MIN_PTS,
    min_points: int = DEFAULT_MIN_POINTS,
) -> np.ndarray:
    """
    Cluster PCA colors with DBSCAN.

    Parameters
    ----------
    colors     : (N, 3) float64 in [0, 1]
    eps        : neighborhood radius in PCA-projected space
    min_pts    : DBSCAN min_samples
    min_points : clusters smaller than this become label -1

    Returns
    -------
    labels : (N,) int  — cluster id per point, -1 = noise
    """
    projected = _pca_project(colors)

    print(f"Running DBSCAN (eps={eps}, min_samples={min_pts}) …")
    db = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1)
    labels = db.fit_predict(projected).astype(np.int32)

    # Mark tiny clusters as noise
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        if (labels == lbl).sum() < min_points:
            labels[labels == lbl] = -1

    n_valid = len(np.unique(labels[labels != -1]))
    print(f"DBSCAN segments (after noise filter): {n_valid}  "
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
) -> list[Path]:
    """
    Write one .ply per segment label (noise label -1 is skipped).

    The exported file keeps the original Utonia PCA float RGB colors so
    the per-segment files are visually consistent with the full scene.

    File naming convention:
        segment_<id>.ply
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    unique_labels = sorted(lbl for lbl in np.unique(labels) if lbl != -1)

    for seg_id, lbl in enumerate(unique_labels):
        mask = labels == lbl
        seg_xyz    = xyz[mask]
        seg_colors = colors[mask]

        seg_pcd = o3d.geometry.PointCloud()
        seg_pcd.points = o3d.utility.Vector3dVector(seg_xyz)
        seg_pcd.colors = o3d.utility.Vector3dVector(seg_colors)

        out_file = output_dir / f"segment_{seg_id:04d}.ply"
        o3d.io.write_point_cloud(str(out_file), seg_pcd)
        saved.append(out_file)

    print(f"\nSaved {len(saved)} segment file(s) → '{output_dir.resolve()}'")
    return saved, unique_labels, labels


# ══════════════════════════════════════════════════════════════════════════════
# Dimension reporting
# ══════════════════════════════════════════════════════════════════════════════

def print_dimensions(
    xyz: np.ndarray,
    labels: np.ndarray,
    unique_labels: list[int],
    saved_paths: list[Path],
) -> None:
    """
    Print AABB height / width / depth for every segment.

    Definitions (matching Utonia / cone-detector convention)
    --------
    height : Z-axis extent  (z_max − z_min)   ← vertical
    width  : X-axis extent  (x_max − x_min)   ← horizontal
    depth  : Y-axis extent  (y_max − y_min)   ← depth
    """
    col_w = 72
    print("\n" + "═" * col_w)
    print(f"  SEGMENT DIMENSIONS  ({len(unique_labels)} object(s))")
    print("═" * col_w)
    print(
        f"{'ID':>4}  {'Pts':>8}  {'Width(X)':>9}  "
        f"{'Depth(Y)':>9}  {'Height(Z)':>10}  {'File'}"
    )
    print("─" * col_w)

    for seg_id, lbl in enumerate(unique_labels):
        mask    = labels == lbl
        seg_xyz = xyz[mask]

        x_min, x_max = float(seg_xyz[:, 0].min()), float(seg_xyz[:, 0].max())
        y_min, y_max = float(seg_xyz[:, 1].min()), float(seg_xyz[:, 1].max())
        z_min, z_max = float(seg_xyz[:, 2].min()), float(seg_xyz[:, 2].max())

        width  = x_max - x_min
        depth  = y_max - y_min
        height = z_max - z_min

        print(
            f"{seg_id:>4}  {mask.sum():>8,}  {width:>9.4f}  "
            f"{depth:>9.4f}  {height:>10.4f}  {saved_paths[seg_id].name}"
        )

    print("═" * col_w)
    print("  Units match the coordinate units of the input point cloud.")
    print("═" * col_w)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export segments from a Utonia PCA-colored point cloud.\n"
            "Segments are recovered by clustering the continuous float RGB\n"
            "values using the same PCA projection strategy as app.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True,
                   help="Path to the Utonia PCA-colored .ply file.")
    p.add_argument("--output_dir", "-o", default=DEFAULT_OUTPUT_DIR,
                   help="Directory for per-segment .ply files.")
    p.add_argument("--method", "-M", default=DEFAULT_METHOD,
                   choices=["kmeans", "dbscan"],
                   help="Clustering method: 'kmeans' (default) or 'dbscan'.")

    # K-Means options
    p.add_argument("--n_segments", "-k", type=int, default=DEFAULT_N_SEGMENTS,
                   help="[K-Means] Number of segments K. Auto-estimated if omitted.")

    # DBSCAN options
    p.add_argument("--eps", "-e", type=float, default=DEFAULT_DBSCAN_EPS,
                   help="[DBSCAN] Neighborhood radius in PCA-projected RGB space.")
    p.add_argument("--min_pts", type=int, default=DEFAULT_DBSCAN_MIN_PTS,
                   help="[DBSCAN] min_samples parameter.")

    p.add_argument("--min_points", "-m", type=int, default=DEFAULT_MIN_POINTS,
                   help="Minimum points for a segment to be kept (others = noise).")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: File not found: {input_path.resolve()}")

    # 1. Load
    xyz, colors = load_pointcloud(input_path)

    # 2. Cluster PCA colors using the same projection strategy as app.py
    if args.method == "kmeans":
        labels = segment_by_kmeans(colors, args.n_segments, args.min_points)
    else:
        labels = segment_by_dbscan(colors, args.eps, args.min_pts, args.min_points)

    n_valid = len(np.unique(labels[labels != -1]))
    if n_valid == 0:
        sys.exit(
            "No valid segments found.\n"
            "  • K-Means: try a different --n_segments value.\n"
            "  • DBSCAN:  try a smaller --eps or --min_pts value."
        )

    # 3. Save — one .ply per segment
    saved_paths, unique_labels, labels = save_segments(
        xyz, colors, labels, Path(args.output_dir)
    )

    # 4. Print height / width table
    print_dimensions(xyz, labels, unique_labels, saved_paths)


if __name__ == "__main__":
    main()
