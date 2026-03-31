"""
cone-detector-color-new.py
----------------------
Detects traffic/construction cones in a point cloud from a construction dig site.

Pipeline:
  1. Load point cloud (with color) from file
  2. Segment and remove the ground plane (RANSAC)
  3. Keep only above-ground points
  4. Filter points by orange or white color
  5. Cluster the remaining colored points (DBSCAN)
  6. Apply a bounding-box size threshold (max 0.8 m in any axis)
  7. Visualize and/or export the detected cone clusters

Dependencies:
    pip install open3d numpy scipy
"""

import argparse
import sys
import numpy as np
import open3d as o3d


# ─────────────────────────────────────────────
#  Color thresholds  (RGB, range 0-1)
# ─────────────────────────────────────────────
ORANGE_LOWER = np.array([0.55, 0.20, 0.00])
ORANGE_UPPER = np.array([1.00, 0.65, 0.30])

WHITE_LOWER  = np.array([0.70, 0.70, 0.70])
WHITE_UPPER  = np.array([1.00, 1.00, 1.00])

# ─────────────────────────────────────────────
#  Tunable parameters
# ─────────────────────────────────────────────
GROUND_DISTANCE_THRESHOLD = 0.10   # RANSAC inlier tolerance (m)
GROUND_RANSAC_N           = 3      # min points for RANSAC plane
GROUND_RANSAC_ITER        = 1000   # RANSAC iterations

ABOVE_GROUND_MARGIN       = 0.05   # keep points this far above ground (m)

DBSCAN_EPS                = 0.10   # neighbourhood radius for DBSCAN (m)
DBSCAN_MIN_POINTS         = 5      # min neighbours to form a core point

MAX_CONE_SIZE             = 0.80   # maximum bounding-box extent in any axis (m)
MIN_CONE_SIZE             = 0.05   # minimum bounding-box extent (filters noise)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from any format Open3D supports."""
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        raise ValueError(f"No points found in '{path}'.")
    if not pcd.has_colors():
        raise ValueError(
            f"Point cloud '{path}' has no colour information. "
            "Cone colour detection requires RGB data."
        )
    print(f"[load]   {len(pcd.points):,} points loaded from '{path}'.")
    return pcd


def remove_ground(pcd: o3d.geometry.PointCloud):
    """
    Fit a ground plane with RANSAC and return (above_ground_pcd, plane_model).
    Points whose signed distance to the plane is > ABOVE_GROUND_MARGIN are kept.
    """
    plane_model, inlier_idx = pcd.segment_plane(
        distance_threshold=GROUND_DISTANCE_THRESHOLD,
        ransac_n=GROUND_RANSAC_N,
        num_iterations=GROUND_RANSAC_ITER,
    )
    a, b, c, d = plane_model
    print(f"[ground] Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    print(f"[ground] {len(inlier_idx):,} ground inliers removed.")

    pts   = np.asarray(pcd.points)
    cols  = np.asarray(pcd.colors)
    normal = np.array([a, b, c])

    # Signed distance of every point to the plane (positive = above)
    signed_dist = (pts @ normal + d) / np.linalg.norm(normal)

    above_mask = signed_dist > ABOVE_GROUND_MARGIN
    above_pcd  = o3d.geometry.PointCloud()
    above_pcd.points = o3d.utility.Vector3dVector(pts[above_mask])
    above_pcd.colors = o3d.utility.Vector3dVector(cols[above_mask])

    print(f"[ground] {above_mask.sum():,} above-ground points retained.")
    return above_pcd, plane_model


def filter_by_color(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Keep only orange or white points."""
    cols = np.asarray(pcd.colors)          # shape (N, 3), values 0-1
    pts  = np.asarray(pcd.points)

    orange_mask = (
        (cols[:, 0] >= ORANGE_LOWER[0]) & (cols[:, 0] <= ORANGE_UPPER[0]) &
        (cols[:, 1] >= ORANGE_LOWER[1]) & (cols[:, 1] <= ORANGE_UPPER[1]) &
        (cols[:, 2] >= ORANGE_LOWER[2]) & (cols[:, 2] <= ORANGE_UPPER[2])
    )
    white_mask = (
        (cols[:, 0] >= WHITE_LOWER[0]) & (cols[:, 0] <= WHITE_UPPER[0]) &
        (cols[:, 1] >= WHITE_LOWER[1]) & (cols[:, 1] <= WHITE_UPPER[1]) &
        (cols[:, 2] >= WHITE_LOWER[2]) & (cols[:, 2] <= WHITE_UPPER[2])
    )
    combined_mask = orange_mask | white_mask

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(pts[combined_mask])
    filtered.colors = o3d.utility.Vector3dVector(cols[combined_mask])

    print(
        f"[color]  {combined_mask.sum():,} colour-matched points "
        f"({orange_mask.sum():,} orange, {white_mask.sum():,} white)."
    )
    return filtered


def cluster_and_filter(pcd: o3d.geometry.PointCloud):
    """
    Cluster colour-filtered points with DBSCAN, then reject clusters whose
    bounding box exceeds MAX_CONE_SIZE or is smaller than MIN_CONE_SIZE.

    Returns a list of (cluster_pcd, aabb) tuples.
    """
    labels = np.array(
        pcd.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=False,
        )
    )

    n_clusters = labels.max() + 1 if labels.max() >= 0 else 0
    print(f"[cluster] {n_clusters} raw cluster(s) found (label -1 = noise).")

    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    accepted = []
    for label in range(n_clusters):
        mask      = labels == label
        cl_pts    = pts[mask]
        cl_cols   = cols[mask]

        # Bounding-box extents
        bb_min    = cl_pts.min(axis=0)
        bb_max    = cl_pts.max(axis=0)
        extents   = bb_max - bb_min           # (dx, dy, dz)
        max_ext   = extents.max()
        min_ext   = extents.min()

        if max_ext > MAX_CONE_SIZE:
            print(
                f"[cluster] Cluster {label}: rejected (too large, "
                f"max extent = {max_ext:.2f} m)."
            )
            continue
        if min_ext < MIN_CONE_SIZE:
            print(
                f"[cluster] Cluster {label}: rejected (too small / flat, "
                f"min extent = {min_ext:.3f} m)."
            )
            continue

        cl_pcd = o3d.geometry.PointCloud()
        cl_pcd.points = o3d.utility.Vector3dVector(cl_pts)
        cl_pcd.colors = o3d.utility.Vector3dVector(cl_cols)

        aabb = cl_pcd.get_axis_aligned_bounding_box()
        aabb.color = (1.0, 0.5, 0.0)   # orange box in viewer

        centroid = cl_pts.mean(axis=0)
        print(
            f"[cluster] Cluster {label}: ACCEPTED  "
            f"pts={mask.sum():,}  extents=({extents[0]:.2f}, "
            f"{extents[1]:.2f}, {extents[2]:.2f}) m  "
            f"centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
        )
        accepted.append((cl_pcd, aabb))

    print(f"[cluster] {len(accepted)} cone candidate(s) passed size filter.")
    return accepted


def visualize(
    original_pcd: o3d.geometry.PointCloud,
    cone_clusters,
    plane_model,
    show_original: bool = True,
):
    """Interactive Open3D visualisation."""
    vis_list = []

    if show_original:
        # Dim the full cloud so cones pop
        orig_copy = o3d.geometry.PointCloud(original_pcd)
        orig_cols = np.asarray(orig_copy.colors) * 0.35
        orig_copy.colors = o3d.utility.Vector3dVector(orig_cols)
        vis_list.append(orig_copy)

    for cl_pcd, aabb in cone_clusters:
        vis_list.append(cl_pcd)
        vis_list.append(aabb)

    if not vis_list:
        print("[vis]    Nothing to visualise – no cones detected.")
        return

    print("[vis]    Launching viewer … close window to exit.")
    o3d.visualization.draw_geometries(
        vis_list,
        window_name="Cone Detector – Construction Site",
        width=1280,
        height=720,
    )


def export_clusters(cone_clusters, base_path: str = "cone"):
    """Save each accepted cluster as a separate PCD file."""
    for i, (cl_pcd, _) in enumerate(cone_clusters):
        out = f"{base_path}_{i:03d}.pcd"
        o3d.io.write_point_cloud(out, cl_pcd)
        print(f"[export] Saved cluster {i} → '{out}'")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Detect orange/white construction cones in a coloured point cloud."
    )
    parser.add_argument("input", help="Path to point cloud file (.pcd, .ply, .xyz, …)")
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip the interactive 3-D visualisation.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export each detected cone cluster to a separate PCD file.",
    )
    parser.add_argument(
        "--export-prefix",
        default="cone",
        help="Filename prefix for exported cluster files (default: 'cone').",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=MAX_CONE_SIZE,
        help=f"Max bounding-box size (m) for a valid cone (default: {MAX_CONE_SIZE}).",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=DBSCAN_EPS,
        help=f"DBSCAN neighbourhood radius in metres (default: {DBSCAN_EPS}).",
    )
    args = parser.parse_args()

    # Override globals from CLI flags
    global MAX_CONE_SIZE, DBSCAN_EPS
    MAX_CONE_SIZE = args.max_size
    DBSCAN_EPS    = args.dbscan_eps

    # ── Pipeline ──────────────────────────────
    pcd            = load_point_cloud(args.input)
    above_pcd, plane_model = remove_ground(pcd)
    colored_pcd    = filter_by_color(above_pcd)

    if not colored_pcd.has_points():
        print("[result] No orange or white points found. Exiting.")
        sys.exit(0)

    cone_clusters  = cluster_and_filter(colored_pcd)

    if not cone_clusters:
        print("[result] No cone-sized objects detected.")
    else:
        print(f"[result] ✓ {len(cone_clusters)} cone(s) detected.")

    if args.export:
        export_clusters(cone_clusters, base_path=args.export_prefix)

    if not args.no_viz:
        visualize(pcd, cone_clusters, plane_model)


if __name__ == "__main__":
    main()
