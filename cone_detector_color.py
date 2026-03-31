#!/usr/bin/env python3
"""
cone_detector_color.py

Detects traffic/safety cones (orange and white) in a point cloud from a
construction dig site. Pipeline:
  1. Load point cloud (supports .pcd, .ply, .las/.laz, .xyz)
  2. Estimate and remove the ground plane (RANSAC)
  3. Keep only above-ground points
  4. Filter by orange / white colour
  5. Cluster the remaining points (DBSCAN)
  6. Apply a bounding-box size threshold (max 0.8 m in any direction)
  7. Report detected cones and (optionally) save/visualise results

Dependencies:
    pip install open3d numpy scipy laspy lazrs-python
"""

import argparse
import sys
import os
import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d is required.  Install with:  pip install open3d")

# ──────────────────────────────────────────────────────────────────────────────
# TUNEABLE PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

# Ground removal
GROUND_DISTANCE_THRESHOLD = 0.10   # RANSAC inlier tolerance (m)
GROUND_RANSAC_N           = 3       # Minimum points per RANSAC sample
GROUND_RANSAC_ITER        = 1000    # RANSAC iterations
ABOVE_GROUND_MARGIN       = 0.05    # Keep points this many metres above ground

# Colour thresholds (RGB in [0, 1])
# ── Orange ──
ORANGE_R_MIN, ORANGE_R_MAX = 0.55, 1.00
ORANGE_G_MIN, ORANGE_G_MAX = 0.20, 0.65
ORANGE_B_MIN, ORANGE_B_MAX = 0.00, 0.35
# ── White ──
WHITE_MIN_BRIGHTNESS       = 0.70   # All channels ≥ this value
WHITE_MAX_SATURATION       = 0.20   # Max (max-min) of RGB channels

# DBSCAN clustering
DBSCAN_EPS           = 0.10   # Neighbourhood radius (m)
DBSCAN_MIN_POINTS    = 15     # Min points to form a cluster

# Size filter
MAX_CONE_DIMENSION   = 0.80   # Maximum extent in any direction (m)
MIN_CONE_DIMENSION   = 0.05   # Minimum extent in any direction (m)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from various formats, normalising colours to [0, 1]."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".pcd", ".ply"):
        pcd = o3d.io.read_point_cloud(path)

    elif ext in (".las", ".laz"):
        try:
            import laspy
        except ImportError:
            sys.exit("laspy is required for .las/.laz files.  pip install laspy lazrs-python")

        las = laspy.read(path)
        xyz = np.vstack((las.x, las.y, las.z)).T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # LAS stores colour as uint16 (0-65535)
        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            r = np.asarray(las.red,   dtype=np.float64) / 65535.0
            g = np.asarray(las.green, dtype=np.float64) / 65535.0
            b = np.asarray(las.blue,  dtype=np.float64) / 65535.0
            pcd.colors = o3d.utility.Vector3dVector(np.vstack((r, g, b)).T)

    elif ext == ".xyz":
        data = np.loadtxt(path)
        pcd  = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
        if data.shape[1] >= 6:
            colours = data[:, 3:6]
            # Auto-detect uint8 range
            if colours.max() > 1.0:
                colours = colours / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colours)

    else:
        sys.exit(f"Unsupported file format: '{ext}'.  "
                 f"Supported: .pcd .ply .las .laz .xyz")

    if not pcd.has_points():
        sys.exit(f"No points loaded from '{path}'.")

    print(f"[load]   Loaded {len(pcd.points):,} points from '{path}'")
    return pcd


def remove_ground(pcd: o3d.geometry.PointCloud):
    """
    Estimate the dominant ground plane with RANSAC, return the
    above-ground point cloud and the plane equation [a, b, c, d].
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold = GROUND_DISTANCE_THRESHOLD,
        ransac_n           = GROUND_RANSAC_N,
        num_iterations     = GROUND_RANSAC_ITER,
    )
    a, b, c, d = plane_model
    print(f"[ground] Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0  "
          f"({len(inliers):,} inliers)")

    # Compute signed distances from the plane; keep points above it
    pts = np.asarray(pcd.points)
    norm = np.sqrt(a**2 + b**2 + c**2)
    signed_dist = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / norm

    # Ensure the normal points upward (positive = above ground)
    if c < 0:
        signed_dist = -signed_dist

    above_mask = signed_dist > ABOVE_GROUND_MARGIN
    above_idx  = np.where(above_mask)[0]

    above_pcd = pcd.select_by_index(above_idx)
    print(f"[ground] Above-ground points: {len(above_pcd.points):,}")
    return above_pcd, plane_model


def filter_by_colour(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Return a point cloud containing only orange or white points.
    If the cloud has no colour data, all points are returned with a warning.
    """
    if not pcd.has_colors():
        print("[colour] WARNING: no colour data – skipping colour filter.")
        return pcd

    colours = np.asarray(pcd.colors)   # shape (N, 3), values in [0, 1]
    r, g, b = colours[:, 0], colours[:, 1], colours[:, 2]

    # ── Orange mask ──────────────────────────────────────────────────────────
    orange_mask = (
        (r >= ORANGE_R_MIN) & (r <= ORANGE_R_MAX) &
        (g >= ORANGE_G_MIN) & (g <= ORANGE_G_MAX) &
        (b >= ORANGE_B_MIN) & (b <= ORANGE_B_MAX)
    )

    # ── White mask ───────────────────────────────────────────────────────────
    brightness    = (r + g + b) / 3.0
    saturation    = np.max(colours, axis=1) - np.min(colours, axis=1)
    white_mask    = (brightness >= WHITE_MIN_BRIGHTNESS) & (saturation <= WHITE_MAX_SATURATION)

    combined_mask = orange_mask | white_mask
    cone_colour_idx = np.where(combined_mask)[0]

    print(f"[colour] Orange points : {orange_mask.sum():,}")
    print(f"[colour] White  points : {white_mask.sum():,}")
    print(f"[colour] Combined      : {len(cone_colour_idx):,}")

    return pcd.select_by_index(cone_colour_idx)


def cluster_and_filter(pcd: o3d.geometry.PointCloud):
    """
    DBSCAN cluster the filtered point cloud, then apply bounding-box
    size constraints.  Returns a list of (cluster_pcd, bbox) tuples.
    """
    if len(pcd.points) == 0:
        print("[cluster] No coloured points to cluster.")
        return []

    labels = np.array(
        pcd.cluster_dbscan(
            eps        = DBSCAN_EPS,
            min_points = DBSCAN_MIN_POINTS,
            print_progress = False,
        )
    )

    max_label = labels.max() if labels.size > 0 else -1
    n_clusters = max_label + 1
    print(f"[cluster] DBSCAN found {n_clusters} cluster(s)  "
          f"(noise points: {(labels == -1).sum():,})")

    cones = []
    for lbl in range(n_clusters):
        idx     = np.where(labels == lbl)[0]
        cluster = pcd.select_by_index(idx)
        bbox    = cluster.get_axis_aligned_bounding_box()
        extent  = bbox.get_extent()            # (dx, dy, dz)

        max_ext = float(np.max(extent))
        min_ext = float(np.min(extent))

        if max_ext > MAX_CONE_DIMENSION:
            print(f"[filter]  Cluster {lbl:3d} REJECTED – too large   "
                  f"({max_ext:.3f} m > {MAX_CONE_DIMENSION} m)")
            continue
        if min_ext < MIN_CONE_DIMENSION:
            print(f"[filter]  Cluster {lbl:3d} REJECTED – too small   "
                  f"({min_ext:.3f} m < {MIN_CONE_DIMENSION} m)")
            continue

        print(f"[filter]  Cluster {lbl:3d} ACCEPTED – extent "
              f"{extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} m  "
              f"({len(cluster.points):,} pts)")
        cones.append((cluster, bbox))

    print(f"\n[result] Detected {len(cones)} cone(s) after size filtering.")
    return cones


def save_results(cones, output_dir: str):
    """Save each detected cone cluster as an individual .ply file."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (cluster, _) in enumerate(cones):
        out_path = os.path.join(output_dir, f"cone_{i:03d}.ply")
        o3d.io.write_point_cloud(out_path, cluster)
        print(f"[save]   Saved cone {i:03d}  →  {out_path}")


def visualise(original_pcd, cones, ground_model):
    """Visualise the full cloud plus colour-coded detected cones."""
    vis_objects = []

    # Original cloud in grey
    orig_vis = o3d.geometry.PointCloud(original_pcd)
    orig_vis.paint_uniform_color([0.6, 0.6, 0.6])
    vis_objects.append(orig_vis)

    # Detected cones in bright orange with bounding boxes
    cone_colour = [1.0, 0.45, 0.0]
    for cluster, bbox in cones:
        cone_vis = o3d.geometry.PointCloud(cluster)
        cone_vis.paint_uniform_color(cone_colour)
        vis_objects.append(cone_vis)

        bbox.color = (1.0, 0.0, 0.0)
        vis_objects.append(bbox)

    o3d.visualization.draw_geometries(
        vis_objects,
        window_name = "Cone Detector – Construction Site",
        width       = 1280,
        height      = 720,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────���────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Detect orange/white safety cones in a construction-site point cloud.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",
        help="Path to input point cloud file (.pcd, .ply, .las, .laz, .xyz)")
    parser.add_argument("-o", "--output-dir", default="cone_results",
        help="Directory to save individual cone point clouds")
    parser.add_argument("--no-save",  action="store_true",
        help="Skip saving detected cone clusters")
    parser.add_argument("--visualise", action="store_true",
        help="Open an interactive 3-D viewer after processing")
    parser.add_argument("--max-size", type=float, default=MAX_CONE_DIMENSION,
        help="Maximum bounding-box extent in any direction (m)")
    parser.add_argument("--min-size", type=float, default=MIN_CONE_DIMENSION,
        help="Minimum bounding-box extent in any direction (m)")
    parser.add_argument("--ground-threshold", type=float,
        default=GROUND_DISTANCE_THRESHOLD,
        help="RANSAC distance threshold for ground plane (m)")
    parser.add_argument("--above-margin", type=float,
        default=ABOVE_GROUND_MARGIN,
        help="Keep points this many metres above ground plane (m)")
    parser.add_argument("--eps", type=float, default=DBSCAN_EPS,
        help="DBSCAN neighbourhood radius (m)")
    parser.add_argument("--min-points", type=int, default=DBSCAN_MIN_POINTS,
        help="DBSCAN minimum cluster size")
    return parser.parse_args()


def main():
    args = parse_args()

    # Allow CLI overrides of module-level constants
    global MAX_CONE_DIMENSION, MIN_CONE_DIMENSION
    global GROUND_DISTANCE_THRESHOLD, ABOVE_GROUND_MARGIN
    global DBSCAN_EPS, DBSCAN_MIN_POINTS
    MAX_CONE_DIMENSION        = args.max_size
    MIN_CONE_DIMENSION        = args.min_size
    GROUND_DISTANCE_THRESHOLD = args.ground_threshold
    ABOVE_GROUND_MARGIN       = args.above_margin
    DBSCAN_EPS                = args.eps
    DBSCAN_MIN_POINTS         = args.min_points

    print("=" * 60)
    print("  Cone Detector – Construction Site Point Cloud")
    print("=" * 60)

    # 1. Load
    pcd = load_point_cloud(args.input)

    # 2. Ground removal
    above_pcd, ground_model = remove_ground(pcd)

    # 3. Colour filter (orange + white)
    colour_pcd = filter_by_colour(above_pcd)

    # 4. Cluster + size filter
    cones = cluster_and_filter(colour_pcd)

    # 5. Save
    if not args.no_save and cones:
        save_results(cones, args.output_dir)

    # 6. Visualise
    if args.visualise:
        visualise(pcd, cones, ground_model)

    print("=" * 60)
    print(f"  Done.  {len(cones)} cone(s) detected.")
    print("=" * 60)
    return 0 if cones else 1


if __name__ == "__main__":
    sys.exit(main())
