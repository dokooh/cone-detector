#!/usr/bin/env python3
"""
cone_detector_color.py

Detects traffic/safety cones (orange and white) in a point cloud from a
construction dig site. Pipeline:
  1. Load point cloud (supports .pcd, .ply, .las/.laz, .xyz)
  2. Estimate and remove the ground plane (RANSAC)
  3. Keep only above-ground points
  4. Filter by orange / white colour separately
  5. Merge orange+white clusters that are spatially close (proximity check)
  6. Apply a bounding-box size threshold (max 0.8 m in any direction)
  7. Save all accepted cones as ONE combined .ply file

Dependencies:
    pip install open3d numpy
    pip install "laspy[lazrs]"   # only needed for .las/.laz input
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
DBSCAN_EPS        = 0.10   # Neighbourhood radius (m)
DBSCAN_MIN_POINTS = 15     # Min points to form a cluster

# Size filter
MAX_CONE_DIMENSION = 0.80   # Maximum extent in any direction (m)
MIN_CONE_DIMENSION = 0.05   # Minimum extent in any direction (m)

# Proximity merge
MERGE_DISTANCE_THRESHOLD = 0.30   # Max centroid-to-centroid distance (m)
                                   # to merge an orange+white cluster pair


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────��───────

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from various formats, normalising colours to [0, 1]."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".pcd", ".ply"):
        pcd = o3d.io.read_point_cloud(path)

    elif ext in (".las", ".laz"):
        try:
            import laspy
        except ImportError:
            sys.exit("laspy is required for .las/.laz files.\n"
                     "Install with:  pip install \"laspy[lazrs]\"")
        las = laspy.read(path)
        xyz = np.vstack((las.x, las.y, las.z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
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
            if colours.max() > 1.0:
                colours = colours / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colours)

    else:
        sys.exit(f"Unsupported file format: '{ext}'.  "
                 f"Supported: .pcd  .ply  .las  .laz  .xyz")

    if not pcd.has_points():
        sys.exit(f"No points loaded from '{path}'.")

    print(f"[load]    Loaded {len(pcd.points):,} points from '{path}'")
    return pcd


def remove_ground(pcd: o3d.geometry.PointCloud):
    """
    Estimate the dominant ground plane with RANSAC.
    Returns the above-ground point cloud and the plane model [a,b,c,d].
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold = GROUND_DISTANCE_THRESHOLD,
        ransac_n           = GROUND_RANSAC_N,
        num_iterations     = GROUND_RANSAC_ITER,
    )
    a, b, c, d = plane_model
    print(f"[ground]  Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0  "
          f"({len(inliers):,} inliers)")

    pts  = np.asarray(pcd.points)
    norm = np.sqrt(a**2 + b**2 + c**2)
    signed_dist = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / norm

    # Ensure positive == above ground
    if c < 0:
        signed_dist = -signed_dist

    above_idx = np.where(signed_dist > ABOVE_GROUND_MARGIN)[0]
    above_pcd = pcd.select_by_index(above_idx)
    print(f"[ground]  Above-ground points: {len(above_pcd.points):,}")
    return above_pcd, plane_model


def _colour_masks(pcd: o3d.geometry.PointCloud):
    """
    Return (orange_mask, white_mask) boolean arrays over pcd points.
    If the cloud has no colours both masks are all-True (no filtering).
    """
    if not pcd.has_colors():
        n = len(pcd.points)
        print("[colour]  WARNING: no colour data – treating all points as candidates.")
        return np.ones(n, dtype=bool), np.ones(n, dtype=bool)

    colours = np.asarray(pcd.colors)          # (N, 3) in [0, 1]
    r, g, b = colours[:, 0], colours[:, 1], colours[:, 2]

    orange_mask = (
        (r >= ORANGE_R_MIN) & (r <= ORANGE_R_MAX) &
        (g >= ORANGE_G_MIN) & (g <= ORANGE_G_MAX) &
        (b >= ORANGE_B_MIN) & (b <= ORANGE_B_MAX)
    )

    brightness   = (r + g + b) / 3.0
    saturation   = np.max(colours, axis=1) - np.min(colours, axis=1)
    white_mask   = (brightness >= WHITE_MIN_BRIGHTNESS) & (saturation <= WHITE_MAX_SATURATION)

    print(f"[colour]  Orange points : {orange_mask.sum():,}")
    print(f"[colour]  White  points : {white_mask.sum():,}")
    return orange_mask, white_mask


def _dbscan_clusters(pcd: o3d.geometry.PointCloud) -> list:
    """
    Run DBSCAN on pcd and return a list of open3d.geometry.PointCloud objects,
    one per cluster (noise label -1 is discarded).
    """
    if len(pcd.points) == 0:
        return []

    labels = np.array(
        pcd.cluster_dbscan(
            eps            = DBSCAN_EPS,
            min_points     = DBSCAN_MIN_POINTS,
            print_progress = False,
        )
    )
    clusters = []
    for lbl in range(labels.max() + 1):
        idx = np.where(labels == lbl)[0]
        clusters.append(pcd.select_by_index(idx))
    return clusters


def _centroid(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points).mean(axis=0)


def detect_and_merge(above_pcd: o3d.geometry.PointCloud) -> list:
    """
    Core detection logic:
      1. Split above-ground cloud into orange / white sub-clouds.
      2. DBSCAN-cluster each colour independently.
      3. For every orange cluster, check if any white cluster centroid
         is within MERGE_DISTANCE_THRESHOLD → merge them (cone body + stripe).
      4. Unmatched orange clusters (no nearby white) are kept as-is.
      5. White clusters that were NOT merged are silently discarded
         (reflective ground markings, helmets, etc.).
      6. Apply bounding-box size filter.
    Returns a list of accepted cone PointClouds.
    """
    orange_mask, white_mask = _colour_masks(above_pcd)

    orange_pcd = above_pcd.select_by_index(np.where(orange_mask)[0])
    white_pcd  = above_pcd.select_by_index(np.where(white_mask)[0])

    orange_clusters = _dbscan_clusters(orange_pcd)
    white_clusters  = _dbscan_clusters(white_pcd)

    print(f"[cluster] Orange clusters: {len(orange_clusters)}  |  "
          f"White clusters: {len(white_clusters)}")

    # Pre-compute white centroids
    white_centroids = [_centroid(wc) for wc in white_clusters]
    white_used      = [False] * len(white_clusters)

    merged_candidates = []

    for oc in orange_clusters:
        oc_centroid = _centroid(oc)
        nearby_white = []

        for wi, wc in enumerate(white_clusters):
            dist = float(np.linalg.norm(oc_centroid - white_centroids[wi]))
            if dist <= MERGE_DISTANCE_THRESHOLD:
                nearby_white.append(wi)
                white_used[wi] = True

        if nearby_white:
            # Merge orange + all nearby white clusters
            parts = [oc] + [white_clusters[wi] for wi in nearby_white]
            merged = _combine(parts)
            print(f"[merge]   Orange cluster merged with "
                  f"{len(nearby_white)} white cluster(s)  "
                  f"→ {len(merged.points):,} pts")
        else:
            merged = oc
            print(f"[merge]   Orange cluster kept solo  "
                  f"({len(merged.points):,} pts, no nearby white)")

        merged_candidates.append(merged)

    # ── Size filter ───────────────────────────────────────────────────────────
    accepted = []
    for i, candidate in enumerate(merged_candidates):
        bbox   = candidate.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_ext = float(np.max(extent))
        min_ext = float(np.min(extent))

        if max_ext > MAX_CONE_DIMENSION:
            print(f"[filter]  Candidate {i:3d} REJECTED – too large  "
                  f"({max_ext:.3f} m > {MAX_CONE_DIMENSION} m)")
            continue
        if min_ext < MIN_CONE_DIMENSION:
            print(f"[filter]  Candidate {i:3d} REJECTED – too small  "
                  f"({min_ext:.3f} m < {MIN_CONE_DIMENSION} m)")
            continue

        print(f"[filter]  Candidate {i:3d} ACCEPTED – "
              f"{extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} m  "
              f"({len(candidate.points):,} pts)")
        accepted.append(candidate)

    unmerged_white = sum(1 for u in white_used if not u)
    if unmerged_white:
        print(f"[merge]   {unmerged_white} white cluster(s) had no nearby orange – discarded.")

    print(f"\n[result]  {len(accepted)} cone(s) accepted after size filtering.")
    return accepted


def _combine(pcds: list) -> o3d.geometry.PointCloud:
    """Concatenate a list of PointClouds into one."""
    all_pts  = np.vstack([np.asarray(p.points)  for p in pcds])
    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(all_pts)

    if all(p.has_colors() for p in pcds):
        all_col = np.vstack([np.asarray(p.colors) for p in pcds])
        combined.colors = o3d.utility.Vector3dVector(all_col)

    return combined


def save_combined(cones: list, output_path: str):
    """Merge ALL accepted cone clusters into one .ply file and save it."""
    if not cones:
        print("[save]    No cones to save.")
        return

    combined = _combine(cones)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    o3d.io.write_point_cloud(output_path, combined)
    print(f"[save]    {len(cones)} cone(s) → {len(combined.points):,} pts  "
          f"saved to '{output_path}'")


def visualise(original_pcd, cones):
    """Interactive 3-D viewer: grey scene + orange cones with red bounding boxes."""
    vis_objects = []

    scene = o3d.geometry.PointCloud(original_pcd)
    scene.paint_uniform_color([0.6, 0.6, 0.6])
    vis_objects.append(scene)

    for cone in cones:
        cone_vis = o3d.geometry.PointCloud(cone)
        cone_vis.paint_uniform_color([1.0, 0.45, 0.0])
        vis_objects.append(cone_vis)

        bbox       = cone.get_axis_aligned_bounding_box()
        bbox.color = (1.0, 0.0, 0.0)
        vis_objects.append(bbox)

    o3d.visualization.draw_geometries(
        vis_objects,
        window_name = "Cone Detector – Construction Site",
        width=1280, height=720,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect orange/white safety cones in a construction-site point cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",
        help="Input point cloud (.pcd, .ply, .las, .laz, .xyz)")
    parser.add_argument("-o", "--output", default="cone_results/cones_detected.ply",
        help="Output .ply file path (all detected cones merged into one file)")
    parser.add_argument("--no-save",  action="store_true",
        help="Skip saving the output file")
    parser.add_argument("--visualise", action="store_true",
        help="Open an interactive 3-D viewer after processing")

    # Overridable thresholds
    parser.add_argument("--max-size",          type=float, default=MAX_CONE_DIMENSION)
    parser.add_argument("--min-size",          type=float, default=MIN_CONE_DIMENSION)
    parser.add_argument("--ground-threshold",  type=float, default=GROUND_DISTANCE_THRESHOLD)
    parser.add_argument("--above-margin",      type=float, default=ABOVE_GROUND_MARGIN)
    parser.add_argument("--eps",               type=float, default=DBSCAN_EPS)
    parser.add_argument("--min-points",        type=int,   default=DBSCAN_MIN_POINTS)
    parser.add_argument("--merge-distance",    type=float, default=MERGE_DISTANCE_THRESHOLD,
        help="Max centroid distance (m) to merge an orange+white cluster pair")
    return parser.parse_args()


def main():
    args = parse_args()

    global MAX_CONE_DIMENSION, MIN_CONE_DIMENSION
    global GROUND_DISTANCE_THRESHOLD, ABOVE_GROUND_MARGIN
    global DBSCAN_EPS, DBSCAN_MIN_POINTS, MERGE_DISTANCE_THRESHOLD

    MAX_CONE_DIMENSION        = args.max_size
    MIN_CONE_DIMENSION        = args.min_size
    GROUND_DISTANCE_THRESHOLD = args.ground_threshold
    ABOVE_GROUND_MARGIN       = args.above_margin
    DBSCAN_EPS                = args.eps
    DBSCAN_MIN_POINTS         = args.min_points
    MERGE_DISTANCE_THRESHOLD  = args.merge_distance

    print("=" * 60)
    print("  Cone Detector – Construction Site Point Cloud")
    print("=" * 60)

    pcd       = load_point_cloud(args.input)
    above_pcd, _ = remove_ground(pcd)
    cones     = detect_and_merge(above_pcd)

    if not args.no_save:
        save_combined(cones, args.output)

    if args.visualise:
        visualise(pcd, cones)

    print("=" * 60)
    print(f"  Done.  {len(cones)} cone(s) detected.")
    print("=" * 60)
    return 0 if cones else 1


if __name__ == "__main__":
    sys.exit(main())
