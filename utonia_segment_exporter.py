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
1.  Convert float RGB [0,1] → HSV.
2.  Build a (cos H, sin H, S, V) feature for circular-safe clustering.
3.  K-Means (default, fast) or DBSCAN (no-K fallback) on that feature.
4.  Optional: merge any two clusters whose mean hue differs < --merge_deg.
5.  Save one .ply per cluster, print AABB dimensions.
6.  Post-process each saved .ply:
      a. Statistical outlier removal (SOR).
      b. Extract the largest connected component (radius neighbour graph).
      c. Overwrite the file with the cleaned result.
      d. Re-print dimensions of the cleaned clouds.
7.  Ground-level removal: detect the dominant ground plane with RANSAC,
    then keep only points that have AT LEAST ONE neighbour also above the
    plane (connectivity filter).  Points that are isolated on the ground
    slab — i.e. they sit above it only because of noise — are discarded.
    Overwrite each .ply and report how many points were stripped.
8.  Reload each cleaned .ply, compute AABB height/length/width, print a
    formatted table, and save all measurements to segments_dimensions.json
    inside the output directory.

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

    # Tune ground removal:
    python utonia_segment_exporter.py --input scene_pca.ply \\
        --ground_ransac_dist 0.02 --ground_neighbor_radius 0.05

    # Use a fixed Z-floor instead of RANSAC:
    python utonia_segment_exporter.py --input scene_pca.ply \\
        --ground_fixed_z 0.03

Requirements
------------
    pip install open3d numpy scikit-learn
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_N_SEGMENTS     = None
DEFAULT_DBSCAN_EPS     = 0.12
DEFAULT_DBSCAN_MIN_PTS = 50
DEFAULT_MIN_POINTS     = 10
DEFAULT_MERGE_DEG      = 0.0

# Post-processing (Step 6)
DEFAULT_SOR_NEIGHBORS  = 20
DEFAULT_SOR_STD_RATIO  = 2.0
DEFAULT_CC_RADIUS      = 0.05

# Ground removal (Step 7)
DEFAULT_GROUND_RANSAC_DIST    = 0.01   # RANSAC inlier distance (scene units)
DEFAULT_GROUND_RANSAC_ITERS   = 1000   # RANSAC iterations
DEFAULT_GROUND_NEIGHBOR_RADIUS = 0.05  # radius for the above-plane connectivity check
DEFAULT_GROUND_FIXED_Z        = None   # hard Z-floor cutoff (skips RANSAC)

# Dimension JSON
DIMENSIONS_JSON = "segments_dimensions.json"


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_pointcloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        sys.exit(f"ERROR: Point cloud is empty: {path}")
    if not pcd.has_colors():
        sys.exit(
            "ERROR: The point cloud has no color information.\n"
            "Please supply the Utonia PCA-colored output (.ply with float RGB)."
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
# RGB → HSV
# ══════════════════════════════════════════════════════════════════════════════

def rgb_to_hsv(colors: np.ndarray) -> np.ndarray:
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    v    = np.maximum.reduce([r, g, b])
    s    = np.where(v > 1e-8,
                    (v - np.minimum.reduce([r, g, b])) / np.clip(v, 1e-8, None),
                    0.0)
    diff = np.clip(v - np.minimum.reduce([r, g, b]), 1e-8, None)
    h = np.where(
        v == r, (g - b) / diff % 6,
        np.where(v == g, (b - r) / diff + 2,
                          (r - g) / diff + 4)
    )
    h = h / 6.0 * 2.0 * np.pi
    return np.stack([h, s, v], axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Clustering features & hue helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_hsv_features(colors: np.ndarray, hsv_weight: float = 1.0) -> np.ndarray:
    hsv = rgb_to_hsv(colors)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    return np.stack([np.cos(h) * s * hsv_weight,
                     np.sin(h) * s * hsv_weight,
                     v], axis=-1)


def _mean_hue(colors_subset: np.ndarray) -> float:
    h = rgb_to_hsv(colors_subset)[:, 0]
    return float(np.arctan2(np.sin(h).mean(), np.cos(h).mean()) % (2 * np.pi))


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


# ══════════════════════════════════════════════════════════════════════════════
# Cluster merging
# ══════════════════════════════════════════════════════════════════════════════

def merge_close_hue_clusters(
    labels: np.ndarray,
    colors: np.ndarray,
    merge_deg: float,
) -> np.ndarray:
    if merge_deg <= 0:
        return labels
    merge_rad = np.deg2rad(merge_deg)
    unique    = [lbl for lbl in np.unique(labels) if lbl != -1]
    mean_hues = {lbl: _mean_hue(colors[labels == lbl]) for lbl in unique}
    parent    = {lbl: lbl for lbl in unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i, la in enumerate(unique):
        for lb in unique[i + 1:]:
            diff = abs(mean_hues[la] - mean_hues[lb])
            if min(diff, 2 * np.pi - diff) < merge_rad:
                union(la, lb)

    root_to_new: dict[int, int] = {}
    new_id = 0
    new_labels = labels.copy()
    for lbl in unique:
        root = find(lbl)
        if root not in root_to_new:
            root_to_new[root] = new_id; new_id += 1
        new_labels[labels == lbl] = root_to_new[root]

    n_before, n_after = len(unique), len(set(root_to_new.values()))
    if n_after < n_before:
        print(f"Merged {n_before} → {n_after} clusters (hue threshold {merge_deg:.1f}°)")
    return new_labels


# ══════════════════════════════════════════════════════════════════════════════
# Clustering
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_k(features: np.ndarray, k_min: int = 2, k_max: int = 20) -> int:
    rng     = np.random.default_rng(42)
    idx     = rng.choice(len(features), size=min(50_000, len(features)), replace=False)
    sample  = features[idx]
    inertias, ks = [], list(range(k_min, k_max + 1))
    for k in ks:
        inertias.append(
            KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
            .fit(sample).inertia_
        )
    if len(inertias) < 3:
        return k_min
    best_k = ks[int(np.argmax(np.diff(np.diff(inertias)))) + 1]
    print(f"Auto-estimated K = {best_k}")
    return best_k


def segment_by_kmeans(
    colors: np.ndarray,
    n_segments: int | None = None,
    min_points: int = DEFAULT_MIN_POINTS,
) -> np.ndarray:
    features = build_hsv_features(colors)
    k        = n_segments if n_segments is not None else _estimate_k(features)
    print(f"Running K-Means (K={k}) in HSV feature space …")
    labels   = KMeans(n_clusters=k, n_init=10, random_state=42,
                      max_iter=300).fit_predict(features).astype(np.int32)
    for lbl in np.unique(labels):
        if (labels == lbl).sum() < min_points:
            labels[labels == lbl] = -1
    print(f"K-Means segments (after noise filter): {len(np.unique(labels[labels != -1]))}")
    return labels


def segment_by_dbscan(
    colors: np.ndarray,
    eps: float = DEFAULT_DBSCAN_EPS,
    min_pts: int = DEFAULT_DBSCAN_MIN_PTS,
    min_points: int = DEFAULT_MIN_POINTS,
) -> np.ndarray:
    features = build_hsv_features(colors)
    print(f"Running DBSCAN (eps={eps}, min_samples={min_pts}) in HSV feature space …")
    labels   = DBSCAN(eps=eps, min_samples=min_pts,
                      n_jobs=-1).fit_predict(features).astype(np.int32)
    for lbl in np.unique(labels):
        if lbl != -1 and (labels == lbl).sum() < min_points:
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
) -> tuple[list[Path], list[int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    unique_labels = sorted(lbl for lbl in np.unique(labels) if lbl != -1)
    for seg_id, lbl in enumerate(unique_labels):
        mask       = labels == lbl
        seg_xyz    = xyz[mask]
        seg_colors = colors[mask]
        hue_deg    = int(np.rad2deg(_mean_hue(seg_colors)) % 360)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg_xyz)
        pcd.colors = o3d.utility.Vector3dVector(seg_colors)
        out = output_dir / f"segment_{seg_id:04d}_hue{hue_deg:03d}deg.ply"
        o3d.io.write_point_cloud(str(out), pcd)
        saved.append(out)
    print(f"\nSaved {len(saved)} segment file(s) → '{output_dir.resolve()}'")
    return saved, unique_labels


# ══════════════════════════════════════════════════════════════════════════════
# Dimension printing
# ══════════════════════════════════════════════════════════════════════════════

def print_dimensions(
    xyz: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    unique_labels: list[int],
    saved_paths: list[Path],
    header: str = "GRADIENT-COLOR SEGMENT DIMENSIONS",
) -> None:
    col_w = 86
    print("\n" + "═" * col_w)
    print(f"  {header}  ({len(unique_labels)} region(s))")
    print("═" * col_w)
    print(f"{'ID':>4}  {'Hue (name)':>16}  {'Pts':>8}  "
          f"{'Width(X)':>9}  {'Depth(Y)':>9}  {'Height(Z)':>10}  File")
    print("─" * col_w)
    for seg_id, lbl in enumerate(unique_labels):
        mask       = labels == lbl
        seg_xyz    = xyz[mask]
        seg_colors = colors[mask]
        x_min, x_max = seg_xyz[:,0].min(), seg_xyz[:,0].max()
        y_min, y_max = seg_xyz[:,1].min(), seg_xyz[:,1].max()
        z_min, z_max = seg_xyz[:,2].min(), seg_xyz[:,2].max()
        hue_deg  = np.rad2deg(_mean_hue(seg_colors)) % 360
        hue_str  = f"{hue_deg:5.1f}° ({_hue_name(hue_deg)})"
        print(
            f"{seg_id:>4}  {hue_str:>16}  {mask.sum():>8,}  "
            f"{x_max-x_min:>9.4f}  {y_max-y_min:>9.4f}  {z_max-z_min:>10.4f}  "
            f"{saved_paths[seg_id].name}"
        )
    print("═" * col_w)
    print("  Units match the coordinate units of the input point cloud.")
    print("═" * col_w)


# ══════════════════════════════════════��═══════════════════════════════════════
# Step 6 — Post-processing: SOR + largest connected component
# ══════════════════════════════════════════════════════════════════════════════

def remove_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int,
    std_ratio: float,
) -> tuple[o3d.geometry.PointCloud, int]:
    n_before  = len(pcd.points)
    clean, _  = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                std_ratio=std_ratio)
    return clean, n_before - len(clean.points)


def largest_connected_component(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    min_points: int,
) -> tuple[o3d.geometry.PointCloud, int, int]:
    if len(pcd.points) == 0:
        return pcd, 0, 0
    cc_labels = np.asarray(
        pcd.cluster_dbscan(eps=radius, min_points=1, print_progress=False)
    )
    unique_cc, counts = np.unique(cc_labels[cc_labels >= 0], return_counts=True)
    n_components      = len(unique_cc)
    if n_components == 0:
        return pcd, len(pcd.points), 0
    best_label = unique_cc[np.argmax(counts)]
    best_count = int(counts[np.argmax(counts)])
    if best_count < min_points:
        print(f"    ⚠  Largest CC has only {best_count} pts "
              f"(< min_points={min_points}); skipping.")
        return pcd, len(pcd.points), n_components
    largest = pcd.select_by_index(np.where(cc_labels == best_label)[0])
    return largest, best_count, n_components


def postprocess_segments(
    saved_paths: list[Path],
    sor_neighbors: int,
    sor_std_ratio: float,
    cc_radius: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_w = 86
    print("\n" + "═" * col_w)
    print("  STEP 6 — POST-PROCESSING: OUTLIER REMOVAL + LARGEST CONNECTED COMPONENT")
    print("═" * col_w)
    print(f"  SOR  : nb_neighbors={sor_neighbors}, std_ratio={sor_std_ratio}\n"
          f"  CC   : radius={cc_radius}, min_points={min_points}")
    print("─" * col_w)
    print(f"{'ID':>4}  {'File':<45}  {'Before':>8}  "
          f"{'−Outliers':>10}  {'−CC noise':>10}  {'After':>8}")
    print("─" * col_w)

    all_xyz: list[np.ndarray]    = []
    all_colors: list[np.ndarray] = []

    for seg_id, ply_path in enumerate(saved_paths):
        pcd      = o3d.io.read_point_cloud(str(ply_path))
        n_before = len(pcd.points)
        pcd_sor, n_sor  = remove_outliers(pcd, sor_neighbors, sor_std_ratio)
        pcd_cc, n_kept, n_comp = largest_connected_component(
            pcd_sor, cc_radius, min_points)
        n_cc = len(pcd_sor.points) - n_kept
        o3d.io.write_point_cloud(str(ply_path), pcd_cc)
        all_xyz.append(np.asarray(pcd_cc.points, dtype=np.float64))
        all_colors.append(np.asarray(pcd_cc.colors, dtype=np.float64))
        print(
            f"{seg_id:>4}  {ply_path.name:<45}  {n_before:>8,}  "
            f"{n_sor:>10,}  {n_cc:>10,}  {n_kept:>8,}"
            + (f"  [{n_comp} CC(s)]" if n_comp > 1 else "")
        )

    print("═" * col_w)
    print("  Cleaned files overwritten in place.\n")

    combined_xyz    = np.concatenate(all_xyz,    axis=0)
    combined_colors = np.concatenate(all_colors, axis=0)
    flat_labels     = np.concatenate(
        [np.full(len(x), i, dtype=np.int32) for i, x in enumerate(all_xyz)]
    )
    return combined_xyz, combined_colors, flat_labels


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Ground-level removal
# ══════════════════════════════════════════════════════════════════════════════

def _fit_ground_ransac(
    pcd: o3d.geometry.PointCloud,
    dist_threshold: float,
    num_iterations: int = DEFAULT_GROUND_RANSAC_ITERS,
) -> tuple[np.ndarray, float]:
    """
    Fit the dominant horizontal plane with RANSAC.

    Returns
    -------
    plane_model : (4,) float64  [a, b, c, d]  —  a·x + b·y + c·z + d = 0
    z_ground    : Z of the plane at the origin  (= −d/c when |c| > 0)
    """
    plane_model, _ = pcd.segment_plane(
        distance_threshold=dist_threshold,
        ransac_n=3,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model
    if abs(c) < 1e-6:
        z_ground = float(np.asarray(pcd.points)[:, 2].min())
        print("    ⚠  RANSAC found a near-vertical plane; falling back to Z_min.")
    else:
        z_ground = float(-d / c)
    return np.array(plane_model, dtype=np.float64), z_ground


def _signed_distances(xyz: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """
    Signed distance of every point in *xyz* from *plane* = [a, b, c, d].

    dist_i = (a·x_i + b·y_i + c·z_i + d) / ‖(a,b,c)‖

    Positive values are on the side the normal points toward.
    The normal is flipped so that "above ground" is always positive.
    """
    a, b, c, d = plane
    norm       = np.sqrt(a**2 + b**2 + c**2)
    dist       = (xyz @ plane[:3] + d) / norm
    # Ensure above-ground side is positive (flip if normal points downward)
    if c < 0:
        dist = -dist
    return dist


def remove_ground_from_cloud(
    pcd: o3d.geometry.PointCloud,
    ransac_dist: float,
    ransac_iters: int,
    neighbor_radius: float,
    fixed_z: float | None,
) -> tuple[o3d.geometry.PointCloud, int, float]:
    """
    Remove ground-level points, keeping only those that are
    **strictly above the plane AND have at least one neighbour that is
    also above the plane** within *neighbor_radius*.

    The second condition is the key change: a lone point floating just
    above the ground slab (noise / micro-artefact) is removed because it
    has no above-plane companion nearby, even though its signed distance
    from the plane is positive.

    Algorithm
    ---------
    1.  Determine the ground Z / plane.
            • fixed_z given  → plane is z = fixed_z  (normal = [0,0,1], d = −fixed_z)
            • otherwise      → RANSAC on the full segment
    2.  Compute signed distance d_i for every point.
    3.  Mark every point with d_i > 0 as a *candidate* (above the plane).
    4.  Build a KD-tree over the *candidate* subset only.
    5.  For each candidate query its neighbours within *neighbor_radius*
        in that same candidate-only tree.
        Keep the point iff neighbour_count ≥ 1  (i.e. at least one other
        candidate is within the radius → the point is not isolated).
    6.  Return the surviving points.

    Parameters
    ----------
    pcd             : input segment point cloud
    ransac_dist     : RANSAC inlier distance threshold (ignored when fixed_z set)
    ransac_iters    : RANSAC iterations
    neighbor_radius : search radius for the above-plane connectivity check
    fixed_z         : hard Z-floor (skips RANSAC when provided)

    Returns
    -------
    clean_pcd : cloud with ground (and isolated near-ground) points removed
    n_removed : total number of points removed
    z_floor   : Z of the detected / specified ground plane (for reporting)
    """
    n_before = len(pcd.points)
    if n_before == 0:
        return pcd, 0, 0.0

    xyz = np.asarray(pcd.points, dtype=np.float64)

    # ── 1. Determine the ground plane ─────────────────────────────────────────
    if fixed_z is not None:
        # Synthetic plane: z = fixed_z  →  0·x + 0·y + 1·z − fixed_z = 0
        plane    = np.array([0.0, 0.0, 1.0, -fixed_z], dtype=np.float64)
        z_floor  = float(fixed_z)
    else:
        plane, z_floor = _fit_ground_ransac(pcd, ransac_dist, ransac_iters)

    # ── 2. Signed distances from the plane ────────────────────────────────────
    dist = _signed_distances(xyz, plane)

    # ── 3. Candidate mask: strictly above the plane ───────────────────────────
    above_mask    = dist > 0.0                        # (N,) bool
    above_indices = np.where(above_mask)[0]           # indices into xyz

    if above_indices.size == 0:
        # Nothing is above the ground — return empty cloud
        empty = pcd.select_by_index([])
        return empty, n_before, z_floor

    # ── 4. KD-tree over candidate (above-plane) points only ───────────────────
    above_xyz  = xyz[above_indices]                   # (M, 3)
    above_pcd  = o3d.geometry.PointCloud()
    above_pcd.points = o3d.utility.Vector3dVector(above_xyz)
    kd_tree    = o3d.geometry.KDTreeFlann(above_pcd)

    # ── 5. Connectivity filter: keep point i iff ≥ 1 neighbour within radius ──
    #
    #   search_radius_vector3d returns [count, indices, distances²].
    #   We query for up to 2 hits: the point itself (distance 0) plus
    #   potentially one more.  If count ≥ 2 the point has at least one
    #   above-plane neighbour → keep it.
    #
    keep_local = np.zeros(len(above_indices), dtype=bool)
    for local_i in range(len(above_indices)):
        count, _, _ = kd_tree.search_radius_vector3d(
            above_pcd.points[local_i], neighbor_radius
        )
        # count includes the point itself, so ≥ 2 means ≥ 1 real neighbour
        if count >= 2:
            keep_local[local_i] = True

    # ── 6. Map back to original indices and build output cloud ─────────────────
    kept_global = above_indices[keep_local]
    clean_pcd   = pcd.select_by_index(kept_global)
    n_removed   = n_before - len(clean_pcd.points)
    return clean_pcd, n_removed, z_floor


def remove_ground_from_segments(
    saved_paths: list[Path],
    ransac_dist: float,
    ransac_iters: int,
    neighbor_radius: float,
    fixed_z: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 7: for every saved segment .ply —
      1. Load it.
      2. Detect the ground plane (RANSAC or fixed Z).
      3. Keep only above-plane points that have ≥ 1 above-plane neighbour
         within *neighbor_radius* (connectivity filter).
      4. Overwrite the .ply with the result.

    Returns (combined_xyz, combined_colors, flat_labels) for reporting.
    """
    col_w = 90
    print("\n" + "═" * col_w)
    print("  STEP 7 — GROUND-LEVEL REMOVAL  "
          "(above-plane points with ≥ 1 above-plane neighbour)")
    print("═" * col_w)
    if fixed_z is not None:
        print(f"  Mode            : fixed Z floor = {fixed_z:.4f}")
    else:
        print(f"  Mode            : RANSAC plane  "
              f"(dist_threshold={ransac_dist}, iters={ransac_iters})")
    print(f"  Neighbor radius : {neighbor_radius}  (connectivity check)")
    print("─" * col_w)
    print(f"{'ID':>4}  {'File':<45}  {'Before':>8}  {'Z-floor':>9}  "
          f"{'Removed':>9}  {'After':>8}")
    print("─" * col_w)

    all_xyz: list[np.ndarray]    = []
    all_colors: list[np.ndarray] = []

    for seg_id, ply_path in enumerate(saved_paths):
        pcd      = o3d.io.read_point_cloud(str(ply_path))
        n_before = len(pcd.points)

        if n_before == 0:
            print(f"{seg_id:>4}  {ply_path.name:<45}  {'EMPTY':>8}")
            all_xyz.append(np.zeros((0, 3), dtype=np.float64))
            all_colors.append(np.zeros((0, 3), dtype=np.float64))
            continue

        clean_pcd, n_removed, z_floor = remove_ground_from_cloud(
            pcd, ransac_dist, ransac_iters, neighbor_radius, fixed_z
        )

        o3d.io.write_point_cloud(str(ply_path), clean_pcd)
        all_xyz.append(np.asarray(clean_pcd.points, dtype=np.float64))
        all_colors.append(np.asarray(clean_pcd.colors, dtype=np.float64))

        print(
            f"{seg_id:>4}  {ply_path.name:<45}  {n_before:>8,}  "
            f"{z_floor:>9.4f}  {n_removed:>9,}  {len(clean_pcd.points):>8,}"
        )

    print("═" * col_w)
    print("  Ground-removed files overwritten in place.\n")

    combined_xyz    = np.concatenate(all_xyz,    axis=0)
    combined_colors = np.concatenate(all_colors, axis=0)
    flat_labels     = np.concatenate(
        [np.full(len(x), i, dtype=np.int32) for i, x in enumerate(all_xyz)]
    )
    return combined_xyz, combined_colors, flat_labels


# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Measure & save dimensions
# ══════════════════════════════════════════════════════════════════════════════

def _aabb_dims(xyz: np.ndarray) -> dict[str, float]:
    x_min, x_max = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    y_min, y_max = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    z_min, z_max = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    return {
        "length_x": round(x_max - x_min, 6),
        "width_y":  round(y_max - y_min, 6),
        "height_z": round(z_max - z_min, 6),
        "x_min": round(x_min, 6), "x_max": round(x_max, 6),
        "y_min": round(y_min, 6), "y_max": round(y_max, 6),
        "z_min": round(z_min, 6), "z_max": round(z_max, 6),
    }


def measure_and_save_dimensions(
    saved_paths: list[Path],
    output_dir: Path,
    source_file: str,
) -> Path:
    col_w = 96
    print("\n" + "═" * col_w)
    print("  STEP 8 — FINAL DIMENSIONS (ground-removed, cleaned point clouds)")
    print("═" * col_w)
    print(
        "  Axis convention:  Length = X extent  │  Width = Y extent  │  Height = Z extent\n"
        "  Units: same coordinate units as the input point cloud."
    )
    print("─" * col_w)
    print(
        f"{'ID':>4}  {'File':<48}  {'Pts':>8}  "
        f"{'Length(X)':>10}  {'Width(Y)':>10}  {'Height(Z)':>10}  {'Hue':>12}"
    )
    print("─" * col_w)

    records: list[dict] = []

    for seg_id, ply_path in enumerate(saved_paths):
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if len(pcd.points) == 0:
            print(f"{seg_id:>4}  {ply_path.name:<48}  {'EMPTY — skipped':>8}")
            records.append({
                "segment_id": seg_id, "file": ply_path.name,
                "n_points": 0, "hue_deg": None, "hue_name": None,
                "length_x": None, "width_y": None, "height_z": None, "aabb": None,
            })
            continue

        xyz    = np.asarray(pcd.points, dtype=np.float64)
        colors = np.asarray(pcd.colors, dtype=np.float64)
        dims   = _aabb_dims(xyz)

        hue_deg = round(float(np.rad2deg(_mean_hue(colors)) % 360), 2)
        hue_str = f"{hue_deg:5.1f}° {_hue_name(hue_deg)}"

        print(
            f"{seg_id:>4}  {ply_path.name:<48}  {len(xyz):>8,}  "
            f"{dims['length_x']:>10.4f}  {dims['width_y']:>10.4f}  "
            f"{dims['height_z']:>10.4f}  {hue_str:>12}"
        )

        records.append({
            "segment_id": seg_id,
            "file":       ply_path.name,
            "n_points":   int(len(xyz)),
            "hue_deg":    hue_deg,
            "hue_name":   _hue_name(hue_deg),
            "length_x":   dims["length_x"],
            "width_y":    dims["width_y"],
            "height_z":   dims["height_z"],
            "aabb": {
                "x_min": dims["x_min"], "x_max": dims["x_max"],
                "y_min": dims["y_min"], "y_max": dims["y_max"],
                "z_min": dims["z_min"], "z_max": dims["z_max"],
            },
        })

    print("═" * col_w)

    payload = {
        "metadata": {
            "source_file": source_file,
            "n_segments":  len(records),
            "axis_convention": {
                "length": "X-axis extent (x_max - x_min)",
                "width":  "Y-axis extent (y_max - y_min)",
                "height": "Z-axis extent (z_max - z_min)",
            },
            "units": "same as input point cloud coordinate units",
        },
        "segments": records,
    }

    json_path = output_dir / DIMENSIONS_JSON
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n  Dimensions saved → '{json_path.resolve()}'")
    print("═" * col_w)
    return json_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════��══════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export gradient-color regions from a Utonia PCA-colored point cloud.\n\n"
            "Examples:\n"
            "  python utonia_segment_exporter.py --input scene.ply\n"
            "  python utonia_segment_exporter.py --input scene.ply --n_segments 8\n"
            "  python utonia_segment_exporter.py --input scene.ply --method dbscan --eps 0.1\n"
            "  python utonia_segment_exporter.py --input scene.ply --merge_deg 15\n"
            "  python utonia_segment_exporter.py --input scene.ply \\\n"
            "      --sor_neighbors 30 --sor_std_ratio 1.5 --cc_radius 0.05\n"
            "  python utonia_segment_exporter.py --input scene.ply \\\n"
            "      --ground_ransac_dist 0.02 --ground_neighbor_radius 0.05\n"
            "  python utonia_segment_exporter.py --input scene.ply \\\n"
            "      --ground_fixed_z 0.03\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",      "-i", required=True)
    p.add_argument("--output_dir", "-o", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--method",     "-M", default=DEFAULT_METHOD,
                   choices=["kmeans", "dbscan"])
    p.add_argument("--n_segments", "-k", type=int,   default=DEFAULT_N_SEGMENTS)
    p.add_argument("--eps",        "-e", type=float, default=DEFAULT_DBSCAN_EPS)
    p.add_argument("--min_pts",          type=int,   default=DEFAULT_DBSCAN_MIN_PTS)
    p.add_argument("--min_points", "-m", type=int,   default=DEFAULT_MIN_POINTS)
    p.add_argument("--merge_deg",        type=float, default=DEFAULT_MERGE_DEG)

    g6 = p.add_argument_group("post-processing — Step 6")
    g6.add_argument("--sor_neighbors", type=int,   default=DEFAULT_SOR_NEIGHBORS)
    g6.add_argument("--sor_std_ratio", type=float, default=DEFAULT_SOR_STD_RATIO)
    g6.add_argument("--cc_radius",     type=float, default=DEFAULT_CC_RADIUS)

    g7 = p.add_argument_group("ground removal — Step 7")
    g7.add_argument(
        "--ground_ransac_dist", type=float, default=DEFAULT_GROUND_RANSAC_DIST,
        help="RANSAC inlier distance threshold in scene units (default: %(default)s). "
             "Ignored when --ground_fixed_z is set.",
    )
    g7.add_argument(
        "--ground_ransac_iters", type=int, default=DEFAULT_GROUND_RANSAC_ITERS,
        help="Number of RANSAC iterations (default: %(default)s).",
    )
    g7.add_argument(
        "--ground_neighbor_radius", type=float, default=DEFAULT_GROUND_NEIGHBOR_RADIUS,
        metavar="R",
        help="Radius for the above-plane connectivity check (scene units, "
             "default: %(default)s).  A point is kept only if at least one "
             "other above-plane point lies within this radius.  "
             "Increase for sparser clouds; decrease for fine detail.",
    )
    g7.add_argument(
        "--ground_fixed_z", type=float, default=DEFAULT_GROUND_FIXED_Z,
        metavar="Z",
        help="Skip RANSAC and treat z = Z as the ground plane.  "
             "All points with signed distance ≤ 0 (i.e. z ≤ Z) are removed "
             "before the connectivity filter is applied.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: File not found: {input_path.resolve()}")

    # 1. Load
    xyz, colors = load_pointcloud(input_path)

    # 2. Cluster
    if args.method == "kmeans":
        labels = segment_by_kmeans(colors, args.n_segments, args.min_points)
    else:
        labels = segment_by_dbscan(colors, args.eps, args.min_pts, args.min_points)

    # 3. Optional hue merging
    if args.merge_deg > 0:
        labels = merge_close_hue_clusters(labels, colors, args.merge_deg)

    if len(np.unique(labels[labels != -1])) == 0:
        sys.exit(
            "No valid segments found.\n"
            "  • K-Means : try a different --n_segments value.\n"
            "  • DBSCAN  : try a smaller --eps or --min_pts value.\n"
            "  • Both    : try lowering --min_points."
        )

    # 4. Save raw segments
    saved_paths, unique_labels = save_segments(
        xyz, colors, labels, Path(args.output_dir)
    )

    # 5. Print raw dimensions
    print_dimensions(xyz, colors, labels, unique_labels, saved_paths,
                     header="RAW SEGMENT DIMENSIONS (before post-processing)")

    # 6. SOR + largest CC → overwrite .ply files
    clean_xyz, clean_colors, clean_labels = postprocess_segments(
        saved_paths,
        sor_neighbors=args.sor_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        cc_radius=args.cc_radius,
        min_points=args.min_points,
    )

    clean_unique = sorted(set(clean_labels.tolist()))
    print_dimensions(
        clean_xyz, clean_colors, clean_labels, clean_unique, saved_paths,
        header="CLEANED SEGMENT DIMENSIONS (after outlier removal + CC extraction)",
    )

    # 7. Ground-level removal → overwrite .ply files
    dg_xyz, dg_colors, dg_labels = remove_ground_from_segments(
        saved_paths,
        ransac_dist=args.ground_ransac_dist,
        ransac_iters=args.ground_ransac_iters,
        neighbor_radius=args.ground_neighbor_radius,
        fixed_z=args.ground_fixed_z,
    )

    dg_unique = sorted(set(dg_labels.tolist()))
    print_dimensions(
        dg_xyz, dg_colors, dg_labels, dg_unique, saved_paths,
        header="DE-GROUNDED SEGMENT DIMENSIONS (after ground removal)",
    )

    # 8. Reload cleaned files → measure AABB → print + save JSON
    measure_and_save_dimensions(
        saved_paths,
        output_dir=Path(args.output_dir),
        source_file=input_path.name,
    )


if __name__ == "__main__":
    main()
