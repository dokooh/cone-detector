#!/usr/bin/env python3
"""
ColorObjectsDetector.py

Detects orange-and-white objects (e.g. traffic cones, barricades) in a
point cloud from a construction site.

Pipeline:
  1. Load point cloud  (.pcd  .ply  .las/.laz  .xyz)
  2. Estimate and remove the ground plane (RANSAC)
  3. Keep only above-ground points
  4. Filter by orange / white colour
  5. Cluster the remaining points (DBSCAN)
  6. For every cluster: measure H × W × L in cm and feet, save as .ply
  7. Render a 2-D image per cluster, run Grounding-DINO (SAM3) zero-shot
     detection with prompts ['cone', 'barricade']; rename matched clusters.

Dependencies:
    pip install open3d numpy laspy lazrs-python pillow
    pip install torch torchvision
    pip install transformers   # for Grounding-DINO via HuggingFace
"""

import argparse
import os
import sys
import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d is required.  Install with:  pip install open3d")

# ──────────────────────────────────────────────────────────────────────────────
# TUNEABLE PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

# Ground removal (RANSAC)
GROUND_DISTANCE_THRESHOLD = 0.10   # inlier tolerance (m)
GROUND_RANSAC_N           = 3      # min points per RANSAC sample
GROUND_RANSAC_ITER        = 2000   # iterations
ABOVE_GROUND_MARGIN       = 0.05   # keep points > this height above ground (m)

# Colour thresholds — RGB in [0, 1]
# ── Orange ──────────────────────────────────────────────────────────────────
# Brown vs orange separation:
#   Brown:  high R, very low G (G/R < 0.35), low B
#   Orange: high R, moderate G (G/R ≥ 0.35), low B
ORANGE_R_MIN, ORANGE_R_MAX = 0.55, 1.00
ORANGE_G_MIN, ORANGE_G_MAX = 0.28, 0.65   # raised from 0.20 to exclude brown
ORANGE_B_MIN, ORANGE_B_MAX = 0.00, 0.30   # tightened from 0.35
ORANGE_MIN_G_OVER_R        = 0.35         # G/R ratio floor – rejects brown
ORANGE_MAX_G_OVER_R        = 0.75         # G/R ratio ceiling – rejects yellow
# ── White ───────────────────────────────────────────────────────────────────
WHITE_MIN_BRIGHTNESS  = 0.70   # all channels ≥ this value
WHITE_MAX_SATURATION  = 0.20   # max(RGB) − min(RGB) ≤ this value

# DBSCAN clustering
DBSCAN_EPS        = 0.10   # neighbourhood radius (m)
DBSCAN_MIN_POINTS = 15     # min points to form a cluster

# Size filter (metres)
MIN_OBJ_DIMENSION = 0.03   # reject clusters smaller than this in every axis (m)
MAX_OBJ_DIMENSION = 5.00   # reject clusters larger than this in any axis (m)
MIN_OBJ_HEIGHT    = 0.10   # reject clusters shorter than this in Z (m) – ~10 cm

# Classification (Grounding-DINO / SAM3)
DETECTION_PROMPTS    = ["cone", "barricade"]
DETECTION_MODEL_ID   = "IDEA-Research/grounding-dino-tiny"   # HuggingFace model id
DETECTION_BOX_THRESH = 0.30   # confidence threshold for a detection to count
RENDER_IMAGE_SIZE    = 512    # pixel size of the rendered cluster image (square)

# ──────────────────────────────────────────────────────────────────────────────
# UNIT CONVERSION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

M_TO_CM   = 100.0
M_TO_FEET = 3.28084


def metres_to_str(value_m: float) -> str:
    """Return a human-readable string with cm and feet."""
    return f"{value_m * M_TO_CM:.1f} cm  ({value_m * M_TO_FEET:.3f} ft)"


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """
    Load a point cloud from .pcd / .ply / .las / .laz / .xyz.
    Colours are normalised to [0, 1].
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".pcd", ".ply"):
        pcd = o3d.io.read_point_cloud(path)

    elif ext in (".las", ".laz"):
        try:
            import laspy
        except ImportError:
            sys.exit("laspy is required for .las/.laz.  pip install laspy lazrs-python")
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
            colours = data[:, 3:6].astype(np.float64)
            if colours.max() > 1.0:
                colours /= 255.0
            pcd.colors = o3d.utility.Vector3dVector(colours)

    else:
        sys.exit(f"Unsupported format '{ext}'.  Supported: .pcd .ply .las .laz .xyz")

    if not pcd.has_points():
        sys.exit(f"No points loaded from '{path}'.")

    print(f"[load]   {len(pcd.points):,} points  ←  '{path}'")
    return pcd


# ──────────────────────────────────────────────────────────────────────────────
# GROUND REMOVAL
# ──────────────────────────────────────────────────────────────────────────────

def remove_ground(pcd: o3d.geometry.PointCloud):
    """
    Fit the dominant ground plane with RANSAC.

    Returns
    -------
    above_pcd   : PointCloud  – points strictly above the ground
    plane_model : (a, b, c, d) plane equation
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=GROUND_DISTANCE_THRESHOLD,
        ransac_n=GROUND_RANSAC_N,
        num_iterations=GROUND_RANSAC_ITER,
    )
    a, b, c, d = plane_model
    print(f"[ground] Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0  "
          f"({len(inliers):,} inliers)")

    pts  = np.asarray(pcd.points)
    norm = np.sqrt(a**2 + b**2 + c**2)
    signed_dist = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / norm

    # Make sure the normal points upward (positive = above ground)
    if c < 0:
        signed_dist = -signed_dist

    above_idx = np.where(signed_dist > ABOVE_GROUND_MARGIN)[0]
    above_pcd = pcd.select_by_index(above_idx)
    print(f"[ground] Above-ground points kept: {len(above_pcd.points):,}")
    return above_pcd, plane_model


# ────────────────────────────��─────────────────────────────────────────────────
# COLOUR FILTERING
# ──────────────────────────────────────────────────────────────────────────────

def filter_by_colour(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Keep only orange and white points.
    Orange is distinguished from brown via a G/R ratio guard:
      - Brown:  G/R < 0.35  (green channel very suppressed relative to red)
      - Orange: 0.35 ≤ G/R ≤ 0.75
    If the cloud carries no colour data, all points are kept (with a warning).
    """
    if not pcd.has_colors():
        print("[colour] WARNING: no colour data – skipping colour filter.")
        return pcd

    colours = np.asarray(pcd.colors)          # (N, 3) in [0, 1]
    r, g, b = colours[:, 0], colours[:, 1], colours[:, 2]

    # Avoid division by zero for near-black points
    r_safe   = np.where(r > 0.01, r, 0.01)
    g_over_r = g / r_safe

    # ── Orange ───────────────────────────────────────────────────────────────
    orange_box = (
        (r >= ORANGE_R_MIN) & (r <= ORANGE_R_MAX) &
        (g >= ORANGE_G_MIN) & (g <= ORANGE_G_MAX) &
        (b >= ORANGE_B_MIN) & (b <= ORANGE_B_MAX)
    )
    # G/R ratio guard separates orange from brown (too low) and yellow (too high)
    orange_ratio = (
        (g_over_r >= ORANGE_MIN_G_OVER_R) &
        (g_over_r <= ORANGE_MAX_G_OVER_R)
    )
    orange_mask = orange_box & orange_ratio

    # ── White ────────────────────────────────────────────────────────────────
    brightness = (r + g + b) / 3.0
    saturation = np.max(colours, axis=1) - np.min(colours, axis=1)
    white_mask = (brightness >= WHITE_MIN_BRIGHTNESS) & (saturation <= WHITE_MAX_SATURATION)

    combined_idx = np.where(orange_mask | white_mask)[0]

    print(f"[colour] Orange : {orange_mask.sum():,} pts  "
          f"(box={orange_box.sum():,}  after G/R ratio={orange_mask.sum():,})")
    print(f"[colour] White  : {white_mask.sum():,} pts")
    print(f"[colour] Total  : {len(combined_idx):,} pts kept after colour filter")

    return pcd.select_by_index(combined_idx)


# ──────────────────────────────────────────────────────────────────────────────
# CLUSTERING + MEASUREMENT
# ──────────────────────────────────────────────────────────────────────────────

def _compute_centroid(cluster: o3d.geometry.PointCloud) -> np.ndarray:
    """Return the mean XYZ position of a cluster as a (3,) array."""
    return np.asarray(cluster.points).mean(axis=0)


def _print_dimensions(label: str, extent, centroid: np.ndarray, n_pts: int):
    """Pretty-print an object's dimensions and location."""
    dx, dy, dz = float(extent[0]), float(extent[1]), float(extent[2])
    width  = min(dx, dy)
    length = max(dx, dy)
    height = dz
    cx, cy, cz = centroid

    print(f"  {label}")
    print(f"    Points   : {n_pts:,}")
    print(f"    Height   : {metres_to_str(height)}")
    print(f"    Width    : {metres_to_str(width)}")
    print(f"    Length   : {metres_to_str(length)}")
    print(f"    Location : X={cx:.4f} m   Y={cy:.4f} m   Z={cz:.4f} m  "
          f"(centroid)")
    return height, width, length


def cluster_and_measure(pcd: o3d.geometry.PointCloud):
    """
    DBSCAN → size filter → height filter → measure each surviving cluster.

    Returns
    -------
    List of dicts with keys:
        cluster            – PointCloud
        bbox               – AxisAlignedBoundingBox
        height_m           – float (metres)
        width_m            – float (metres)
        length_m           – float (metres)
        centroid           – np.ndarray shape (3,)  [X, Y, Z in metres]
        label              – int  (DBSCAN cluster index)
        index              – int  (accepted-object counter)
        class_name         – str  (set to "object" here; updated by classify_objects)
        confidence         – float (0.0 until classify_objects runs)
    """
    if len(pcd.points) == 0:
        print("[cluster] No coloured points to cluster.")
        return []

    labels = np.array(
        pcd.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=False,
        )
    )

    n_clusters = int(labels.max()) + 1 if labels.size > 0 else 0
    print(f"[cluster] DBSCAN: {n_clusters} cluster(s)  "
          f"(noise: {(labels == -1).sum():,} pts)")

    objects = []
    accepted = 0

    for lbl in range(n_clusters):
        idx     = np.where(labels == lbl)[0]
        cluster = pcd.select_by_index(idx)
        bbox    = cluster.get_axis_aligned_bounding_box()
        extent  = bbox.get_extent()          # numpy array [dx, dy, dz]

        max_ext  = float(np.max(extent))
        min_ext  = float(np.min(extent))
        height_m = float(extent[2])          # Z axis = height

        # ── Size checks ───────────────────────────────────────────────────────
        if max_ext > MAX_OBJ_DIMENSION:
            print(f"[filter] Cluster {lbl:3d}  REJECTED  too large   "
                  f"max={max_ext*100:.1f} cm  (limit={MAX_OBJ_DIMENSION*100:.0f} cm)")
            continue

        if min_ext < MIN_OBJ_DIMENSION:
            print(f"[filter] Cluster {lbl:3d}  REJECTED  too small   "
                  f"min={min_ext*100:.1f} cm  (limit={MIN_OBJ_DIMENSION*100:.0f} cm)")
            continue

        # ── Height check ──────────────────────────────────────────────────────
        if height_m < MIN_OBJ_HEIGHT:
            print(f"[filter] Cluster {lbl:3d}  REJECTED  too short   "
                  f"height={height_m*100:.1f} cm  (min={MIN_OBJ_HEIGHT*100:.0f} cm)")
            continue

        dx, dy, dz = float(extent[0]), float(extent[1]), float(extent[2])
        width_m    = min(dx, dy)
        length_m   = max(dx, dy)
        centroid   = _compute_centroid(cluster)

        print(f"[filter] Cluster {lbl:3d}  ACCEPTED  →  Object #{accepted}")
        _print_dimensions(
            f"Object #{accepted}  (cluster {lbl})", extent, centroid, len(idx)
        )

        objects.append(dict(
            label      = lbl,
            index      = accepted,
            cluster    = cluster,
            bbox       = bbox,
            height_m   = height_m,
            width_m    = width_m,
            length_m   = length_m,
            centroid   = centroid,
            class_name = "object",   # placeholder; updated in classify_objects()
            confidence = 0.0,
        ))
        accepted += 1

    print(f"\n[result] {len(objects)} object(s) detected after size filtering.")
    return objects


# ──────────────────────────────────────────────────────────────────────────────
# 2-D RENDERING
# ──────────────────────────────────────────────────────────────────────────────

def render_cluster_to_image(cluster: o3d.geometry.PointCloud,
                             size: int = RENDER_IMAGE_SIZE):
    """
    Project a single cluster onto a 2-D top-down image (XY plane).

    Strategy
    --------
    • Normalise XY to [0, size-1] pixel coordinates.
    • Colour each pixel with the point's RGB (or solid orange if no colours).
    • Return a PIL Image (RGB).

    The top-down view reliably shows the footprint + colours for both cones
    and barricades and does not require an OpenGL window.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        sys.exit("Pillow is required for rendering.  pip install pillow")

    pts = np.asarray(cluster.points)          # (N, 3)

    if cluster.has_colors():
        cols = (np.asarray(cluster.colors) * 255).astype(np.uint8)   # (N, 3)
    else:
        cols = np.full((len(pts), 3), [255, 115, 0], dtype=np.uint8)  # orange

    # Normalise XY to [0, size-1]
    xy   = pts[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1.0   # avoid /0 for degenerate cases

    px = ((xy - mins) / span * (size - 1)).astype(int)

    # Build image — later points overwrite earlier ones (fine for our use case)
    img_arr = np.full((size, size, 3), 40, dtype=np.uint8)   # dark grey background
    img_arr[size - 1 - px[:, 1], px[:, 0]] = cols            # flip Y so up=up

    # FIX 1: drop the deprecated `mode=` kwarg; uint8 (H,W,3) is auto-detected as RGB
    return PILImage.fromarray(img_arr)


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION  (Grounding-DINO / SAM3 via HuggingFace)
# ──────────────────────────────────────────────────────────────────────────────

def _load_grounding_dino():
    """
    Load Grounding-DINO (the open-vocabulary detector used inside SAM3)
    from HuggingFace Transformers.  Downloads weights on first run (~700 MB).
    """
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
    except ImportError:
        sys.exit(
            "transformers and torch are required for classification.\n"
            "  pip install torch torchvision transformers"
        )

    print(f"[classify] Loading model  '{DETECTION_MODEL_ID}'  from HuggingFace …")
    processor = AutoProcessor.from_pretrained(DETECTION_MODEL_ID)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTION_MODEL_ID)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()
    print(f"[classify] Model loaded  (device: {device})")
    return processor, model, device


def classify_objects(objects: list,
                     prompts: list     = DETECTION_PROMPTS,
                     box_thresh: float = DETECTION_BOX_THRESH):
    """
    For every accepted cluster:
      1. Render a top-down 2-D image.
      2. Run Grounding-DINO zero-shot detection with the given text prompts.
      3. If a prompt is detected above box_thresh, update obj['class_name']
         and obj['confidence'] with the highest-scoring label.

    Modifies `objects` in-place and returns it.
    """
    if not objects:
        return objects

    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        print("[classify] WARNING: transformers / torch not installed – "
              "skipping classification step.")
        return objects

    processor, model, device = _load_grounding_dino()

    # Grounding-DINO expects a single dot-separated prompt string
    # e.g. "cone . barricade ."
    text_prompt = " . ".join(prompts) + " ."

    print(f"[classify] Text prompt   : '{text_prompt}'")
    print(f"[classify] Box threshold : {box_thresh}")

    for obj in objects:
        idx     = obj["index"]
        cluster = obj["cluster"]

        # ── 1. Render ─────────────────────────────────────────────────────────
        image = render_cluster_to_image(cluster, size=RENDER_IMAGE_SIZE)

        # ── 2. Inference ──────────────────────────────────────────────────────
        import torch
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process → boxes + scores + labels per image
        target_sizes = torch.tensor([image.size[::-1]], device=device)  # (H, W)

        # FIX 2: use `threshold=` (single param) instead of the removed
        # `box_threshold=` / `text_threshold=` keyword arguments.
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold    = box_thresh,
            target_sizes = target_sizes,
        )[0]   # single image → first (and only) result

        scores = results["scores"].cpu().numpy()   # (K,)
        labels = results["labels"]                 # list[str]

        # ── 3. Pick the best detection ────────────────────────────────────────
        if len(scores) == 0:
            print(f"[classify] Object #{idx:03d}  →  no match  "
                  f"(class kept as '{obj['class_name']}')")
            continue

        best_i     = int(np.argmax(scores))
        best_label = labels[best_i].strip().lower()
        best_score = float(scores[best_i])

        # Accept only if the label matches one of our prompts
        matched_prompt = None
        for prompt in prompts:
            if prompt.lower() in best_label:
                matched_prompt = prompt
                break

        if matched_prompt is None:
            print(f"[classify] Object #{idx:03d}  →  detected '{best_label}' "
                  f"(score={best_score:.2f}) but not in prompts – kept as "
                  f"'{obj['class_name']}'")
            continue

        obj["class_name"] = matched_prompt
        obj["confidence"] = best_score
        print(f"[classify] Object #{idx:03d}  →  '{matched_prompt}'  "
              f"(confidence={best_score:.2f})")

    return objects


# ──────────────────────────────────────────────────────────────────────────────
# SAVING
# ──────────────────────────────────────────────────────────────────────────────

def save_objects(objects: list, output_dir: str):
    """
    Save every detected object as:
        <output_dir>/<class_name>_NNN.ply        (point cloud)
        <output_dir>/<class_name>_NNN_view.png   (2-D render)
        <output_dir>/summary.csv                 (measurements + location + class)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Track per-class counters so names are cone_000, cone_001, barricade_000 …
    class_counters: dict = {}

    csv_rows = [
        "index,cluster_label,class_name,confidence,"
        "centroid_x_m,centroid_y_m,centroid_z_m,"
        "height_cm,height_ft,"
        "width_cm,width_ft,"
        "length_cm,length_ft,"
        "points,ply_file"
    ]

    for obj in objects:
        idx        = obj["index"]
        class_name = obj["class_name"]
        conf       = obj["confidence"]

        # Build a unique filename per class
        class_counters.setdefault(class_name, 0)
        class_idx = class_counters[class_name]
        class_counters[class_name] += 1
        stem      = f"{class_name}_{class_idx:03d}"

        # ── point cloud ──────────────────────────────────────────────────────
        ply_path = os.path.join(output_dir, f"{stem}.ply")
        o3d.io.write_point_cloud(ply_path, obj["cluster"])
        print(f"[save]   {stem}.ply  "
              f"({len(obj['cluster'].points):,} pts)  →  {ply_path}")

        # ── 2-D render ───────────────────────────────────────────────────────
        try:
            img      = render_cluster_to_image(obj["cluster"])
            img_path = os.path.join(output_dir, f"{stem}_view.png")
            img.save(img_path)
            print(f"[save]   {stem}_view.png  →  {img_path}")
        except Exception as exc:
            print(f"[save]   WARNING: could not save render for {stem}: {exc}")

        # ── CSV row ──────────────────────────────────────────────────────────
        h, w, l    = obj["height_m"], obj["width_m"], obj["length_m"]
        cx, cy, cz = obj["centroid"]
        csv_rows.append(
            f"{idx},{obj['label']},{class_name},{conf:.3f},"
            f"{cx:.4f},{cy:.4f},{cz:.4f},"
            f"{h*M_TO_CM:.2f},{h*M_TO_FEET:.4f},"
            f"{w*M_TO_CM:.2f},{w*M_TO_FEET:.4f},"
            f"{l*M_TO_CM:.2f},{l*M_TO_FEET:.4f},"
            f"{len(obj['cluster'].points)},{stem}.ply"
        )

    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w") as fh:
        fh.write("\n".join(csv_rows) + "\n")
    print(f"[save]   Summary CSV  →  {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

# Distinct colours for up to 10 objects; cycles if more exist.
_PALETTE = [
    [1.00, 0.45, 0.00],   # orange
    [0.20, 0.60, 1.00],   # blue
    [0.10, 0.80, 0.30],   # green
    [0.90, 0.10, 0.10],   # red
    [0.80, 0.00, 0.80],   # purple
    [0.00, 0.80, 0.80],   # cyan
    [0.90, 0.90, 0.00],   # yellow
    [1.00, 0.50, 0.70],   # pink
    [0.50, 0.30, 0.10],   # brown
    [0.40, 0.40, 0.40],   # grey
]


def visualise(original_pcd, objects, ground_model):
    """Render the full cloud (grey) with each detected object in its own colour."""
    vis_items = []

    grey = o3d.geometry.PointCloud(original_pcd)
    grey.paint_uniform_color([0.6, 0.6, 0.6])
    vis_items.append(grey)

    for obj in objects:
        colour = _PALETTE[obj["index"] % len(_PALETTE)]
        coloured = o3d.geometry.PointCloud(obj["cluster"])
        coloured.paint_uniform_color(colour)
        vis_items.append(coloured)

        bbox = obj["bbox"]
        bbox.color = tuple(colour)
        vis_items.append(bbox)

    o3d.visualization.draw_geometries(
        vis_items,
        window_name="ColorObjectsDetector – Construction Site",
        width=1280,
        height=720,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect orange/white objects in a construction-site point cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",
        help="Input point cloud (.pcd .ply .las .laz .xyz)")
    p.add_argument("-o", "--output-dir", default="detected_objects",
        help="Directory to save individual object clouds + CSV summary")
    p.add_argument("--no-save", action="store_true",
        help="Skip writing output files")
    p.add_argument("--visualise", action="store_true",
        help="Open interactive 3-D viewer after processing")
    p.add_argument("--no-classify", action="store_true",
        help="Skip the Grounding-DINO classification step")
    p.add_argument("--prompts", nargs="+", default=DETECTION_PROMPTS,
        help="Text prompts for zero-shot classification  "
             "(default: cone barricade)")
    p.add_argument("--box-thresh", type=float, default=DETECTION_BOX_THRESH,
        help="Confidence threshold for Grounding-DINO detections")

    # Overridable thresholds
    p.add_argument("--ground-threshold", type=float, default=GROUND_DISTANCE_THRESHOLD,
        help="RANSAC distance threshold for ground plane (m)")
    p.add_argument("--above-margin", type=float, default=ABOVE_GROUND_MARGIN,
        help="Height above ground plane to start keeping points (m)")
    p.add_argument("--eps", type=float, default=DBSCAN_EPS,
        help="DBSCAN neighbourhood radius (m)")
    p.add_argument("--min-points", type=int, default=DBSCAN_MIN_POINTS,
        help="DBSCAN minimum cluster size")
    p.add_argument("--min-size", type=float, default=MIN_OBJ_DIMENSION,
        help="Minimum bounding-box extent in any direction (m)")
    p.add_argument("--max-size", type=float, default=MAX_OBJ_DIMENSION,
        help="Maximum bounding-box extent in any direction (m)")
    p.add_argument("--min-height", type=float, default=MIN_OBJ_HEIGHT,
        help="Minimum object height / Z extent (m)  [default: 0.10 = 10 cm]")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Apply CLI overrides to module-level constants
    global GROUND_DISTANCE_THRESHOLD, ABOVE_GROUND_MARGIN
    global DBSCAN_EPS, DBSCAN_MIN_POINTS
    global MIN_OBJ_DIMENSION, MAX_OBJ_DIMENSION, MIN_OBJ_HEIGHT

    GROUND_DISTANCE_THRESHOLD = args.ground_threshold
    ABOVE_GROUND_MARGIN       = args.above_margin
    DBSCAN_EPS                = args.eps
    DBSCAN_MIN_POINTS         = args.min_points
    MIN_OBJ_DIMENSION         = args.min_size
    MAX_OBJ_DIMENSION         = args.max_size
    MIN_OBJ_HEIGHT            = args.min_height

    banner = "  ColorObjectsDetector – Construction Site"
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))

    # ── 1. Load ───────────────────────────────────────────────────────────────
    pcd = load_point_cloud(args.input)

    # ── 2. Remove ground ─────────────────────────────────────────────────────
    above_pcd, ground_model = remove_ground(pcd)

    # ── 3. Colour filter ──────────────────────────────────────────────────────
    colour_pcd = filter_by_colour(above_pcd)

    # ── 4. Cluster + measure ──────────────────────────────────────────────────
    objects = cluster_and_measure(colour_pcd)

    # ── 5. Classify with Grounding-DINO (SAM3) ────────────────────────────────
    if not args.no_classify and objects:
        print(f"\n[classify] Running zero-shot classification "
              f"with prompts {args.prompts} …")
        objects = classify_objects(
            objects,
            prompts    = args.prompts,
            box_thresh = args.box_thresh,
        )

    # ── 6. Save ───────────────────────────────────────────────────────────────
    if not args.no_save:
        if objects:
            save_objects(objects, args.output_dir)
        else:
            print("[save]   Nothing to save.")

    # ── 7. Visualise ──────────────────────────────────────────────────────────
    if args.visualise:
        visualise(pcd, objects, ground_model)

    print("=" * len(banner))
    print(f"  Done.  {len(objects)} object(s) detected.")
    # Print final class summary
    from collections import Counter
    counts = Counter(obj["class_name"] for obj in objects)
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls:20s}: {cnt}")
    print("=" * len(banner))
    return 0 if objects else 1


if __name__ == "__main__":
    sys.exit(main())
