"""
cone_image_detector.py
======================
Safety-cone detector for VGGT outdoor construction-site pipelines.

Strategy
--------
1.  Load the VGGT prediction dict  (world_points, images, extrinsic, intrinsic,
    depth, depth_conf – same structure as demo_viser.py).
2.  Pick the "best" camera frame (highest median confidence) and render it as
    a single composite image, or optionally composite all frames into a top-down
    orthographic projection.
3.  Run YOLOv8 (ultralytics) cone detection on that image.
4.  For each 2-D bounding box, back-project every pixel inside it through the
    known depth map and camera intrinsics/extrinsics → get a 3-D point cloud
    sub-set belonging to that detection.
5.  Measure the cone in 3-D:
      • height above the RANSAC ground plane
      • footprint area (convex hull on the ground plane)
      • aspect ratio (height / sqrt(footprint))
      • 3-D bounding box (min / max XYZ)
6.  Export each cone cluster as a .ply and write a JSON/CSV summary.

Usage
-----
    # From a pre-computed VGGT prediction file:
    python cone_image_detector.py \
        --pred_pkl  predictions/scene.pkl \
        --output_dir cone_results \
        --conf 0.35 \
        --visualize

    # From a directory of prediction files produced by vggt_video_inference.py:
    python cone_image_detector.py \
        --pred_dir  /kaggle/working/predictions \
        --output_dir cone_results

    # End-to-end from a raw video (runs VGGT reconstruction first):
    python cone_image_detector.py \
        --video     site_footage.mp4 \
        --output_dir cone_results \
        --max_frames 60 \
        --target_fps 2 \
        --visualize

    # From a GLB/GLTF 3-D mesh (synthetic camera views generated automatically):
    python cone_image_detector.py \
        --glb       scene.glb \
        --output_dir cone_results \
        --conf 0.35 \
        --visualize

Dependencies
------------
    pip install ultralytics open3d numpy pillow scipy opencv-python-headless
    pip install trimesh          # optional – GLB fallback when open3d is absent
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── optional heavy deps ───────────────────────────────────────────────────────
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import trimesh as _trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    HAS_VGGT = True
except ImportError:
    HAS_VGGT = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConeDetection2D:
    """A single YOLO bounding-box hit."""
    frame_idx: int          # which source frame the detection came from
    x1: float; y1: float
    x2: float; y2: float
    conf: float
    label: str


@dataclass
class ConeDetection3D:
    """Full 3-D cone extracted from a 2-D detection."""
    det2d: ConeDetection2D
    points_xyz: np.ndarray          # (N, 3) world-space XYZ
    points_rgb: Optional[np.ndarray]  # (N, 3) or None
    centroid: np.ndarray            # (3,)
    height_m: float
    footprint_area_m2: float
    aspect_ratio: float
    bbox_min: np.ndarray            # (3,) world-space
    bbox_max: np.ndarray            # (3,) world-space
    score: float

    def __repr__(self) -> str:
        return (
            f"Cone  frame={self.det2d.frame_idx}  "
            f"conf2d={self.det2d.conf:.2f}  "
            f"centroid=({self.centroid[0]:.2f},{self.centroid[1]:.2f},{self.centroid[2]:.2f})  "
            f"h={self.height_m:.2f}m  "
            f"foot={self.footprint_area_m2:.3f}m²  "
            f"score={self.score:.3f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers  (minimal, self-contained)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_ground_ransac(xyz: np.ndarray, dist_thresh: float = 0.05,
                       n_iter: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Robust ground-plane fit.
    Returns (unit_normal [3], d) such that  normal·p + d ≈ 0  for ground pts.
    Normal is oriented upward (+Z).
    """
    if HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        model, _ = pcd.segment_plane(dist_thresh, 3, n_iter)
        a, b, c, d = model
        normal = np.array([a, b, c], dtype=float)
    else:
        # PCA fallback
        centroid = xyz.mean(0)
        _, _, Vt = np.linalg.svd(xyz - centroid)
        normal = Vt[-1].astype(float)
        d = float(-normal @ centroid)

    normal /= np.linalg.norm(normal)
    # Ensure upward orientation
    if normal[2] < 0:
        normal, d = -normal, -d
    return normal, float(d)


def _height_above_plane(xyz: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Signed distance above the ground plane for each point."""
    return xyz @ normal + d


def _footprint_area(xyz: np.ndarray, normal: np.ndarray) -> float:
    """Convex-hull area of the point footprint projected onto the ground plane."""
    # Build in-plane basis
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    pts2d = np.column_stack([xyz @ u, xyz @ v])
    if HAS_SCIPY and len(pts2d) >= 3:
        try:
            return float(ConvexHull(pts2d).volume)  # .volume == area in 2-D
        except Exception:
            pass
    ranges = pts2d.max(0) - pts2d.min(0)
    return float(ranges[0] * ranges[1])


# ══════════════════════════════════════════════════════════════════════════════
# Prediction loader  (handles .pkl / .npy / directory)
# ══════════════════════════════════════════════════════════════════════════════

def load_predictions(path: str) -> dict:
    """
    Load a VGGT prediction dict.

    Accepted sources
    ----------------
    • A single .pkl file  (torch tensors or numpy arrays)
    • A single .npy file  (structured array or plain (N,3) coords)
    • A directory         (scans for the first .pkl / .npy / .npz)

    The returned dict is guaranteed to have numpy arrays (not torch tensors).
    Required keys:  world_points  (S,H,W,3)
                    images        (S,3,H,W)  OR  (S,H,W,3)
                    extrinsic     (S,3,4)
                    intrinsic     (S,3,3)
    Optional keys:  depth         (S,H,W,1) or (S,H,W)
                    depth_conf    (S,H,W)
                    world_points_conf  (S,H,W)
    """
    p = Path(path)
    if p.is_dir():
        for pattern in ("*.pkl", "*.npy", "*.npz", "*.glb", "*.gltf"):
            hits = sorted(p.glob(pattern))
            if hits:
                p = hits[0]
                log.info(f"Found prediction file: {p.name}")
                break
        else:
            raise FileNotFoundError(f"No prediction file found in {path}")

    log.info(f"Loading predictions from {p}")
    if p.suffix == ".pkl":
        with open(p, "rb") as f:
            data = pickle.load(f)
    elif p.suffix == ".npz":
        data = dict(np.load(p, allow_pickle=True))
    elif p.suffix == ".npy":
        data = np.load(p, allow_pickle=True).item()
    elif p.suffix.lower() in (".glb", ".gltf"):
        return _load_glb_as_pred(str(p))
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

    # Convert any torch tensors → numpy
    data = _to_numpy_recursive(data)
    return data


def _to_numpy_recursive(obj):
    """Recursively convert torch tensors / nested structures to numpy."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_numpy_recursive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_numpy_recursive(v) for v in obj)
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# GLB / GLTF loader  (mesh → synthetic camera prediction dict)
# ══════════════════════════════════════════════════════════════════════════════

def _load_glb_pointcloud(glb_path: str, n_sample: int = 100_000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a GLB/GLTF mesh and return sampled surface points.

    Returns
    -------
    xyz : (N, 3) float64  – world-space positions
    rgb : (N, 3) float32  – colours in [0, 1]
    """
    p = str(glb_path)

    # ── Try Open3D first (reads GLB as a triangle mesh) ───────────────────────
    if HAS_O3D:
        try:
            mesh = o3d.io.read_triangle_mesh(p)
            if len(mesh.vertices) > 0:
                n_pts = max(n_sample, len(mesh.vertices) * 5)
                pcd = mesh.sample_points_uniformly(number_of_points=n_pts)
                xyz = np.asarray(pcd.points, dtype=np.float64)
                rgb = (
                    np.asarray(pcd.colors, dtype=np.float32)
                    if pcd.has_colors()
                    else np.full((len(xyz), 3), 0.7, dtype=np.float32)
                )
                log.info(f"GLB loaded via Open3D: {len(xyz)} points sampled from {p}")
                return xyz, rgb
        except Exception as exc:
            log.warning(f"Open3D GLB load failed ({exc}), trying trimesh")

    # ── trimesh fallback ──────────────────────────────────────────────────────
    if HAS_TRIMESH:
        import trimesh as tm
        scene = tm.load(p)
        meshes = (
            list(scene.geometry.values())
            if hasattr(scene, "geometry")
            else [scene]
        )
        all_xyz: List[np.ndarray] = []
        all_rgb: List[np.ndarray] = []
        for mesh in meshes:
            if not hasattr(mesh, "vertices"):
                continue
            verts = np.array(mesh.vertices, dtype=np.float64)
            all_xyz.append(verts)
            vc = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
            if vc is not None:
                all_rgb.append(np.array(vc, dtype=np.float32)[:, :3] / 255.0)
            else:
                all_rgb.append(np.full((len(verts), 3), 0.7, dtype=np.float32))

        if all_xyz:
            xyz = np.concatenate(all_xyz)
            rgb = np.concatenate(all_rgb)
            if len(xyz) > n_sample:
                idx = np.random.choice(len(xyz), n_sample, replace=False)
                xyz, rgb = xyz[idx], rgb[idx]
            log.info(f"GLB loaded via trimesh: {len(xyz)} points from {p}")
            return xyz, rgb

    raise RuntimeError(
        f"Cannot load GLB: {glb_path}\n"
        "Install open3d or trimesh:  pip install open3d trimesh"
    )


def _load_glb_as_pred(
    path: str,
    img_h: int = 512,
    img_w: int = 512,
    n_cameras: int = 4,
    n_sample_pts: int = 100_000,
    camera_dist_factor: float = 2.5,
) -> dict:
    """
    Convert a GLB/GLTF mesh file to a VGGT-compatible prediction dict by
    sampling a point cloud from the mesh surface and rendering synthetic
    perspective views around the scene.

    Parameters
    ----------
    path               : Path to the .glb or .gltf file.
    img_h / img_w      : Height / width of each synthetic rendered frame.
    n_cameras          : Number of camera views evenly spaced in azimuth.
    n_sample_pts       : Points sampled from the mesh surface.
    camera_dist_factor : Camera orbit radius as a multiple of the scene radius.

    Returns
    -------
    Prediction dict with keys: world_points (S,H,W,3), images (S,3,H,W),
                               extrinsic (S,3,4), intrinsic (S,3,3),
                               world_points_conf (S,H,W)
    """
    xyz, rgb = _load_glb_pointcloud(path, n_sample_pts)

    centroid = xyz.mean(axis=0)
    xyz_c = xyz - centroid                                     # centred coords
    scene_radius = float(np.percentile(np.linalg.norm(xyz_c, axis=1), 95))
    if scene_radius < 1e-6:
        scene_radius = 1.0
    cam_dist = scene_radius * camera_dist_factor

    # Intrinsic matrix for ~60° horizontal FoV
    fx = fy = img_w / (2.0 * np.tan(np.radians(30.0)))
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    world_pts_list: List[np.ndarray] = []
    images_list:    List[np.ndarray] = []
    extrin_list:    List[np.ndarray] = []
    intrin_list:    List[np.ndarray] = []
    conf_list:      List[np.ndarray] = []

    azimuths  = np.linspace(0.0, 2.0 * np.pi, n_cameras, endpoint=False)
    elevation = np.radians(20.0)   # slight downward angle

    for az in azimuths:
        # Camera position on a sphere around the scene centroid
        cam_pos = np.array([
            cam_dist * np.cos(az) * np.cos(elevation),
            cam_dist * np.sin(az) * np.cos(elevation),
            cam_dist * np.sin(elevation),
        ])

        # Camera axes: look toward origin, Z-up
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up_ref  = np.array([0.0, 0.0, 1.0])
        right   = np.cross(forward, up_ref)
        if np.linalg.norm(right) < 1e-6:
            up_ref = np.array([0.0, 1.0, 0.0])
            right  = np.cross(forward, up_ref)
        right /= np.linalg.norm(right)
        up_cam = np.cross(right, forward)

        # World-to-camera extrinsic  [R | t]  (3×4)
        R_wc = np.stack([right, up_cam, forward], axis=0)   # (3, 3)
        t_wc = -(R_wc @ cam_pos)                            # (3,)
        extrinsic = np.concatenate([R_wc, t_wc[:, None]], axis=1)  # (3, 4)

        # Project all centred points into this camera
        pts_cam = (R_wc @ xyz_c.T).T + t_wc   # (N, 3)
        in_front = pts_cam[:, 2] > 0
        pts_f  = pts_cam[in_front]
        rgb_f  = rgb[in_front]
        xyz_f  = xyz[in_front]                 # original world coords

        world_pts_frame = np.zeros((img_h, img_w, 3), dtype=np.float64)
        img_hwc         = np.zeros((img_h, img_w, 3), dtype=np.float32)
        conf_frame      = np.zeros((img_h, img_w),    dtype=np.float32)

        if len(pts_f) > 0:
            u = (pts_f[:, 0] * fx / pts_f[:, 2] + cx).astype(int)
            v = (pts_f[:, 1] * fy / pts_f[:, 2] + cy).astype(int)
            in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
            u, v   = u[in_img], v[in_img]
            rgb_m  = rgb_f[in_img]
            xyz_m  = xyz_f[in_img]
            dep_m  = pts_f[in_img, 2]

            # Painter's algorithm (far → near) so nearest point wins
            order = np.argsort(dep_m)[::-1]
            u, v, rgb_m, xyz_m = u[order], v[order], rgb_m[order], xyz_m[order]

            world_pts_frame[v, u] = xyz_m
            img_hwc[v, u]         = rgb_m
            conf_frame[v, u]      = 1.0

        world_pts_list.append(world_pts_frame[None])              # (1,H,W,3)
        images_list.append(img_hwc.transpose(2, 0, 1)[None])     # (1,3,H,W)
        extrin_list.append(extrinsic[None])                       # (1,3,4)
        intrin_list.append(K[None])                               # (1,3,3)
        conf_list.append(conf_frame[None])                        # (1,H,W)

    pred = {
        "world_points":      np.concatenate(world_pts_list).astype(np.float64),  # (S,H,W,3)
        "images":            np.concatenate(images_list).astype(np.float32),      # (S,3,H,W)
        "extrinsic":         np.concatenate(extrin_list).astype(np.float64),      # (S,3,4)
        "intrinsic":         np.concatenate(intrin_list).astype(np.float64),      # (S,3,3)
        "world_points_conf": np.concatenate(conf_list).astype(np.float32),        # (S,H,W)
    }
    log.info(
        f"GLB prediction dict: world_points {pred['world_points'].shape}, "
        f"images {pred['images'].shape}  ({n_cameras} synthetic views)"
    )
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# Step 0 – VGGT Reconstruction from Video
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames_from_video(
    video_path: str,
    out_dir: Path,
    max_frames: int = 60,
    target_fps: Optional[float] = None,
) -> List[str]:
    """
    Extract frames from a video file and save them as PNGs.

    Parameters
    ----------
    video_path  : Path to the input video file.
    out_dir     : Directory to write extracted frames.
    max_frames  : Hard cap on the number of frames extracted.
    target_fps  : Sub-sample the video to approximately this frame rate
                  before passing to VGGT.  None = keep all frames up to
                  max_frames without sub-sampling.

    Returns
    -------
    Sorted list of absolute paths to the extracted PNG files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(native_fps / target_fps))) if target_fps else 1
    log.info(
        f"Video: {total_frames} frames @ {native_fps:.1f} fps  "
        f"→ sampling every {step} frame(s), cap={max_frames}"
    )

    saved: List[str] = []
    frame_idx = 0
    while len(saved) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            fname = out_dir / f"frame_{len(saved):05d}.png"
            cv2.imwrite(str(fname), frame)
            saved.append(str(fname))
        frame_idx += 1

    cap.release()
    log.info(f"Extracted {len(saved)} frames to {out_dir}")
    return sorted(saved)


def vggt_reconstruct_from_video(
    video_path: str,
    output_dir: str = "cone_results",
    max_frames: int = 60,
    target_fps: Optional[float] = 2.0,
    device: Optional[str] = None,
    vggt_model_path: Optional[str] = None,
    save_predictions: bool = True,
) -> dict:
    """
    Run VGGT reconstruction on a video file and return a prediction dict
    that is ready to be passed directly to :func:`run_pipeline`.

    This is Step 0 of the end-to-end pipeline:
        video → frame extraction → VGGT inference → prediction dict
                                                        ↓
                                              run_pipeline (cone detection)

    Parameters
    ----------
    video_path        : Path to the input video (.mp4 / .avi / .mov, etc.).
    output_dir        : Root output directory; frames and predictions are
                        persisted here.
    max_frames        : Maximum number of frames to extract and feed to VGGT.
                        More frames → richer reconstruction but slower inference.
    target_fps        : Sub-sample video to approximately this frame rate
                        before VGGT.  None = use all frames up to max_frames.
    device            : ``'cuda'`` / ``'cpu'``.  Defaults to CUDA if available.
    vggt_model_path   : Local path to VGGT weights (.pt).  If None the weights
                        are downloaded from HuggingFace (facebook/VGGT-1B).
    save_predictions  : Persist the prediction dict as ``predictions.pkl``
                        inside output_dir so it can be reused without
                        re-running VGGT.

    Returns
    -------
    dict with keys: world_points (S,H,W,3), images (S,3,H,W),
                    extrinsic (S,3,4), intrinsic (S,3,3),
                    depth (S,H,W,1), depth_conf (S,H,W),
                    world_points_conf (S,H,W)
    """
    if not HAS_VGGT:
        raise RuntimeError(
            "The vggt package is required for video reconstruction.\n"
            "Install with:  pip install vggt\n"
            "Or obtain it from:  https://github.com/facebookresearch/vggt"
        )

    import pickle as _pkl  # already in stdlib; only needed locally here

    out_dir = Path(output_dir)
    frames_dir = out_dir / "extracted_frames"

    # ── 0.1  Extract frames ───────────────────────────────────────────────────
    log.info("Step 0.1 – Extracting frames from video")
    frame_paths = extract_frames_from_video(
        video_path, frames_dir, max_frames=max_frames, target_fps=target_fps
    )
    if not frame_paths:
        raise RuntimeError(f"No frames could be extracted from video: {video_path}")

    # ── 0.2  Load VGGT model ──────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Step 0.2 – Loading VGGT model (device={device})")

    model = VGGT()
    if vggt_model_path and Path(vggt_model_path).exists():
        state = torch.load(vggt_model_path, map_location="cpu")
        model.load_state_dict(state)
        log.info(f"  Loaded weights from {vggt_model_path}")
    else:
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        log.info("  Downloading VGGT weights from HuggingFace …")
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
        )

    model.eval()
    model = model.to(device)

    # ── 0.3  Pre-process images ───────────────────────────────────────────────
    log.info(f"Step 0.3 – Pre-processing {len(frame_paths)} frames")
    images = load_and_preprocess_images(frame_paths).to(device)
    log.info(f"  Image tensor shape: {tuple(images.shape)}")

    # ── 0.4  VGGT inference ───────────────────────────────────────────────────
    log.info("Step 0.4 – Running VGGT inference")
    use_amp = device == "cuda"
    dtype = torch.float16
    if use_amp and torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype, enabled=use_amp):
            predictions = model(images)

    # ── 0.5  Decode pose encoding → extrinsic / intrinsic matrices ───────────
    log.info("Step 0.5 – Converting pose encoding to camera matrices")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # ── 0.6  Move everything to CPU numpy (strip batch dimension) ─────────────
    pred_np: dict = {}
    for key, val in predictions.items():
        if isinstance(val, torch.Tensor):
            pred_np[key] = val.detach().cpu().numpy().squeeze(0)
        else:
            pred_np[key] = val

    log.info(
        f"  Reconstruction complete:  "
        f"world_points {pred_np['world_points'].shape},  "
        f"images {pred_np['images'].shape}"
    )

    # ── 0.7  Optionally persist predictions ───────────────────────────────────
    if save_predictions:
        pred_path = out_dir / "predictions.pkl"
        with open(pred_path, "wb") as f:
            _pkl.dump(pred_np, f)
        log.info(f"  Predictions saved to {pred_path}")

    return pred_np


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – Render the point cloud to a single composite image
# ══════════════════════════════════════════════════════════════════════════════

def select_best_frame(pred: dict) -> int:
    """
    Return the index of the frame with the highest median depth confidence
    (proxy for well-reconstructed, in-focus frames).
    """
    conf_key = "depth_conf" if "depth_conf" in pred else "world_points_conf"
    if conf_key not in pred:
        return 0
    conf = pred[conf_key]           # (S, H, W)
    if conf.ndim == 4:
        conf = conf[..., 0]         # strip trailing dim if present
    medians = np.median(conf.reshape(conf.shape[0], -1), axis=1)
    return int(np.argmax(medians))


def render_frame_image(pred: dict, frame_idx: int) -> np.ndarray:
    """
    Return a uint8 BGR image (H, W, 3) for the chosen frame.
    Uses the raw camera image stored in the prediction dict.
    """
    images = pred["images"]  # (S, 3, H, W)  or  (S, H, W, 3)
    if images.ndim == 4 and images.shape[1] == 3:
        # CHW → HWC
        img = images[frame_idx].transpose(1, 2, 0)
    else:
        img = images[frame_idx]

    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    # RGB → BGR for OpenCV / YOLO
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def render_composite_image(pred: dict, n_cols: int = 4) -> Tuple[np.ndarray, List[dict]]:
    """
    Tile all frames into one composite image for a single YOLO pass.

    Returns
    -------
    composite : (H_total, W_total, 3) uint8 BGR
    tile_info  : list of dicts with keys  frame_idx, row, col, h, w
                 so detections can be mapped back to source frames.
    """
    images = pred["images"]        # (S, C, H, W) or (S, H, W, C)
    S = images.shape[0]

    frames = []
    for i in range(S):
        frames.append(render_frame_image(pred, i))

    H, W = frames[0].shape[:2]
    n_rows = int(np.ceil(S / n_cols))

    composite = np.zeros((n_rows * H, n_cols * W, 3), dtype=np.uint8)
    tile_info: List[dict] = []

    for i, frame in enumerate(frames):
        r, c = divmod(i, n_cols)
        composite[r * H:(r + 1) * H, c * W:(c + 1) * W] = frame
        tile_info.append(dict(frame_idx=i, row=r, col=c, h=H, w=W))

    return composite, tile_info


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Detect safety cones in the 2-D image with YOLOv8
# ══════════════════════════════════════════════════════════════════════════════

# COCO class 9 = "traffic light", but there is no "traffic cone" in COCO-80.
# Use a fine-tuned cone model when available.  As a fallback we run a generic
# detector and accept any class whose name contains "cone" (case-insensitive).
CONE_CLASS_NAMES = {"cone", "traffic cone", "safety cone", "road cone"}


def load_yolo_model(model_path: str = "yolov8n.pt") -> "YOLO":
    """
    Load a YOLO model.  If model_path points to a cone-specific checkpoint
    (e.g. from Roboflow) it is used directly; otherwise the standard
    YOLOv8n COCO checkpoint is loaded and 'cone'-class hits are filtered.
    """
    if not HAS_YOLO:
        raise RuntimeError(
            "ultralytics is required.\n"
            "Install with:  pip install ultralytics"
        )
    log.info(f"Loading YOLO model: {model_path}")
    return YOLO(model_path)


def detect_cones_yolo(
    model: "YOLO",
    image_bgr: np.ndarray,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
) -> List[dict]:
    """
    Run inference and return cone detections as a list of dicts:
      { x1, y1, x2, y2, conf, label }  (pixel coordinates, float).
    """
    results = model.predict(
        image_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, str(cls_id)).lower()
            # Accept any class whose name contains "cone"
            if not any(kw in label for kw in ("cone",)):
                # Also accept "traffic_light" as a coarse proxy if no cone class
                if "traffic" not in label and "barrier" not in label:
                    continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append(
                dict(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, label=label)
            )

    log.info(f"YOLO: {len(detections)} cone hit(s) at conf≥{conf_threshold}")
    return detections


def map_detections_to_frames(
    detections: List[dict],
    tile_info: List[dict],
) -> List[ConeDetection2D]:
    """
    Convert detections on the composite image back to per-frame pixel coords.
    """
    result: List[ConeDetection2D] = []
    for det in detections:
        cx = (det["x1"] + det["x2"]) / 2
        cy = (det["y1"] + det["y2"]) / 2
        # Find which tile the centre falls in
        for tile in tile_info:
            r, c, h, w = tile["row"], tile["col"], tile["h"], tile["w"]
            tx0, ty0 = c * w, r * h
            if tx0 <= cx < tx0 + w and ty0 <= cy < ty0 + h:
                # Translate to frame-local coordinates
                result.append(ConeDetection2D(
                    frame_idx=tile["frame_idx"],
                    x1=det["x1"] - tx0, y1=det["y1"] - ty0,
                    x2=det["x2"] - tx0, y2=det["y2"] - ty0,
                    conf=det["conf"],
                    label=det["label"],
                ))
                break
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – Back-project 2-D boxes into 3-D point cloud
# ══════════════════════════════════════════════════════════════════════════════

def backproject_box_to_3d(
    pred: dict,
    det: ConeDetection2D,
    depth_conf_threshold: float = 0.3,
    padding_px: int = 4,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract the 3-D points that correspond to the 2-D bounding box.

    Uses the VGGT world_points map (S, H, W, 3) which gives a direct XYZ
    for every pixel — no explicit depth unprojection needed.

    Returns
    -------
    xyz   : (N, 3) float64  – 3-D points inside the box
    rgb   : (N, 3) float32  or None
    """
    fi = det.frame_idx
    H_img = pred["world_points"].shape[1]
    W_img = pred["world_points"].shape[2]

    # Clamp box to image bounds (with optional padding)
    x1 = max(0, int(det.x1) - padding_px)
    y1 = max(0, int(det.y1) - padding_px)
    x2 = min(W_img, int(det.x2) + padding_px)
    y2 = min(H_img, int(det.y2) + padding_px)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3)), None

    # World-space XYZ for every pixel in the crop
    wp = pred["world_points"][fi]          # (H, W, 3)
    crop_xyz = wp[y1:y2, x1:x2].reshape(-1, 3)

    # Confidence mask
    conf_key = "depth_conf" if "depth_conf" in pred else "world_points_conf"
    if conf_key in pred:
        conf = pred[conf_key][fi]           # (H, W)  or  (H, W, 1)
        if conf.ndim == 3:
            conf = conf[..., 0]
        crop_conf = conf[y1:y2, x1:x2].ravel()
        # Normalise confidence to [0, 1]
        if crop_conf.max() > 1.0:
            crop_conf = crop_conf / crop_conf.max()
        valid = crop_conf >= depth_conf_threshold
        crop_xyz = crop_xyz[valid]
        rgb_valid_mask = valid
    else:
        rgb_valid_mask = np.ones(len(crop_xyz), dtype=bool)

    # Colour
    images = pred["images"]                # (S, 3, H, W) or (S, H, W, 3)
    if images.ndim == 4 and images.shape[1] == 3:
        img = images[fi].transpose(1, 2, 0)
    else:
        img = images[fi]
    crop_rgb = img[y1:y2, x1:x2].reshape(-1, 3)
    if crop_rgb.max() > 1.0:
        crop_rgb = crop_rgb / 255.0
    crop_rgb = crop_rgb[rgb_valid_mask].astype(np.float32)

    # Remove degenerate points (inf / nan / zero)
    finite_mask = np.all(np.isfinite(crop_xyz), axis=1) & np.any(crop_xyz != 0, axis=1)
    return crop_xyz[finite_mask].astype(np.float64), crop_rgb[finite_mask]


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 – Measure cone geometry in 3-D
# ══════════════════════════════════════════════════════════════════════════════

def measure_cone_3d(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray],
    det: ConeDetection2D,
    ground_normal: np.ndarray,
    ground_d: float,
    min_pts: int = 10,
) -> Optional[ConeDetection3D]:
    """
    Given the 3-D points for a single 2-D detection, compute cone metrics.
    Returns None if the cluster is too small or clearly not cone-shaped.
    """
    if len(xyz) < min_pts:
        log.debug(f"  Frame {det.frame_idx} box: only {len(xyz)} pts – skipping")
        return None

    heights = _height_above_plane(xyz, ground_normal, ground_d)
    h_min = float(heights.min())
    h_max = float(heights.max())
    height_m = h_max - h_min

    if height_m < 0.05:
        log.debug(f"  Cluster too flat ({height_m:.3f}m) – skipping")
        return None

    footprint = _footprint_area(xyz, ground_normal)
    aspect = height_m / (np.sqrt(footprint) + 1e-9) if footprint > 0 else 0.0

    # Scoring  (same formula as cone-detector.py)
    height_score = float(np.exp(-0.5 * ((height_m - 0.55) / 0.25) ** 2))
    aspect_score = float(np.tanh(max(aspect - 1.5, 0) / 2.0))
    foot_score   = float(np.exp(-footprint / 0.1))
    score = (height_score + aspect_score + foot_score) / 3.0

    centroid = xyz.mean(axis=0)
    return ConeDetection3D(
        det2d=det,
        points_xyz=xyz,
        points_rgb=rgb,
        centroid=centroid,
        height_m=height_m,
        footprint_area_m2=footprint,
        aspect_ratio=aspect,
        bbox_min=xyz.min(axis=0),
        bbox_max=xyz.max(axis=0),
        score=score,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 – Visualise & export
# ══════════════════════════════════════════════════════════════════════════════

def annotate_image(
    image_bgr: np.ndarray,
    detections: List[ConeDetection2D],
    cones_3d: List[ConeDetection3D],
    tile_info: Optional[List[dict]] = None,
) -> np.ndarray:
    """Draw bounding boxes and 3-D labels on the composite image."""
    out = image_bgr.copy()
    cone_map: Dict[int, ConeDetection3D] = {id(d.det2d): d for d in cones_3d}

    for det in detections:
        cone = cone_map.get(id(det))
        colour = (0, 200, 255) if cone else (100, 100, 100)
        thickness = 2

        # Translate back to composite coordinates if tile_info provided
        ox, oy = 0, 0
        if tile_info:
            for tile in tile_info:
                if tile["frame_idx"] == det.frame_idx:
                    ox = tile["col"] * tile["w"]
                    oy = tile["row"] * tile["h"]
                    break

        pt1 = (int(det.x1 + ox), int(det.y1 + oy))
        pt2 = (int(det.x2 + ox), int(det.y2 + oy))
        cv2.rectangle(out, pt1, pt2, colour, thickness)

        label = f"{det.label} {det.conf:.2f}"
        if cone:
            label += f" | h={cone.height_m:.2f}m s={cone.score:.2f}"
        cv2.putText(out, label, (pt1[0], pt1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    return out


def save_cone_ply(cone: ConeDetection3D, out_dir: Path, idx: int) -> Path:
    """Save cone point cloud as a .ply file."""
    fname = out_dir / f"cone_{idx:03d}_h{cone.height_m:.2f}m.ply"
    if HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cone.points_xyz)
        if cone.points_rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(
                cone.points_rgb.clip(0, 1).astype(np.float64))
        o3d.io.write_point_cloud(str(fname), pcd)
    else:
        # Fallback: write ASCII PLY manually
        pts = cone.points_xyz
        has_color = cone.points_rgb is not None
        rgb = (cone.points_rgb * 255).astype(np.uint8) if has_color else None
        with open(fname, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if has_color:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(len(pts)):
                row = f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f}"
                if has_color:
                    row += f" {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}"
                f.write(row + "\n")
    log.info(f"  Saved {fname.name}")
    return fname


def save_summary(cones: List[ConeDetection3D], out_dir: Path):
    """Write cone measurements to JSON and CSV."""
    records = []
    for i, c in enumerate(cones):
        records.append(dict(
            cone_id=i,
            frame_idx=c.det2d.frame_idx,
            conf2d=round(c.det2d.conf, 4),
            centroid_x=round(float(c.centroid[0]), 4),
            centroid_y=round(float(c.centroid[1]), 4),
            centroid_z=round(float(c.centroid[2]), 4),
            height_m=round(c.height_m, 4),
            footprint_area_m2=round(c.footprint_area_m2, 4),
            aspect_ratio=round(c.aspect_ratio, 3),
            bbox_min=c.bbox_min.tolist(),
            bbox_max=c.bbox_max.tolist(),
            score=round(c.score, 4),
        ))

    json_path = out_dir / "cone_summary.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    log.info(f"Saved {json_path}")

    csv_path = out_dir / "cone_summary.csv"
    if records:
        with open(csv_path, "w", newline="") as f:
            flat_keys = [k for k in records[0] if not isinstance(records[0][k], list)]
            writer = csv.DictWriter(f, fieldnames=flat_keys)
            writer.writeheader()
            for r in records:
                writer.writerow({k: r[k] for k in flat_keys})
        log.info(f"Saved {csv_path}")


def visualize_open3d(
    pred: dict,
    cones: List[ConeDetection3D],
    frame_idx: int = 0,
):
    """Open an Open3D viewer with the full scene + highlighted cone clusters."""
    if not HAS_O3D:
        log.warning("open3d required for 3-D visualisation – skipping.")
        return

    geometries = []

    # Full-scene point cloud (one frame, grey)
    wp = pred["world_points"][frame_idx].reshape(-1, 3)
    finite = np.all(np.isfinite(wp), axis=1) & np.any(wp != 0, axis=1)
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(wp[finite])
    scene_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    geometries.append(scene_pcd)

    palette = [
        [1.0, 0.40, 0.0],
        [1.0, 0.85, 0.0],
        [0.8, 0.00, 0.0],
        [0.0, 0.80, 0.0],
        [0.0, 0.50, 1.0],
    ]
    for i, cone in enumerate(cones):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cone.points_xyz)
        pcd.paint_uniform_color(palette[i % len(palette)])
        geometries.append(pcd)

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=cone.bbox_min, max_bound=cone.bbox_max)
        bbox.color = palette[i % len(palette)]
        geometries.append(bbox)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Cone Detector – VGGT Pipeline",
        width=1280, height=720,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    pred: dict,
    yolo_model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    depth_conf_threshold: float = 0.3,
    output_dir: str = "cone_results",
    use_composite: bool = True,
    visualize: bool = False,
    save_ply: bool = True,
) -> List[ConeDetection3D]:
    """
    Full pipeline:  predictions dict → detected + measured cones.

    Parameters
    ----------
    pred               : VGGT prediction dict (numpy arrays).
    yolo_model_path    : Path to YOLOv8 weights (.pt). Use a cone-specific
                         checkpoint for best results; falls back to COCO.
    conf_threshold     : YOLO confidence threshold.
    iou_threshold      : YOLO NMS IoU threshold.
    depth_conf_threshold : Min normalised depth confidence to keep a 3-D point.
    output_dir         : Where to write .ply files and summary CSVs.
    use_composite      : If True, tile all frames into one image for detection.
                         If False, only the best-confidence frame is used.
    visualize          : Open Open3D viewer after detection.
    save_ply           : Save per-cone .ply files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Render ──────────────────────────────────────────────────────
    log.info("Step 1 – Rendering image(s) from prediction dict")
    if use_composite:
        composite, tile_info = render_composite_image(pred)
        detect_image = composite
        log.info(f"  Composite image: {composite.shape[1]}×{composite.shape[0]} px  "
                 f"({len(tile_info)} frames)")
    else:
        best_frame = select_best_frame(pred)
        detect_image = render_frame_image(pred, best_frame)
        tile_info = [dict(frame_idx=best_frame, row=0, col=0,
                          h=detect_image.shape[0], w=detect_image.shape[1])]
        log.info(f"  Single frame: {best_frame}  ({detect_image.shape[1]}×{detect_image.shape[0]} px)")

    render_path = out_dir / "render_for_detection.png"
    cv2.imwrite(str(render_path), detect_image)
    log.info(f"  Saved render to {render_path}")

    # ── Step 2: 2-D detection ────────────────────────────────────────────────
    log.info("Step 2 – Running YOLOv8 cone detection")
    model = load_yolo_model(yolo_model_path)
    raw_dets = detect_cones_yolo(model, detect_image, conf_threshold, iou_threshold)

    if not raw_dets:
        log.warning("No cone detections found. Try lowering --conf or use a fine-tuned model.")
        return []

    dets_2d = map_detections_to_frames(raw_dets, tile_info)
    log.info(f"  {len(dets_2d)} detection(s) mapped to source frames")

    # Save annotated image (2-D only at this point – 3-D labels added later)
    ann_img = annotate_image(detect_image, dets_2d, [], tile_info)
    cv2.imwrite(str(out_dir / "detections_2d.png"), ann_img)

    # ── Step 3: Back-project boxes → 3-D points ──────────────────────────────
    log.info("Step 3 – Back-projecting 2-D boxes to 3-D")

    # Estimate single global ground plane from the best frame's world_points
    best_frame = tile_info[0]["frame_idx"]
    wp_all = pred["world_points"][best_frame].reshape(-1, 3)
    finite_mask = np.all(np.isfinite(wp_all), axis=1) & np.any(wp_all != 0, axis=1)
    ground_normal, ground_d = _fit_ground_ransac(wp_all[finite_mask])
    log.info(f"  Ground plane normal={ground_normal.round(3)}, d={ground_d:.3f}")

    cone_clusters: List[ConeDetection3D] = []
    for det in dets_2d:
        xyz, rgb = backproject_box_to_3d(pred, det, depth_conf_threshold)
        log.info(f"  Frame {det.frame_idx} box ({det.label}): {len(xyz)} pts back-projected")
        if len(xyz) == 0:
            continue
        cone_3d = measure_cone_3d(xyz, rgb, det, ground_normal, ground_d)
        if cone_3d is not None:
            cone_clusters.append(cone_3d)
            log.info(f"    {cone_3d}")

    log.info(f"\n{'═'*60}")
    log.info(f"  {len(cone_clusters)} 3-D cone(s) extracted")
    log.info(f"{'═'*60}")

    # ── Step 4: Export ────────────────────────────────────────────────────────
    log.info("Step 4 – Saving results")

    if save_ply:
        for i, cone in enumerate(cone_clusters):
            save_cone_ply(cone, out_dir, i)

    save_summary(cone_clusters, out_dir)

    # Annotated image with 3-D measurements
    ann_img_3d = annotate_image(detect_image, dets_2d, cone_clusters, tile_info)
    cv2.imwrite(str(out_dir / "detections_3d.png"), ann_img_3d)
    log.info(f"  Saved annotated image to {out_dir / 'detections_3d.png'}")

    # ── Step 5 (optional): Visualise ─────────────────────────────────────────
    if visualize:
        log.info("Step 5 – Opening 3-D viewer")
        visualize_open3d(pred, cone_clusters, frame_idx=best_frame)

    return cone_clusters


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect & measure safety cones in a VGGT point cloud via image-space detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred_pkl",  type=str, help="Path to VGGT predictions .pkl file")
    src.add_argument("--pred_dir",  type=str, help="Directory containing VGGT prediction files")
    src.add_argument("--pred_npy",  type=str, help="Path to VGGT predictions .npy file")
    src.add_argument(
        "--glb", type=str,
        help=(
            "Path to a .glb or .gltf 3-D mesh file. "
            "Synthetic camera views are generated automatically from the mesh surface "
            "and fed into the cone-detection pipeline."
        ),
    )
    src.add_argument(
        "--video", type=str,
        help=(
            "Path to a video file (.mp4/.avi/.mov/etc.). "
            "Triggers Step 0: VGGT reconstruction is run on extracted frames "
            "before cone detection."
        ),
    )

    # ── VGGT reconstruction options (only relevant when --video is given) ─────
    p.add_argument(
        "--max_frames", type=int, default=60,
        help="(--video) Maximum number of frames extracted from the video.",
    )
    p.add_argument(
        "--target_fps", type=float, default=2.0,
        help=(
            "(--video) Sub-sample video to approximately this frame rate "
            "before VGGT inference. Use 0 to keep all frames up to --max_frames."
        ),
    )
    p.add_argument(
        "--vggt_model", type=str, default=None,
        help=(
            "(--video) Path to local VGGT weights (.pt). "
            "If omitted the model is downloaded from HuggingFace."
        ),
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="(--video) Compute device for VGGT inference: 'cuda' or 'cpu'. "
             "Defaults to CUDA when available.",
    )
    p.add_argument(
        "--no_save_pred", action="store_true",
        help="(--video) Skip saving the VGGT prediction dict to predictions.pkl.",
    )

    p.add_argument("--model",      type=str, default="yolov8n.pt",
                   help="YOLOv8 weights (use cone-specific checkpoint for best results)")
    p.add_argument("--conf",       type=float, default=0.35, help="YOLO confidence threshold")
    p.add_argument("--iou",        type=float, default=0.45, help="YOLO NMS IoU threshold")
    p.add_argument("--depth_conf", type=float, default=0.30,
                   help="Minimum depth confidence to keep a 3-D point")
    p.add_argument("--output_dir", type=str,  default="cone_results")
    p.add_argument("--single_frame", action="store_true",
                   help="Use only the best-confidence frame instead of tiling all frames")
    p.add_argument("--no_ply",     action="store_true", help="Skip saving .ply files")
    p.add_argument("--visualize",  action="store_true", help="Open Open3D 3-D viewer")
    return p


def main():
    args = build_parser().parse_args()

    if args.video:
        # ── Step 0: VGGT reconstruction from raw video ────────────────────────
        log.info(f"Input mode: video  →  {args.video}")
        pred = vggt_reconstruct_from_video(
            video_path=args.video,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            target_fps=args.target_fps if args.target_fps > 0 else None,
            device=args.device,
            vggt_model_path=args.vggt_model,
            save_predictions=not args.no_save_pred,
        )
    elif args.glb:
        # ── Load GLB/GLTF mesh and generate synthetic prediction dict ─────────
        log.info(f"Input mode: GLB  →  {args.glb}")
        pred = load_predictions(args.glb)
    else:
        # ── Load pre-computed predictions ─────────────────────────────────────
        pred_path = args.pred_pkl or args.pred_dir or args.pred_npy
        pred = load_predictions(pred_path)

    cones = run_pipeline(
        pred=pred,
        yolo_model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        depth_conf_threshold=args.depth_conf,
        output_dir=args.output_dir,
        use_composite=not args.single_frame,
        visualize=args.visualize,
        save_ply=not args.no_ply,
    )

    print(f"\n{'═'*60}")
    print(f"  DETECTED {len(cones)} SAFETY CONE(S)")
    print(f"{'═'*60}")
    for i, c in enumerate(cones):
        print(f"  [{i:02d}] {c}")
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
