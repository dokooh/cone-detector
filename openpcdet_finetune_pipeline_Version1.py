#!/usr/bin/env python3
"""
OpenPCDet Finetuning Pipeline for Custom 3D Object Detection
=============================================================
Classes: SilverPlate, SafetyCone, Barricade
Based on: https://github.com/open-mmlab/OpenPCDet

Workflow:
  1. Dataset preparation (directory layout, label + point cloud ingestion)
  2. Config generation (dataset YAML + model YAML)
  3. Info-file and GT-database creation
  4. Finetuning (train.py) with pretrained backbone
  5. Evaluation / inference (test.py / demo.py)

Requirements (install before running):
  pip install torch torchvision open3d numpy pyyaml easydict
  # Install OpenPCDet (with spconv):
  #   git clone https://github.com/open-mmlab/OpenPCDet
  #   cd OpenPCDet && pip install -r requirements.txt && python setup.py develop
"""

import os
import sys
import glob
import random
import shutil
import pickle
import logging
import argparse
import subprocess
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["SilverPlate", "SafetyCone", "Barricade"]

# KITTI eval needs each class mapped to a KITTI-style name
MAP_CLASS_TO_KITTI = {
    "SilverPlate": "Car",       # flat/compact  → reuse Car evaluation slot
    "SafetyCone":  "Pedestrian",# vertical/slim → reuse Pedestrian slot
    "Barricade":   "Cyclist",   # elongated     → reuse Cyclist slot
}

# Anchor sizes [l, w, h] (metres) tuned for excavation-site objects
ANCHOR_SIZES = {
    "SilverPlate": [0.6,  0.6,  0.05],  # thin, flat plate
    "SafetyCone":  [0.45, 0.45, 1.0 ],  # standard road cone
    "Barricade":   [2.0,  0.5,  1.0 ],  # road barricade
}

# Matched / unmatched IoU thresholds per class
IOU_THRESHOLDS = {
    "SilverPlate": (0.35, 0.20),
    "SafetyCone":  (0.35, 0.20),
    "Barricade":   (0.45, 0.30),
}

# Point cloud spatial range [xmin, ymin, zmin, xmax, ymax, zmax]
POINT_CLOUD_RANGE = [-50.0, -50.0, -3.0, 50.0, 50.0, 3.0]

# Voxel size — small voxels help resolve thin/small objects
VOXEL_SIZE = [0.05, 0.05, 0.10]

# Train / val split ratio
TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
# remainder → test

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ===========================================================================
# STEP 1 — Dataset directory layout
# ===========================================================================

def create_dataset_layout(root: Path) -> None:
    """
    Expected layout (mirrors OpenPCDet CustomDataset):

        <root>/
          points/          ← *.npy files  (N, 4) float32: x y z intensity
          labels/          ← *.txt files  one box per line: x y z l w h yaw ClassName
          ImageSets/
            train.txt
            val.txt
            test.txt
    """
    for d in ["points", "labels", "ImageSets",
              "gt_database", "custom_models"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    log.info("Dataset directory layout created under: %s", root)


# ===========================================================================
# STEP 2 — Point cloud ingestion helpers
# ===========================================================================

def load_bin(filepath: str) -> np.ndarray:
    """Load a Velodyne-style .bin file → (N, 4) float32."""
    pts = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    return pts


def load_pcd_open3d(filepath: str) -> np.ndarray:
    """Load a .pcd/.ply file via Open3D → (N, 4) float32 (intensity=0)."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("pip install open3d")
    pcd = o3d.io.read_point_cloud(filepath)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    intensity = np.zeros((len(xyz), 1), dtype=np.float32)
    return np.hstack([xyz, intensity])


def ingest_point_cloud(src: str, dst_npy: str) -> None:
    """Convert .bin / .pcd / .ply to the .npy format OpenPCDet expects."""
    ext = Path(src).suffix.lower()
    if ext == ".bin":
        pts = load_bin(src)
    elif ext in (".pcd", ".ply"):
        pts = load_pcd_open3d(src)
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")
    np.save(dst_npy, pts)
    log.debug("Saved %s → %s  (shape %s)", src, dst_npy, pts.shape)


def batch_ingest_point_clouds(src_dir: str, dst_dir: Path,
                               extensions: tuple = (".bin", ".pcd", ".ply")) -> list:
    """
    Walk *src_dir*, convert every point cloud to .npy in *dst_dir*.
    Returns a sorted list of sample IDs (file stems).
    """
    src_dir = Path(src_dir)
    sample_ids = []
    files = sorted(f for f in src_dir.rglob("*") if f.suffix.lower() in extensions)
    if not files:
        log.warning("No point cloud files found in %s", src_dir)
    for f in files:
        sid = f.stem
        ingest_point_cloud(str(f), str(dst_dir / f"{sid}.npy"))
        sample_ids.append(sid)
    log.info("Ingested %d point clouds.", len(sample_ids))
    return sample_ids


# ===========================================================================
# STEP 3 — Label helpers (KITTI-style .txt)
# ===========================================================================

def write_label_file(save_path: Path, boxes: np.ndarray, names: list) -> None:
    """
    Write one label .txt.
    Each line: x y z l w h yaw ClassName
    boxes: (N, 7) float — [cx, cy, cz, l, w, h, yaw]
    """
    with open(save_path, "w") as f:
        for box, name in zip(boxes, names):
            assert name in CLASS_NAMES, f"Unknown class: {name}"
            f.write(
                f"{box[0]:.4f} {box[1]:.4f} {box[2]:.4f} "
                f"{box[3]:.4f} {box[4]:.4f} {box[5]:.4f} "
                f"{box[6]:.4f} {name}\n"
            )


def parse_label_file(label_path: Path):
    """
    Parse a label .txt → (boxes np.ndarray (N,7), names list[str]).
    """
    boxes, names = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            boxes.append([float(p) for p in parts[:7]])
            names.append(parts[7])
    return np.array(boxes, dtype=np.float32), names


# ===========================================================================
# STEP 4 — Train / val / test split
# ===========================================================================

def generate_splits(sample_ids: list, root: Path,
                    train_ratio: float = TRAIN_RATIO,
                    val_ratio:   float = VAL_RATIO,
                    seed: int = 42) -> dict:
    """Randomly split sample IDs and write ImageSets txt files."""
    rng = random.Random(seed)
    ids = list(sample_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    splits = {
        "train": ids[:n_train],
        "val":   ids[n_train: n_train + n_val],
        "test":  ids[n_train + n_val:],
    }
    for split, sids in splits.items():
        path = root / "ImageSets" / f"{split}.txt"
        with open(path, "w") as f:
            f.write("\n".join(sids) + "\n")
        log.info("Split %-5s: %d samples → %s", split, len(sids), path)
    return splits


# ===========================================================================
# STEP 5 — Config generation
# ===========================================================================

def make_dataset_yaml(root: Path, output_path: Path) -> None:
    """Generate tools/cfgs/dataset_configs/custom_excavation_dataset.yaml"""

    # GT sampling config per class
    sample_groups = [f"{cls}:20" for cls in CLASS_NAMES]
    filter_min_pts = [f"{cls}:3" for cls in CLASS_NAMES]

    cfg = {
        "DATASET": "CustomDataset",
        "DATA_PATH": str(root),

        "POINT_CLOUD_RANGE": POINT_CLOUD_RANGE,

        "MAP_CLASS_TO_KITTI": MAP_CLASS_TO_KITTI,

        "DATA_SPLIT": {"train": "train", "test": "val"},

        "INFO_PATH": {
            "train": ["custom_infos_train.pkl"],
            "test":  ["custom_infos_val.pkl"],
        },

        "POINT_FEATURE_ENCODING": {
            "encoding_type": "absolute_coordinates_encoding",
            "used_feature_list": ["x", "y", "z", "intensity"],
            "src_feature_list":  ["x", "y", "z", "intensity"],
        },

        "DATA_AUGMENTOR": {
            "DISABLE_AUG_LIST": ["placeholder"],
            "AUG_CONFIG_LIST": [
                {
                    "NAME": "gt_sampling",
                    "USE_ROAD_PLANE": False,
                    "DB_INFO_PATH": ["custom_dbinfos_train.pkl"],
                    "PREPARE": {"filter_by_min_points": filter_min_pts},
                    "SAMPLE_GROUPS": sample_groups,
                    "NUM_POINT_FEATURES": 4,
                    "DATABASE_WITH_FAKELIDAR": False,
                    "REMOVE_EXTRA_WIDTH": [0.0, 0.0, 0.0],
                    "LIMIT_WHOLE_SCENE": True,
                },
                {"NAME": "random_world_flip",    "ALONG_AXIS_LIST": ["x"]},
                {"NAME": "random_world_rotation","WORLD_ROT_ANGLE": [-0.3927, 0.3927]},
                {"NAME": "random_world_scaling", "WORLD_SCALE_RANGE": [0.95, 1.05]},
            ],
        },

        "DATA_PROCESSOR": [
            {"NAME": "mask_points_and_boxes_outside_range",
             "REMOVE_OUTSIDE_BOXES": True},
            {"NAME": "shuffle_points",
             "SHUFFLE_ENABLED": {"train": True, "test": False}},
            {
                "NAME": "transform_points_to_voxels",
                "VOXEL_SIZE": VOXEL_SIZE,
                "MAX_POINTS_PER_VOXEL": 5,
                "MAX_NUMBER_OF_VOXELS": {"train": 150000, "test": 150000},
            },
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("Dataset YAML written → %s", output_path)


def _anchor_cfg_entry(class_name: str, feature_map_stride: int = 8) -> dict:
    sizes = ANCHOR_SIZES[class_name]
    matched, unmatched = IOU_THRESHOLDS[class_name]
    return {
        "class_name": class_name,
        "anchor_sizes": [sizes],
        "anchor_rotations": [0, 1.5708],
        "anchor_bottom_heights": [0],
        "align_center": False,
        "feature_map_stride": feature_map_stride,
        "matched_threshold": matched,
        "unmatched_threshold": unmatched,
    }


def make_model_yaml(dataset_yaml_path: Path, output_path: Path) -> None:
    """
    Generate a PointPillars-style model YAML for the custom classes.
    (Swap MODEL.NAME to CenterPoint / PV-RCNN as needed.)
    """
    anchor_configs = [_anchor_cfg_entry(cls) for cls in CLASS_NAMES]

    # Class-balanced loss weights (SilverPlate rare → higher weight)
    cls_weight_map = {
        "SilverPlate": 3.0,
        "SafetyCone":  1.5,
        "Barricade":   1.0,
    }

    cfg = {
        "CLASS_NAMES": CLASS_NAMES,

        "DATA_CONFIG": {
            "_BASE_CONFIG_": str(dataset_yaml_path),
        },

        "MODEL": {
            "NAME": "PointPillar",

            "VFE": {
                "NAME": "PillarVFE",
                "WITH_DISTANCE": False,
                "USE_ABSLOTE_XYZ": True,
                "USE_NORM": True,
                "NUM_FILTERS": [64],
            },

            "MAP_TO_BEV": {
                "NAME": "PointPillarScatter",
                "NUM_BEV_FEATURES": 64,
            },

            "BACKBONE_2D": {
                "NAME": "BaseBEVBackbone",
                "LAYER_NUMS": [3, 5, 5],
                "LAYER_STRIDES": [2, 2, 2],
                "NUM_FILTERS": [64, 128, 256],
                "UPSAMPLE_STRIDES": [1, 2, 4],
                "NUM_UPSAMPLE_FILTERS": [128, 128, 128],
            },

            "DENSE_HEAD": {
                "NAME": "AnchorHeadSingle",
                "CLASS_AGNOSTIC": False,
                "USE_DIRECTION_CLASSIFIER": True,
                "DIR_OFFSET": 0.78539,
                "DIR_LIMIT_OFFSET": 0.0,
                "NUM_DIR_BINS": 2,
                "ANCHOR_GENERATOR_CONFIG": anchor_configs,
                "TARGET_ASSIGNER_CONFIG": {
                    "NAME": "AxisAlignedTargetAssigner",
                    "POS_FRACTION": -1.0,
                    "SAMPLE_SIZE": 512,
                    "NORM_BY_NUM_EXAMPLES": False,
                    "MATCH_HEIGHT": False,
                    "BOX_CODER": "ResidualCoder",
                },
                "LOSS_CONFIG": {
                    "LOSS_WEIGHTS": {
                        "cls_weight": 1.0,
                        "loc_weight": 2.0,
                        "dir_weight": 0.2,
                        "code_weights": [1.0]*7,
                    }
                },
            },

            "POST_PROCESSING": {
                "RECALL_THRESH_LIST": [0.3, 0.5, 0.7],
                "SCORE_THRESH": 0.1,
                "OUTPUT_RAW_SCORE": False,
                "EVAL_METRIC": "kitti",
                "NMS_CONFIG": {
                    "MULTI_CLASSES_NMS": False,
                    "NMS_TYPE": "nms_gpu",
                    "NMS_THRESH": 0.5,
                    "NMS_PRE_MAXSIZE": 4096,
                    "NMS_POST_MAXSIZE": 500,
                },
            },
        },

        "OPTIMIZATION": {
            "BATCH_SIZE_PER_GPU": 4,
            "NUM_EPOCHS": 80,
            "OPTIMIZER": "adam_onecycle",
            "LR": 0.001,           # lower LR for finetuning
            "WEIGHT_DECAY": 0.01,
            "MOMENTUM": 0.9,
            "MOMS": [0.95, 0.85],
            "PCT_START": 0.4,
            "DIV_FACTOR": 10,
            "DECAY_STEP_LIST": [35, 45],
            "LR_DECAY": 0.1,
            "LR_CLIP": 1e-7,
            "LR_WARMUP": True,
            "WARMUP_EPOCH": 2,
            "GRAD_NORM_CLIP": 10,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("Model YAML written → %s", output_path)


# ===========================================================================
# STEP 6 — Info-file and GT-database creation
# ===========================================================================

def create_infos_and_gt_database(openpcdet_root: Path, data_root: Path,
                                  dataset_yaml: Path) -> None:
    """
    Calls OpenPCDet's built-in create_custom_infos() via subprocess so it
    uses the installed package (handles all internal imports cleanly).
    """
    script = openpcdet_root / "pcdet" / "datasets" / "custom" / "custom_dataset.py"
    cmd = [
        sys.executable, str(script),
        "create_custom_infos",
        str(dataset_yaml),
    ]
    log.info("Running: %s", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(openpcdet_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(openpcdet_root), env=env)
    log.info("Info files and GT database created under: %s", data_root)


# ===========================================================================
# STEP 7 — Finetuning
# ===========================================================================

def finetune(openpcdet_root: Path,
             model_yaml: Path,
             pretrained_ckpt: Path | None,
             output_dir: Path,
             num_gpus: int = 1,
             batch_size: int = 4,
             epochs: int = 80,
             extra_tag: str = "excavation_finetune") -> None:
    """
    Launch training via tools/train.py (single or multi-GPU).
    """
    train_script = openpcdet_root / "tools" / "train.py"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(train_script),
        "--cfg_file",  str(model_yaml),
        "--batch_size", str(batch_size),
        "--epochs",     str(epochs),
        "--output_dir", str(output_dir),
        "--extra_tag",  extra_tag,
    ]

    if pretrained_ckpt and pretrained_ckpt.exists():
        cmd += ["--pretrained_model", str(pretrained_ckpt)]
        log.info("Using pretrained weights: %s", pretrained_ckpt)
    else:
        log.warning("No pretrained checkpoint provided — training from scratch.")

    if num_gpus > 1:
        # Replace python with torchrun for multi-GPU
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            str(train_script),
        ] + cmd[2:]  # strip 'python <script>'

    log.info("Launching training:\n  %s", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(openpcdet_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(openpcdet_root / "tools"), env=env)


# ===========================================================================
# STEP 8 — Evaluation
# ===========================================================================

def evaluate(openpcdet_root: Path,
             model_yaml: Path,
             checkpoint: Path,
             output_dir: Path,
             batch_size: int = 4,
             extra_tag: str = "eval") -> None:
    """Run tools/test.py to compute mAP on the val set."""
    test_script = openpcdet_root / "tools" / "test.py"
    cmd = [
        sys.executable, str(test_script),
        "--cfg_file",   str(model_yaml),
        "--batch_size", str(batch_size),
        "--ckpt",       str(checkpoint),
        "--output_dir", str(output_dir),
        "--extra_tag",  extra_tag,
        "--eval_all",
    ]
    log.info("Running evaluation:\n  %s", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(openpcdet_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(openpcdet_root / "tools"), env=env)


# ===========================================================================
# STEP 9 — Inference / demo on new point clouds
# ===========================================================================

def run_inference(openpcdet_root: Path,
                  model_yaml: Path,
                  checkpoint: Path,
                  point_cloud_path: str,
                  output_dir: Path) -> None:
    """
    Run tools/demo.py on a single point cloud file and save visualisation.
    Accepts .bin, .npy, .pcd, .ply.
    """
    demo_script = openpcdet_root / "tools" / "demo.py"
    ext = Path(point_cloud_path).suffix.lower()

    # demo.py expects a .bin file; convert if needed
    tmp_bin = None
    if ext != ".bin":
        tmp_npy = Path("/tmp/_demo_cloud.npy")
        if ext in (".pcd", ".ply"):
            pts = load_pcd_open3d(point_cloud_path)
        elif ext == ".npy":
            pts = np.load(point_cloud_path).astype(np.float32)
        else:
            raise ValueError(f"Unsupported format for inference: {ext}")
        np.save(str(tmp_npy), pts)
        # demo.py loads .bin; write a bin
        tmp_bin = Path("/tmp/_demo_cloud.bin")
        pts.tofile(str(tmp_bin))
        point_cloud_path = str(tmp_bin)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(demo_script),
        "--cfg_file",      str(model_yaml),
        "--ckpt",          str(checkpoint),
        "--data_path",     point_cloud_path,
        "--ext",           ".bin",
        "--output_dir",    str(output_dir),
    ]
    log.info("Running inference:\n  %s", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(openpcdet_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(openpcdet_root / "tools"), env=env)

    if tmp_bin:
        tmp_bin.unlink(missing_ok=True)


# ===========================================================================
# STEP 10 — Synthetic annotation helper (semi-automatic labelling stub)
# ===========================================================================

def semi_automatic_label(points: np.ndarray,
                          min_cluster_pts: int = 5,
                          eps: float = 0.4,
                          voxel_size: float = 0.1) -> list[dict]:
    """
    Very simple DBSCAN-based cluster-then-label helper.
    For each cluster, build a tight oriented bounding box and assign a
    placeholder class for manual review.

    Returns list of dicts: {class_name, box_3d=[x,y,z,l,w,h,yaw], points}
    Install: pip install scikit-learn open3d
    """
    try:
        import open3d as o3d
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError("pip install open3d scikit-learn")

    xyz = points[:, :3]
    labels = DBSCAN(eps=eps, min_samples=min_cluster_pts).fit_predict(xyz)

    results = []
    for label_id in set(labels):
        if label_id < 0:
            continue  # noise
        mask = labels == label_id
        cluster_pts = xyz[mask]

        # Fit oriented bounding box with Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster_pts)
        obb = pcd.get_oriented_bounding_box()

        center = np.array(obb.center)
        extent = np.sort(np.array(obb.extent))          # [min→max]
        l, w, h = extent[2], extent[1], extent[0]
        # Extract yaw from rotation matrix (first axis in XY plane)
        R = np.array(obb.R)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))

        # Heuristic class guess based on geometry
        if h < 0.15 and l < 1.0:
            cls = "SilverPlate"
        elif h > 0.6 and l < 0.7:
            cls = "SafetyCone"
        else:
            cls = "Barricade"

        results.append({
            "class_name": cls,
            "box_3d": [float(center[0]), float(center[1]), float(center[2]),
                       float(l), float(w), float(h), float(yaw)],
            "num_points": int(mask.sum()),
        })

    log.info("Semi-auto labelling: found %d clusters.", len(results))
    return results


# ===========================================================================
# STEP 11 — Dataset statistics / sanity check
# ===========================================================================

def dataset_statistics(root: Path) -> dict:
    """Print per-class instance counts across all label files."""
    counts: dict[str, int] = {cls: 0 for cls in CLASS_NAMES}
    label_files = sorted((root / "labels").glob("*.txt"))
    if not label_files:
        log.warning("No label files found in %s", root / "labels")
        return counts

    for lf in label_files:
        _, names = parse_label_file(lf)
        for n in names:
            if n in counts:
                counts[n] += 1
            else:
                log.warning("Unknown class '%s' in %s", n, lf.name)

    log.info("Dataset statistics:")
    for cls, cnt in counts.items():
        log.info("  %-15s: %d instances", cls, cnt)
    return counts


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="OpenPCDet finetuning pipeline for SilverPlate / SafetyCone / Barricade"
    )
    p.add_argument("--stage", required=True,
                   choices=["prepare", "create_infos", "train", "eval", "infer", "all"],
                   help="Pipeline stage to run")
    p.add_argument("--openpcdet_root", default="./OpenPCDet",
                   help="Path to cloned OpenPCDet repository")
    p.add_argument("--data_root", default="./data/excavation",
                   help="Root directory for the custom dataset")
    p.add_argument("--raw_clouds_dir", default=None,
                   help="Directory containing raw point clouds (.bin/.pcd/.ply)")
    p.add_argument("--pretrained_ckpt", default=None,
                   help="Path to a pretrained checkpoint (e.g. nuScenes PointPillars)")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint to use for eval / inference")
    p.add_argument("--infer_file", default=None,
                   help="Single point cloud file for inference demo")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--output_dir", default="./output/excavation")
    return p.parse_args()


def main():
    args = parse_args()

    openpcdet_root = Path(args.openpcdet_root).resolve()
    data_root      = Path(args.data_root).resolve()
    output_dir     = Path(args.output_dir).resolve()

    # Generated config paths (inside the OpenPCDet tree so train.py finds them)
    dataset_yaml = openpcdet_root / "tools" / "cfgs" / "dataset_configs" / "excavation_dataset.yaml"
    model_yaml   = openpcdet_root / "tools" / "cfgs" / "custom_models" / "excavation_pointpillars.yaml"

    # -----------------------------------------------------------------------
    # PREPARE — create directories, ingest clouds, generate configs
    # -----------------------------------------------------------------------
    if args.stage in ("prepare", "all"):
        log.info("=== STAGE: prepare ===")
        create_dataset_layout(data_root)

        if args.raw_clouds_dir:
            sample_ids = batch_ingest_point_clouds(
                args.raw_clouds_dir, data_root / "points"
            )
            if sample_ids:
                generate_splits(sample_ids, data_root)
        else:
            log.warning(
                "--raw_clouds_dir not set. "
                "Put *.npy files in %s/points/ and populate ImageSets/ manually.",
                data_root
            )

        make_dataset_yaml(data_root, dataset_yaml)
        make_model_yaml(dataset_yaml, model_yaml)

        # Quick sanity check
        dataset_statistics(data_root)

    # -----------------------------------------------------------------------
    # CREATE INFOS — build .pkl info files + GT database
    # -----------------------------------------------------------------------
    if args.stage in ("create_infos", "all"):
        log.info("=== STAGE: create_infos ===")
        if not openpcdet_root.exists():
            raise FileNotFoundError(
                f"OpenPCDet root not found: {openpcdet_root}\n"
                "  git clone https://github.com/open-mmlab/OpenPCDet"
            )
        create_infos_and_gt_database(openpcdet_root, data_root, dataset_yaml)

    # -----------------------------------------------------------------------
    # TRAIN — finetune with optional pretrained weights
    # -----------------------------------------------------------------------
    if args.stage in ("train", "all"):
        log.info("=== STAGE: train ===")
        pretrained = Path(args.pretrained_ckpt) if args.pretrained_ckpt else None
        finetune(
            openpcdet_root=openpcdet_root,
            model_yaml=model_yaml,
            pretrained_ckpt=pretrained,
            output_dir=output_dir,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

    # -----------------------------------------------------------------------
    # EVAL — compute mAP on val set
    # -----------------------------------------------------------------------
    if args.stage in ("eval", "all"):
        log.info("=== STAGE: eval ===")
        if not args.checkpoint:
            # Try to find latest checkpoint automatically
            ckpts = sorted(output_dir.rglob("*.pth"))
            if ckpts:
                args.checkpoint = str(ckpts[-1])
                log.info("Auto-selected checkpoint: %s", args.checkpoint)
            else:
                raise ValueError("Provide --checkpoint for evaluation.")
        evaluate(
            openpcdet_root=openpcdet_root,
            model_yaml=model_yaml,
            checkpoint=Path(args.checkpoint),
            output_dir=output_dir / "eval",
            batch_size=args.batch_size,
        )

    # -----------------------------------------------------------------------
    # INFER — run demo on a single point cloud
    # -----------------------------------------------------------------------
    if args.stage in ("infer",):
        log.info("=== STAGE: infer ===")
        if not args.infer_file:
            raise ValueError("Provide --infer_file for inference.")
        if not args.checkpoint:
            raise ValueError("Provide --checkpoint for inference.")
        run_inference(
            openpcdet_root=openpcdet_root,
            model_yaml=model_yaml,
            checkpoint=Path(args.checkpoint),
            point_cloud_path=args.infer_file,
            output_dir=output_dir / "inference",
        )

    log.info("Done.")


if __name__ == "__main__":
    main()