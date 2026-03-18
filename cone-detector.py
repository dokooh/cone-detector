"""
cone_detector.py
================
Safety-cone detector for VGGT + Utonia point-cloud pipelines.

Works with:
  (A) A raw / full-scene point cloud  (.ply  or  Nx3 / Nx6 numpy array)
  (B) Pre-segmented clusters          (list of Nx3 / Nx6 numpy arrays)

Algorithm
---------
1. Estimate the ground plane with RANSAC (or PCA fallback).
2. Remove ground points → keep "above-ground" clusters.
3. For each cluster (or, in raw-scene mode, DBSCAN-cluster the survivors):
   a. Measure height above the ground plane.
   b. Measure footprint area on the ground plane (convex-hull / bounding-box).
   c. Score each cluster: prefer tall-relative-to-footprint, short overall,
      small footprint (cone-like aspect ratio).
4. Return clusters whose geometry falls inside the tunable cone envelope.

Usage
-----
# From segmentation output (list of numpy arrays):
    from cone_detector import ConeDetector
    detector = ConeDetector()
    cones = detector.detect_from_segments(segments)

# From a raw .ply file:
    cones = detector.detect_from_file("scene.ply")

# From a raw numpy array (Nx3 XYZ or Nx6 XYZRGB):
    cones = detector.detect_from_pointcloud(points)

# CLI:
    python cone_detector.py --input scene.ply --visualize
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── optional heavy deps ────────────────────────────────────────────────────────
try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConeDetectorConfig:
    """
    All thresholds in *metres* (assumes metric reconstruction).
    Adjust to match your scene scale if the reconstruction is not metric.
    """

    # ── Ground-plane estimation ───────────────────────────────────────────────
    ground_ransac_distance: float = 0.05   # inlier threshold for RANSAC
    ground_ransac_n: int = 3               # min pts per RANSAC hypothesis
    ground_ransac_iterations: int = 1000
    ground_percentile: float = 5.0         # fallback: lowest N-th percentile
    ground_thickness: float = 0.10         # pts within this distance = ground

    # ── DBSCAN (used in raw-scene mode) ──────────────────────────────────────
    dbscan_eps: float = 0.15              # neighbourhood radius
    dbscan_min_samples: int = 20

    # ── Cone geometry envelope (standard traffic cone ≈ 50–75 cm tall) ───────
    cone_min_height: float = 0.20          # shorter → not a cone
    cone_max_height: float = 1.20          # taller  → not a cone
    cone_min_footprint_area: float = 0.005 # m²  – avoids tiny noise blobs
    cone_max_footprint_area: float = 0.40  # m²  – avoids large objects
    cone_min_aspect_ratio: float = 1.5     # height/sqrt(footprint) – pointy
    cone_max_aspect_ratio: float = 15.0
    cone_ground_gap_max: float = 0.15      # base must be near ground (m)

    # ── Output ────────────────────────────────────────────────────────────────
    visualize: bool = False
    save_cones: bool = False
    output_dir: str = "cone_detections"


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.05,
    n: int = 3,
    max_iterations: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC plane fit. Returns (normal [3], inlier_mask [N]).
    Falls back to numpy SVD (PCA) if Open3D is unavailable.
    """
    if HAS_O3D:
        pcd = _np_to_o3d(points)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=n,
            num_iterations=max_iterations,
        )
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        signed_dist = (points @ normal + d) / np.linalg.norm(normal)
        mask = np.abs(signed_dist) < distance_threshold
        return normal, mask, float(d)

    # ── Pure-numpy PCA fallback ──────────────────────────────────────────────
    log.warning("open3d not found – using PCA plane fit (less robust).")
    return fit_plane_pca(points, distance_threshold)


def fit_plane_pca(
    points: np.ndarray,
    distance_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """PCA-based plane fit using the smallest eigenvector."""
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    normal = Vt[-1]                        # smallest singular-value direction
    d = -float(normal @ centroid)
    signed_dist = points @ normal + d
    mask = np.abs(signed_dist) < distance_threshold
    return normal, mask, d


def signed_height_above_plane(
    points: np.ndarray, normal: np.ndarray, d: float
) -> np.ndarray:
    """Signed distance of each point from the plane (positive = above)."""
    n_hat = normal / np.linalg.norm(normal)
    return points @ n_hat + d


def project_to_plane(
    points: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """
    Project points onto the plane (returns 2-D coords in the plane's local frame).
    Used for footprint-area estimation.
    """
    n_hat = normal / np.linalg.norm(normal)
    # Build an orthonormal basis in the plane
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n_hat, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n_hat, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    return np.column_stack([points @ u, points @ v])


def convex_hull_area_2d(pts2d: np.ndarray) -> float:
    """Shoelace formula area of 2-D convex hull."""
    try:
        from scipy.spatial import ConvexHull
        if len(pts2d) < 3:
            return 0.0
        hull = ConvexHull(pts2d)
        return float(hull.volume)          # in 2-D, .volume == area
    except Exception:
        # Bounding-box fallback
        ranges = pts2d.max(axis=0) - pts2d.min(axis=0)
        return float(ranges[0] * ranges[1])


def _np_to_o3d(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] == 6:
        colors = points[:, 3:6]
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ════════════���═════════════════════════════════════════════════════════════════
# Core detector
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectedCone:
    """Result object for one detected safety cone."""
    cluster_id: int
    points: np.ndarray                  # Nx3 or Nx6
    centroid: np.ndarray                # (3,) world-space centroid
    height_m: float                     # height above ground plane
    footprint_area_m2: float
    aspect_ratio: float                 # height / sqrt(footprint)
    ground_gap_m: float                 # distance of lowest point to ground
    score: float                        # higher = more cone-like
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __repr__(self) -> str:
        return (
            f"Cone #{self.cluster_id:03d}  "
            f"centroid=({self.centroid[0]:.2f}, {self.centroid[1]:.2f}, {self.centroid[2]:.2f})  "
            f"h={self.height_m:.2f}m  "
            f"foot={self.footprint_area_m2:.3f}m²  "
            f"aspect={self.aspect_ratio:.1f}  "
            f"score={self.score:.3f}"
        )


class ConeDetector:
    """
    Identify safety cones from a VGGT point cloud reconstruction.

    Parameters
    ----------
    config : ConeDetectorConfig
        Tunable geometry thresholds (see class docstring).
    """

    def __init__(self, config: Optional[ConeDetectorConfig] = None):
        self.cfg = config or ConeDetectorConfig()
        self._ground_normal: Optional[np.ndarray] = None
        self._ground_d: Optional[float] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def detect_from_file(self, path: str) -> List[DetectedCone]:
        """Load a .ply / .pcd file and detect cones."""
        points = self._load_pointcloud(path)
        return self.detect_from_pointcloud(points)

    def detect_from_pointcloud(self, points: np.ndarray) -> List[DetectedCone]:
        """
        Detect cones from a raw (unsegmented) Nx3 or Nx6 point cloud.
        Internally runs DBSCAN to form clusters then evaluates each.
        """
        log.info(f"Raw-scene mode: {len(points):,} points")
        xyz = points[:, :3]

        # 1. Ground plane
        normal, ground_mask, d = self._estimate_ground(xyz)
        self._ground_normal = normal
        self._ground_d = d
        log.info(f"Ground plane normal={normal.round(3)}, d={d:.3f}  "
                 f"({ground_mask.sum():,} ground pts)")

        # 2. Above-ground points
        heights = signed_height_above_plane(xyz, normal, d)
        above_mask = (~ground_mask) & (heights > 0.01)
        above_pts = points[above_mask]
        above_heights = heights[above_mask]
        log.info(f"Above-ground points: {above_mask.sum():,}")

        if above_mask.sum() < self.cfg.dbscan_min_samples:
            log.warning("Too few above-ground points – no cones found.")
            return []

        # 3. Cluster
        clusters = self._cluster(above_pts)
        log.info(f"DBSCAN clusters: {len(clusters)}")

        # 4. Evaluate
        return self._evaluate_clusters(clusters, normal, d)

    def detect_from_segments(
        self, segments: List[np.ndarray]
    ) -> List[DetectedCone]:
        """
        Detect cones from pre-segmented clusters (e.g. Utonia / PCA output).

        Parameters
        ----------
        segments : list of np.ndarray
            Each array is Nx3 (XYZ) or Nx6 (XYZRGB) for one segment.
        """
        log.info(f"Segmented mode: {len(segments)} input segments")

        # Estimate ground from ALL segments combined
        all_pts = np.concatenate([s[:, :3] for s in segments], axis=0)
        normal, _, d = self._estimate_ground(all_pts)
        self._ground_normal = normal
        self._ground_d = d
        log.info(f"Ground plane normal={normal.round(3)}, d={d:.3f}")

        return self._evaluate_clusters(segments, normal, d)

    # ──────────────────────────────────────────────────────────────────────────
    # Ground estimation
    # ──────────────────────────────────────────────────────────────────────────

    def _estimate_ground(
        self, xyz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate the dominant ground plane.
        Uses RANSAC (via Open3D if available, else PCA).
        Falls back to lowest-percentile slab if RANSAC fails.
        """
        try:
            normal, mask, d = fit_plane_ransac(
                xyz,
                distance_threshold=self.cfg.ground_ransac_distance,
                n=self.cfg.ground_ransac_n,
                max_iterations=self.cfg.ground_ransac_iterations,
            )
            # Sanity: ground normal should be mostly vertical
            vertical = np.array([0.0, 0.0, 1.0])
            # Try both orientations
            if abs(np.dot(normal / np.linalg.norm(normal), vertical)) < 0.5:
                log.warning("RANSAC plane is not near-horizontal – falling back to PCA.")
                raise ValueError("Non-horizontal plane")
            # Ensure normal points upward
            if np.dot(normal, vertical) < 0:
                normal = -normal
                d = -d
            return normal, mask, d
        except Exception as exc:
            log.warning(f"Ground RANSAC failed ({exc}), using percentile fallback.")
            return self._ground_percentile_fallback(xyz)

    def _ground_percentile_fallback(
        self, xyz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fit a plane through the lowest-N-th-percentile points."""
        z_thresh = np.percentile(xyz[:, 2], self.cfg.ground_percentile)
        ground_pts = xyz[xyz[:, 2] <= z_thresh + self.cfg.ground_thickness]
        normal, mask, d = fit_plane_pca(xyz, self.cfg.ground_ransac_distance)
        if np.dot(normal, np.array([0.0, 0.0, 1.0])) < 0:
            normal = -normal
            d = -d
        return normal, mask, d

    # ──────────────────────────────────────────────────────────────────────────
    # DBSCAN clustering (raw-scene mode only)
    # ──────────────────────────────────────────────────────────────────────────

    def _cluster(self, points: np.ndarray) -> List[np.ndarray]:
        """Return a list of cluster point arrays using DBSCAN."""
        if not HAS_SKLEARN and not HAS_O3D:
            raise RuntimeError(
                "scikit-learn or open3d is required for DBSCAN clustering.\n"
                "Install with:  pip install scikit-learn  or  pip install open3d"
            )

        xyz = points[:, :3]

        if HAS_O3D:
            pcd = _np_to_o3d(points)
            labels = np.array(
                pcd.cluster_dbscan(
                    eps=self.cfg.dbscan_eps,
                    min_points=self.cfg.dbscan_min_samples,
                    print_progress=False,
                )
            )
        else:
            labels = DBSCAN(
                eps=self.cfg.dbscan_eps,
                min_samples=self.cfg.dbscan_min_samples,
            ).fit_predict(xyz)

        unique_labels = set(labels) - {-1}
        clusters = [points[labels == lbl] for lbl in unique_labels]
        log.info(f"  {len(clusters)} clusters found (noise pts excluded)")
        return clusters

    # ──────────────────────────────────────────────────────────────────────────
    # Cluster evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_clusters(
        self,
        clusters: List[np.ndarray],
        normal: np.ndarray,
        d: float,
    ) -> List[DetectedCone]:
        """Score every cluster and return those matching cone geometry."""
        cones: List[DetectedCone] = []
        cfg = self.cfg

        for cid, cluster in enumerate(clusters):
            xyz = cluster[:, :3]
            if len(xyz) < 5:
                continue

            heights = signed_height_above_plane(xyz, normal, d)
            min_h = float(heights.min())
            max_h = float(heights.max())
            cluster_height = max_h - min_h

            # ── Filter: ground-gap (base close to ground) ─────────────────
            ground_gap = abs(min_h)  # how far above ground is the lowest pt
            if ground_gap > cfg.cone_ground_gap_max:
                continue

            # ── Filter: height range ──────────────────────────────────────
            if not (cfg.cone_min_height <= cluster_height <= cfg.cone_max_height):
                continue

            # ── Footprint area ────────────────────────────────────────────
            pts2d = project_to_plane(xyz, normal)
            footprint = convex_hull_area_2d(pts2d)
            if not (cfg.cone_min_footprint_area <= footprint <= cfg.cone_max_footprint_area):
                continue

            # ── Aspect ratio: height / sqrt(footprint) ────────────────────
            aspect = cluster_height / (np.sqrt(footprint) + 1e-9)
            if not (cfg.cone_min_aspect_ratio <= aspect <= cfg.cone_max_aspect_ratio):
                continue

            # ── Score (higher = more cone-like) ──────────────────────────
            #   • Reward high aspect ratio (pointy)
            #   • Reward small footprint
            #   • Reward height near typical cone height (0.5 m)
            height_score = np.exp(-0.5 * ((cluster_height - 0.55) / 0.25) ** 2)
            aspect_score = np.tanh((aspect - cfg.cone_min_aspect_ratio) / 2.0)
            foot_score = np.exp(-footprint / 0.1)
            score = float((height_score + aspect_score + foot_score) / 3.0)

            centroid = xyz.mean(axis=0)
            cone = DetectedCone(
                cluster_id=cid,
                points=cluster,
                centroid=centroid,
                height_m=cluster_height,
                footprint_area_m2=footprint,
                aspect_ratio=aspect,
                ground_gap_m=ground_gap,
                score=score,
                bbox_min=xyz.min(axis=0),
                bbox_max=xyz.max(axis=0),
            )
            cones.append(cone)
            log.info(f"  ✓ {cone}")

        # Sort by score descending
        cones.sort(key=lambda c: c.score, reverse=True)
        log.info(f"\n{'═'*60}")
        log.info(f"Detected {len(cones)} safety cone(s).")
        return cones

    # ──────────────────────────────────────────────────────────────────────────
    # I/O helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_pointcloud(self, path: str) -> np.ndarray:
        """Load a point cloud from .ply / .pcd → Nx3 or Nx6 numpy array."""
        if not HAS_O3D:
            raise RuntimeError(
                "open3d is required to load .ply/.pcd files.\n"
                "Install with:  pip install open3d"
            )
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors) * 255.0
            pts = np.hstack([pts, colors])
        log.info(f"Loaded '{path}': {len(pts):,} points")
        return pts

    def save_cones(self, cones: List[DetectedCone], output_dir: str = "cone_detections"):
        """Save each detected cone as a separate .ply file."""
        if not HAS_O3D:
            log.warning("open3d required to save .ply files – skipping.")
            return
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for cone in cones:
            pcd = _np_to_o3d(cone.points)
            fname = out / f"cone_{cone.cluster_id:03d}_h{cone.height_m:.2f}m.ply"
            o3d.io.write_point_cloud(str(fname), pcd)
            log.info(f"Saved {fname}")

    def visualize(self, all_points: np.ndarray, cones: List[DetectedCone]):
        """
        Open an Open3D visualiser showing:
          • Original scene (grey)
          • Ground plane inliers (brown)
          • Detected cones (bright orange / per-cone colour)
        """
        if not HAS_O3D:
            log.warning("open3d required for visualisation.")
            return

        geometries = []

        # Full scene (grey)
        scene_pcd = _np_to_o3d(all_points)
        scene_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        geometries.append(scene_pcd)

        # Cone clusters (bright colours)
        palette = [
            [1.0, 0.40, 0.0],   # orange
            [1.0, 0.85, 0.0],   # yellow
            [0.8, 0.0,  0.0],   # red
            [0.0, 0.8,  0.0],   # green
            [0.0, 0.5,  1.0],   # blue
        ]
        for i, cone in enumerate(cones):
            pcd = _np_to_o3d(cone.points)
            pcd.paint_uniform_color(palette[i % len(palette)])
            geometries.append(pcd)

            # Bounding box
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=cone.bbox_min,
                max_bound=cone.bbox_max,
            )
            bbox.color = palette[i % len(palette)]
            geometries.append(bbox)

        o3d.visualization.draw_geometries(
            geometries,
            window_name="Safety Cone Detector – VGGT Pipeline",
            width=1280,
            height=720,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect safety cones in a VGGT point cloud reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Path to .ply / .pcd scene file")
    src.add_argument(
        "--npy_dir",
        type=str,
        help="Directory of .npy segment files (one file per cluster)",
    )

    p.add_argument("--visualize", action="store_true", help="Open 3-D visualiser")
    p.add_argument("--save", action="store_true", help="Save cone .ply files")
    p.add_argument("--output_dir", type=str, default="cone_detections")

    # Geometry knobs
    p.add_argument("--min_height", type=float, default=0.20, help="Min cone height (m)")
    p.add_argument("--max_height", type=float, default=1.20, help="Max cone height (m)")
    p.add_argument("--min_foot",   type=float, default=0.005, help="Min footprint area (m²)")
    p.add_argument("--max_foot",   type=float, default=0.40,  help="Max footprint area (m²)")
    p.add_argument("--min_aspect", type=float, default=1.5,   help="Min aspect ratio")
    p.add_argument("--ground_gap", type=float, default=0.15,  help="Max ground gap (m)")
    p.add_argument("--ground_dist",type=float, default=0.05,  help="RANSAC ground distance (m)")
    p.add_argument("--dbscan_eps", type=float, default=0.15,  help="DBSCAN epsilon (m)")
    p.add_argument("--dbscan_min", type=int,   default=20,    help="DBSCAN min samples")
    return p


def main():
    args = build_arg_parser().parse_args()

    cfg = ConeDetectorConfig(
        cone_min_height=args.min_height,
        cone_max_height=args.max_height,
        cone_min_footprint_area=args.min_foot,
        cone_max_footprint_area=args.max_foot,
        cone_min_aspect_ratio=args.min_aspect,
        cone_ground_gap_max=args.ground_gap,
        ground_ransac_distance=args.ground_dist,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min,
        visualize=args.visualize,
        save_cones=args.save,
        output_dir=args.output_dir,
    )
    detector = ConeDetector(cfg)

    if args.input:
        # ── Mode A: raw scene file ─────────────────────────────────────────
        cones = detector.detect_from_file(args.input)
        raw_pts = detector._load_pointcloud(args.input)
    else:
        # ── Mode B: pre-segmented .npy clusters ────────────────────────────
        npy_dir = Path(args.npy_dir)
        segments = [np.load(str(f)) for f in sorted(npy_dir.glob("*.npy"))]
        log.info(f"Loaded {len(segments)} segments from {npy_dir}")
        cones = detector.detect_from_segments(segments)
        raw_pts = np.concatenate([s[:, :3] for s in segments], axis=0)

    print("\n" + "═" * 60)
    print(f"  DETECTED {len(cones)} SAFETY CONE(S)")
    print("═" * 60)
    for c in cones:
        print(f"  {c}")

    if args.save:
        detector.save_cones(cones, output_dir=args.output_dir)

    if args.visualize:
        detector.visualize(raw_pts, cones)


if __name__ == "__main__":
    main()