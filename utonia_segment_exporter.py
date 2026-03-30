"""
utonia_segment_exporter.py
==========================
Reads a color-segmented point cloud produced by Utonia's PCA pipeline
(.ply with per-point RGB), groups points by their unique segment color,
writes each segment to its own .ply file, and prints the height/width
of every segmented object.

Usage
-----
    python utonia_segment_exporter.py --input segmented_scene.ply
    python utonia_segment_exporter.py --input segmented_scene.ply --output_dir my_segments --tolerance 3

Requirements
------------
    pip install open3d numpy
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d is not installed.  Run:  pip install open3d")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration defaults
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR = "segments_out"
DEFAULT_COLOR_TOLERANCE = 2   # uint8 bucket size (0 = exact match)
DEFAULT_MIN_POINTS = 5        # segments smaller than this are discarded


# ══════════════════════════════════════════════════════════════════════════════
# Core logic
# ══════════════════════════════════════════════════════════════════════════════

def load_pointcloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a .ply file and return (xyz, colors).

    xyz    : (N, 3) float64  — XYZ coordinates
    colors : (N, 3) float64  — RGB in [0, 1] as stored by Open3D
    """
    pcd = o3d.io.read_point_cloud(str(path))

    if len(pcd.points) == 0:
        sys.exit(f"ERROR: Point cloud is empty: {path}")
    if not pcd.has_colors():
        sys.exit(
            "ERROR: The point cloud has no color information.\n"
            "Please supply the Utonia / PCA color-segmented output (.ply with RGB)."
        )

    xyz = np.asarray(pcd.points)    # (N, 3)
    colors = np.asarray(pcd.colors) # (N, 3)  float64 in [0, 1]

    print(f"Loaded  : {path.name}")
    print(f"Points  : {len(xyz):,}")
    print(
        f"XYZ range  "
        f"X=[{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]  "
        f"Y=[{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]  "
        f"Z=[{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]"
    )
    return xyz, colors


def group_by_color(
    colors: np.ndarray,
    tolerance: int = DEFAULT_COLOR_TOLERANCE,
    min_points: int = DEFAULT_MIN_POINTS,
) -> dict[tuple[int, int, int], np.ndarray]:
    """
    Group point indices by their quantised RGB color.

    Utonia assigns a unique flat RGB to each PCA segment, so every group
    of identically-colored points corresponds to one segment.

    Parameters
    ----------
    colors    : (N, 3) float64 in [0, 1]
    tolerance : bucket size in uint8 space (0 = exact match, 2-5 = tolerant)
    min_points: groups smaller than this are discarded as noise

    Returns
    -------
    dict mapping (R, G, B) uint8 tuple → array of point indices
    """
    # float [0, 1] → uint8 [0, 255]
    colors_u8 = (colors * 255.0).round().astype(np.uint8)

    if tolerance > 0:
        colors_q = (colors_u8 // tolerance) * tolerance
    else:
        colors_q = colors_u8

    groups: dict[tuple, list[int]] = defaultdict(list)
    for idx, (r, g, b) in enumerate(colors_q):
        groups[(int(r), int(g), int(b))].append(idx)

    segments = {
        color: np.array(indices)
        for color, indices in groups.items()
        if len(indices) >= min_points
    }

    n_total = len(groups)
    n_valid = len(segments)
    print(f"\nUnique color groups        : {n_total}")
    print(f"Valid segments (≥{min_points} pts) : {n_valid}")
    print(f"Dropped (too small)        : {n_total - n_valid}")

    return segments


def save_segments(
    xyz: np.ndarray,
    colors: np.ndarray,
    segments: dict[tuple[int, int, int], np.ndarray],
    output_dir: Path,
) -> list[Path]:
    """
    Write each segment as an individual .ply file.

    File naming convention:
        segment_<id>_rgb<R>-<G>-<B>.ply
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for seg_id, (color_key, indices) in enumerate(segments.items()):
        seg_xyz = xyz[indices]
        seg_colors = colors[indices]  # float [0, 1] — Open3D expects this

        seg_pcd = o3d.geometry.PointCloud()
        seg_pcd.points = o3d.utility.Vector3dVector(seg_xyz)
        seg_pcd.colors = o3d.utility.Vector3dVector(seg_colors)

        r, g, b = color_key
        out_file = output_dir / f"segment_{seg_id:04d}_rgb{r:03d}-{g:03d}-{b:03d}.ply"
        o3d.io.write_point_cloud(str(out_file), seg_pcd)
        saved.append(out_file)

    print(f"\nSaved {len(saved)} segment file(s) → '{output_dir.resolve()}'")
    return saved


def print_dimensions(
    xyz: np.ndarray,
    segments: dict[tuple[int, int, int], np.ndarray],
    saved_paths: list[Path],
) -> None:
    """
    Print the bounding-box dimensions of every segment.

    Definitions
    -----------
    height : Z-axis extent  (z_max − z_min)
    width  : X-axis extent  (x_max − x_min)
    depth  : Y-axis extent  (y_max − y_min)  — reported for completeness
    """
    col_w = 72
    print("\n" + "═" * col_w)
    print(f"  SEGMENT DIMENSIONS  ({len(segments)} object(s))")
    print("═" * col_w)
    print(
        f"{'ID':>4}  {'Color (R,G,B)':>16}  {'Pts':>7}"
        f"  {'Width(X)':>9}  {'Depth(Y)':>9}  {'Height(Z)':>10}"
    )
    print("─" * col_w)

    for seg_id, (color_key, indices) in enumerate(segments.items()):
        seg_xyz = xyz[indices]
        r, g, b = color_key

        x_min, x_max = float(seg_xyz[:, 0].min()), float(seg_xyz[:, 0].max())
        y_min, y_max = float(seg_xyz[:, 1].min()), float(seg_xyz[:, 1].max())
        z_min, z_max = float(seg_xyz[:, 2].min()), float(seg_xyz[:, 2].max())

        width  = x_max - x_min   # X-extent  → "width"
        depth  = y_max - y_min   # Y-extent  → "depth"
        height = z_max - z_min   # Z-extent  → "height"

        color_str = f"({r:3d},{g:3d},{b:3d})"
        print(
            f"{seg_id:>4}  {color_str:>16}  {len(indices):>7,}"
            f"  {width:>9.4f}  {depth:>9.4f}  {height:>10.4f}"
        )
        print(
            f"{'':>4}  {'↳ file':>16}: {saved_paths[seg_id].name}"
        )

    print("═" * col_w)
    print("  Units match the coordinate units of the input point cloud.")
    print("═" * col_w)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export segments from a Utonia / PCA color-segmented point cloud "
            "and report their height & width."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to the Utonia PCA color-segmented .ply file.",
    )
    p.add_argument(
        "--output_dir", "-o", default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-segment .ply files will be written.",
    )
    p.add_argument(
        "--tolerance", "-t", type=int, default=DEFAULT_COLOR_TOLERANCE,
        help=(
            "RGB quantisation tolerance (uint8). "
            "0 = exact color match; 2-5 handles minor float-rounding differences."
        ),
    )
    p.add_argument(
        "--min_points", "-m", type=int, default=DEFAULT_MIN_POINTS,
        help="Minimum number of points for a segment to be kept.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: File not found: {input_path.resolve()}")

    # 1. Load
    xyz, colors = load_pointcloud(input_path)

    # 2. Group by segment color
    segments = group_by_color(colors, args.tolerance, args.min_points)

    if not segments:
        sys.exit("No valid segments found.  Try lowering --min_points or --tolerance.")

    # 3. Save each segment as .ply
    saved_paths = save_segments(xyz, colors, segments, Path(args.output_dir))

    # 4. Print height / width table
    print_dimensions(xyz, segments, saved_paths)


if __name__ == "__main__":
    main()
