"""
GLB/GLTF to PLY converter.

Handles GLB files with mixed geometry types:
  - trimesh.PointCloud  → extracted directly (XYZ + colors)
  - trimesh.Trimesh     → surface-sampled or exported as mesh
  - trimesh.Scene       → all sub-geometries processed with world-space transforms

Output modes:
    --mode pointcloud   Extract/sample points from all geometries (default)
    --mode mesh         Export Trimesh faces as a triangle mesh (PointClouds
                        are still written as points, without faces)

Dependencies:
    pip install trimesh numpy scipy Pillow

Usage:
    python glb_to_ply.py input.glb output.ply
    python glb_to_ply.py input.glb output.ply --num-points 5000000
    python glb_to_ply.py input.glb output.ply --mode mesh
    python glb_to_ply.py input.glb output.ply --no-markers
"""

import argparse
import sys
import struct
import numpy as np
from pathlib import Path

try:
    import trimesh
except ImportError:
    sys.exit("Missing dependency: pip install trimesh")



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint8_colors(colors: np.ndarray) -> np.ndarray:
    """Normalise any color array to (N, 4) uint8 RGBA."""
    colors = np.asarray(colors)
    if colors.dtype.kind == "f":
        colors = (colors * 255).clip(0, 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)
    if colors.shape[1] == 3:
        colors = np.column_stack([colors, np.full(len(colors), 255, np.uint8)])
    return colors


def _mesh_vertex_colors(mesh: "trimesh.Trimesh") -> "np.ndarray | None":
    """Return (N, 4) uint8 RGBA from a Trimesh's visual, or None."""
    visual = mesh.visual
    if visual is None:
        return None
    try:
        vc = visual.vertex_colors
        if vc is not None and len(vc) == len(mesh.vertices):
            return _to_uint8_colors(vc)
    except Exception:
        pass
    return None


def _collect_geometries(loaded) -> list:
    """
    Return a list of (geometry, transform_4x4) tuples covering every
    PointCloud and Trimesh in the loaded object.
    """
    identity = np.eye(4)

    if isinstance(loaded, trimesh.Scene):
        result = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            geom = loaded.geometry[geom_name]
            if isinstance(geom, (trimesh.Trimesh, trimesh.PointCloud)):
                result.append((geom, transform))
        return result

    if isinstance(loaded, (trimesh.Trimesh, trimesh.PointCloud)):
        return [(loaded, identity)]

    return []


def _write_ply(path: str,
               points:  np.ndarray,
               colors:  "np.ndarray | None",
               normals: "np.ndarray | None",
               faces:   "np.ndarray | None") -> None:
    """Write a binary-little-endian PLY file."""
    n_verts     = len(points)
    has_color   = colors  is not None
    has_normals = normals is not None
    has_faces   = faces   is not None

    # ---- header --------------------------------------------------------
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Converted by glb_to_ply.py",
        f"element vertex {n_verts}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_normals:
        lines += ["property float nx", "property float ny", "property float nz"]
    if has_color:
        lines += ["property uchar red", "property uchar green",
                  "property uchar blue", "property uchar alpha"]
    if has_faces:
        lines += [f"element face {len(faces)}",
                  "property list uchar int vertex_indices"]
    lines.append("end_header")
    header = "\n".join(lines) + "\n"

    # ---- vertex buffer -------------------------------------------------
    dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_normals:
        dtype_fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if has_color:
        dtype_fields += [("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")]

    buf = np.zeros(n_verts, dtype=np.dtype(dtype_fields))
    buf["x"] = points[:, 0].astype(np.float32)
    buf["y"] = points[:, 1].astype(np.float32)
    buf["z"] = points[:, 2].astype(np.float32)
    if has_normals:
        buf["nx"] = normals[:, 0].astype(np.float32)
        buf["ny"] = normals[:, 1].astype(np.float32)
        buf["nz"] = normals[:, 2].astype(np.float32)
    if has_color:
        c = np.asarray(colors, dtype=np.uint8)
        buf["r"] = c[:, 0]
        buf["g"] = c[:, 1]
        buf["b"] = c[:, 2]
        buf["a"] = c[:, 3] if c.shape[1] == 4 else 255

    # ---- write ---------------------------------------------------------
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(buf.tobytes())
        if has_faces:
            for tri in faces.astype(np.int32):
                f.write(struct.pack("<Biii", 3, *tri))


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert(glb_path: str, ply_path: str,
            mode: str = "pointcloud",
            num_points: int = 2_000_000,
            include_markers: bool = True) -> None:
    """
    Load *glb_path* and write a PLY point cloud / mesh to *ply_path*.

    Parameters
    ----------
    glb_path:         Source GLB/GLTF file.
    ply_path:         Destination PLY file.
    mode:             'pointcloud' or 'mesh'.
    num_points:       Target sample count (pointcloud mode, spread across meshes).
    include_markers:  If False, skip single-colour Trimesh objects that appear
                      to be annotation markers (all vertices same colour).
    """
    print(f"Loading: {glb_path}")
    loaded = trimesh.load(glb_path, process=False)
    print(f"Loaded as: {type(loaded).__name__}")

    geom_list = _collect_geometries(loaded)
    if not geom_list:
        sys.exit("No usable geometry found in the GLB file.")

    # ---- separate point clouds from meshes ----------------------------
    pc_geoms    = [(g, T) for g, T in geom_list if isinstance(g, trimesh.PointCloud)]
    mesh_geoms  = [(g, T) for g, T in geom_list if isinstance(g, trimesh.Trimesh)]

    if include_markers is False:
        # Drop meshes where all vertices share one color (marker objects)
        filtered = []
        for g, T in mesh_geoms:
            vc = _mesh_vertex_colors(g)
            if vc is not None and len(np.unique(vc, axis=0)) == 1:
                continue   # skip single-colour marker
            filtered.append((g, T))
        dropped = len(mesh_geoms) - len(filtered)
        if dropped:
            print(f"Skipped {dropped} single-colour marker mesh(es).")
        mesh_geoms = filtered

    print(f"PointCloud geometries : {len(pc_geoms)}")
    print(f"Trimesh geometries    : {len(mesh_geoms)}")

    all_points  = []
    all_colors  = []
    all_normals = []
    all_faces   = []
    vertex_offset = 0

    # ---- 1. PointCloud geometries -------------------------------------
    for pc, T in pc_geoms:
        verts = np.asarray(pc.vertices, dtype=np.float64)
        # Apply world-space transform
        ones = np.ones((len(verts), 1))
        verts_h = np.hstack([verts, ones])
        verts = (T @ verts_h.T).T[:, :3]

        vc = pc.colors
        if vc is not None and len(vc) == len(verts):
            colors = _to_uint8_colors(vc)
        else:
            colors = np.full((len(verts), 4), [200, 200, 200, 255], dtype=np.uint8)

        all_points.append(verts)
        all_colors.append(colors)
        all_normals.append(None)   # PointClouds have no normals
        print(f"  PointCloud: {len(verts):,} points  (colors: {'yes' if vc is not None else 'no'})")

    # ---- 2. Trimesh geometries ----------------------------------------
    total_mesh_verts = sum(len(g.vertices) for g, _ in mesh_geoms)
    for g, T in mesh_geoms:
        verts = np.asarray(g.vertices, dtype=np.float64)
        # Apply world-space transform
        ones = np.ones((len(verts), 1))
        verts_h = np.hstack([verts, ones])
        verts = (T @ verts_h.T).T[:, :3]

        vc = _mesh_vertex_colors(g)
        if vc is None:
            vc = np.full((len(verts), 4), [200, 200, 200, 255], dtype=np.uint8)

        fn = np.asarray(g.face_normals, dtype=np.float32)   # always geometrically valid

        if mode == "mesh":
            faces = np.asarray(g.faces, dtype=np.int32) + vertex_offset
            all_faces.append(faces)
            # Per-vertex normals: average the face normals for each vertex
            vn = np.zeros((len(verts), 3), dtype=np.float32)
            np.add.at(vn, g.faces[:, 0], fn)
            np.add.at(vn, g.faces[:, 1], fn)
            np.add.at(vn, g.faces[:, 2], fn)
            norms = np.linalg.norm(vn, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vn /= norms

            all_points.append(verts)
            all_colors.append(vc)
            all_normals.append(vn)
            vertex_offset += len(verts)
            print(f"  Trimesh (mesh): {len(verts):,} verts, {len(g.faces):,} faces")

        else:  # pointcloud mode — sample surface
            # Proportional sample count for this mesh
            n = max(1, int(num_points * len(g.vertices) / max(total_mesh_verts, 1)))
            pts, fidx, s_colors = trimesh.sample.sample_surface(g, n, sample_color=True)

            # Apply transform to sampled points
            ones = np.ones((len(pts), 1))
            pts_h = np.hstack([pts, ones])
            pts = (T @ pts_h.T).T[:, :3]

            if s_colors is not None and len(s_colors) > 0:
                s_colors = _to_uint8_colors(s_colors)
                # Fall back to solid vertex colour if sampling gave uniform colour
                if len(np.unique(s_colors, axis=0)) == 1 and vc is not None:
                    # Use the mesh solid colour for all samples
                    s_colors = np.tile(vc[0:1], (len(pts), 1))
            else:
                s_colors = np.tile(vc[0:1], (len(pts), 1))

            normals = fn[fidx]   # face normal per sampled point
            all_points.append(pts)
            all_colors.append(s_colors)
            all_normals.append(normals)
            print(f"  Trimesh (sampled): {len(pts):,} pts from {len(g.vertices):,} verts")

    if not all_points:
        sys.exit("No points were extracted.")

    # ---- combine -------------------------------------------------------
    points  = np.vstack(all_points).astype(np.float32)
    colors  = np.vstack(all_colors).astype(np.uint8)

    # Normals: combine where available, None where not (PointClouds)
    if any(n is not None for n in all_normals):
        normals_list = []
        for pts_arr, n_arr in zip(all_points, all_normals):
            if n_arr is None:
                normals_list.append(np.zeros((len(pts_arr), 3), dtype=np.float32))
            else:
                normals_list.append(n_arr)
        normals = np.vstack(normals_list).astype(np.float32)
    else:
        normals = None

    faces = np.vstack(all_faces).astype(np.int32) if all_faces else None

    print(f"\nTotal points : {len(points):,}")
    print(f"Has colors   : yes")
    print(f"Has normals  : {'yes' if normals is not None else 'no'}")
    print(f"Has faces    : {'yes (%d)' % len(faces) if faces is not None else 'no'}")

    _write_ply(ply_path, points, colors, normals, faces)
    print(f"\nSaved  →  {ply_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a GLB/GLTF file to PLY. Handles PointCloud and Trimesh geometries."
    )
    parser.add_argument("input",  help="Source .glb or .gltf file")
    parser.add_argument("output", help="Destination .ply file")
    parser.add_argument(
        "--mode",
        choices=["pointcloud", "mesh"],
        default="pointcloud",
        help="'pointcloud' (default) or 'mesh' (preserve triangle faces)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2_000_000,
        help="Points to sample from mesh geometries in pointcloud mode (default: 2 000 000)",
    )
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Skip single-colour Trimesh annotation markers",
    )
    args = parser.parse_args()
    convert(args.input, args.output,
            mode=args.mode,
            num_points=args.num_points,
            include_markers=not args.no_markers)


if __name__ == "__main__":
    main()
