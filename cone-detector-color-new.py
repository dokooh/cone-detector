def main():
    # ── Declare globals FIRST, before any use ──
    global MAX_CONE_SIZE, DBSCAN_EPS

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
    MAX_CONE_SIZE = args.max_size
    DBSCAN_EPS    = args.dbscan_eps

    # ── Pipeline ──────────────────────────────
    pcd                     = load_point_cloud(args.input)
    above_pcd, plane_model  = remove_ground(pcd)
    colored_pcd             = filter_by_color(above_pcd)

    if not colored_pcd.has_points():
        print("[result] No orange or white points found. Exiting.")
        sys.exit(0)

    cone_clusters = cluster_and_filter(colored_pcd)

    if not cone_clusters:
        print("[result] No cone-sized objects detected.")
    else:
        print(f"[result] ✓ {len(cone_clusters)} cone(s) detected.")

    if args.export:
        export_clusters(cone_clusters, base_path=args.export_prefix)

    if not args.no_viz:
        visualize(pcd, cone_clusters, plane_model)
