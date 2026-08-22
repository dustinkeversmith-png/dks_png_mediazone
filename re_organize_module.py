#!/usr/bin/env python3
import shutil
import sys
from pathlib import Path

# Mapping: relative source under 'src/video/modules/' -> destination under 'src/video/modules/'
MODULE_MAPPING = {
    # contour
    "boundary_tracing/dual_contouring.cpp": "contour/dual_contouring",
    "boundary_tracing/dual_contouring.hpp": "contour/dual_contouring",
    "featurizations/laplace_gaussian": "contour/laplace_gaussian",
    "boundary_tracing/level_set.cpp": "contour/level_set",
    "boundary_tracing/level_set.hpp": "contour/level_set",
    "boundary_tracing/live_wire.cpp": "contour/live_wire",
    "boundary_tracing/live_wire.hpp": "contour/live_wire",
    "boundary_tracing/marching_squares.hpp": "contour/marching_squares",
    "boundary_tracing/moore_neighbor.hpp": "contour/moore_neighborhood",
    "vectorization_geometry/rdp": "contour/rdp",
    "boundary_tracing/snakes.cpp": "contour/snakes",
    "boundary_tracing/snakes.hpp": "contour/snakes",

    # features
    "filters/crease_sink": "features/crease_sink",
    "featurizations/rbf": "features/rbf",
    "vectorization_geometry/measurements/turning_function": "features/turning_function",

    # filters
    "filters/bilateral.hpp": "filters/bilateral",
    "featurizations/edge/canny.hpp": "filters/edge/canny",
    "featurizations/gvh": "filters/gvf",
    "filters/hue_gradient": "filters/hue_gradient",
    "color/lab_color_space.hpp": "filters/lab_color",
    "filters/morph_clean": "filters/morph_clean",
    "filters/sobel.cpp": "filters/sobel",
    "filters/sobel.hpp": "filters/sobel",

    # geometrify
    "featurizations/fourier": "geometrify/fourier_descriptors",
    "featurizations/hu_moments": "geometrify/hu_moments",
    "vectorization_geometry/medial_axis": "geometrify/media_axis",

    # graphs
    "boundary_tracing/graph_cut.cpp": "graphs/graph_cut",
    "boundary_tracing/graph_cut.hpp": "graphs/graph_cut",
    "graphs/pslg-color": "graphs/pslg",

    # sdf
    "featurizations/sdf/8SSEDT.hpp": "sdf/8ssedt",
    "featurizations/sdf/chamfer.cpp": "sdf/chamfer",
    "featurizations/sdf/chamfer.hpp": "sdf/chamfer",
    "filters/edt": "sdf/edt",

    # segmentation
    "bbox": "segmentation/bbox_auto",
    "screen_detection/CCL": "segmentation/ccl",
    "vectorization_geometry/convex_hull": "segmentation/convex_hull",
    "filters/watershed": "segmentation/watershed",

    # structures
    "spatial_trees/2d_r_tree": "structures/2d_r_tree",
    "lookup_tables": "structures/lookup_tables",
    "spatial_trees/lsh": "structures/lsh",
    "graphs/half-region-adj": "structures/region_adjacency",
    "graphs/slic": "structures/slic",
    "spatial_trees/vp_tree": "structures/vp_tree",

    # topology
    "vectorization_geometry/euler": "topology/euler",

    # templates
    "screen_detection/LOOKUP": "segmentation/template_ncc",
}

# Directories to keep as shared core modules
CORE_MODULES = ["contour_kit", "math"]

def run_reorganization(dry_run: bool = True):
    repo_root = Path.cwd()
    video_dir = repo_root / "src" / "video"
    modules_dir = video_dir / "modules"
    staging_dir = video_dir / "modules_reorganized"
    backup_dir = video_dir / "modules_backup"

    if not modules_dir.exists():
        print(f"Error: Expected modules path at '{modules_dir}' was not found.")
        print("Ensure you are running from the repository root (C:\\...\\generative-media-research).")
        sys.exit(1)

    mode_label = "DRY RUN (No files modified)" if dry_run else "EXECUTING REORGANIZATION"
    print(f"=== {mode_label} ===")
    print(f"Target directory: {modules_dir}\n")

    for src_rel, dst_rel in MODULE_MAPPING.items():
        src_path = modules_dir / src_rel
        dst_path = staging_dir / dst_rel

        if not src_path.exists():
            print(f"[SKIP / MISSING] {src_rel}")
            continue

        if dry_run:
            print(f"[STAGE] {src_path.relative_to(modules_dir)} -> {dst_rel}")
            continue

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            dst_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path / src_path.name)

    # Handle unmapped shared dependencies
    for core in CORE_MODULES:
        core_src = modules_dir / core
        if core_src.exists():
            if dry_run:
                print(f"[PRESERVE CORE] {core_src.relative_to(modules_dir)} -> {core}")
            else:
                dst_core = staging_dir / core
                dst_core.mkdir(parents=True, exist_ok=True)
                shutil.copytree(core_src, dst_core, dirs_exist_ok=True)

    if dry_run:
        print("\nVerification complete. No files were moved.")
        print("To apply changes, run the script with the `--apply` flag.")
    else:
        print("\nSwapping directories...")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(str(modules_dir), str(backup_dir))
        shutil.move(str(staging_dir), str(modules_dir))
        print(f"[DONE] Successfully restructured 'src/video/modules'.")
        print(f"[BACKUP] Original structure preserved at '{backup_dir}'.")

if __name__ == "__main__":
    apply_changes = "--apply" in sys.argv
    run_reorganization(dry_run=not apply_changes)