#!/usr/bin/env python3
"""Bulk-fix includes and .pixels -> .data for vision refactor."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VIDEO = REPO / "src" / "video"

SKIP_DIRS = {"_archive", "modules_reorganized", "contour_kit"}

INCLUDE_REPLACEMENTS = [
    ('#include "contour_kit/types.hpp"', '#include "math/contour_compat.hpp"'),
    ('#include "../contour_kit/types.hpp"', '#include "math/contour_compat.hpp"'),
    ('#include "../../contour_kit/types.hpp"', '#include "math/contour_compat.hpp"'),
    ('#include "contour_kit/metrics.hpp"', '#include "math/contour_metrics.hpp"'),
    ('#include "../contour_kit/metrics.hpp"', '#include "math/contour_metrics.hpp"'),
    ('#include "types.hpp"', '#include "math/contour_compat.hpp"'),
    ('#include "../types.hpp"', '#include "math/contour_compat.hpp"'),
    ('#include "filters/edt/edt.hpp"', '#include "sdf/edt/edt.hpp"'),
    ('#include "filters/sobel.hpp"', '#include "filters/sobel/sobel.hpp"'),
    ('#include "../filters/sobel.hpp"', '#include "filters/sobel/sobel.hpp"'),
    ('#include "../bbox/bbox_auto.hpp"', '#include "segmentation/bbox_auto/bbox_auto.hpp"'),
    ('#include "bbox/bbox_auto.hpp"', '#include "segmentation/bbox_auto/bbox_auto.hpp"'),
    ('#include "../filters/bilateral.hpp"', '#include "filters/bilateral/bilateral.hpp"'),
    ('#include "filters/bilateral.hpp"', '#include "filters/bilateral/bilateral.hpp"'),
    ('#include "../featurizations/edge/canny.hpp"', '#include "filters/edge/canny/canny.hpp"'),
    ('#include "featurizations/edge/canny.hpp"', '#include "filters/edge/canny/canny.hpp"'),
    ('#include "../featurizations/gvh/gvh.hpp"', '#include "filters/gvf/gvh.hpp"'),
    ('#include "featurizations/gvh/gvh.hpp"', '#include "filters/gvf/gvh.hpp"'),
    ('#include "../color/lab_color_space.hpp"', '#include "filters/lab_color/lab_color_space.hpp"'),
    ('#include "color/lab_color_space.hpp"', '#include "filters/lab_color/lab_color_space.hpp"'),
    ('#include "featurizations/sdf/8SSEDT.hpp"', '#include "sdf/8ssedt/8SSEDT.hpp"'),
    ('#include "featurizations/sdf/chamfer.hpp"', '#include "sdf/chamfer/chamfer.hpp"'),
    ('#include "filters/edt/edt.hpp"', '#include "sdf/edt/edt.hpp"'),
    ('#include "filters/watershed/watershed.hpp"', '#include "segmentation/watershed/watershed.hpp"'),
    ('#include "boundary_tracing/dual_contouring.hpp"', '#include "contour/dual_contouring/dual_contouring.hpp"'),
    ('#include "boundary_tracing/marching_squares.hpp"', '#include "contour/marching_squares/marching_squares.hpp"'),
    ('#include "boundary_tracing/moore_neighbor.hpp"', '#include "contour/moore_neighborhood/moore_neighbor.hpp"'),
    ('#include "boundary_tracing/live_wire.hpp"', '#include "contour/live_wire/live_wire.hpp"'),
    ('#include "boundary_tracing/snakes.hpp"', '#include "contour/snakes/snakes.hpp"'),
    ('#include "boundary_tracing/level_set.hpp"', '#include "contour/level_set/level_set.hpp"'),
    ('#include "boundary_tracing/graph_cut.hpp"', '#include "graphs/graph_cut/graph_cut.hpp"'),
    ('#include "featurizations/laplace_gaussian/laplace_gaussian.hpp"', '#include "contour/laplace_gaussian/laplace_gaussian.hpp"'),
    ('#include "vectorization_geometry/rdp/rdp.hpp"', '#include "contour/rdp/rdp.hpp"'),
    ('#include "vectorization_geometry/convex_hull/convex_hull.hpp"', '#include "segmentation/convex_hull/convex_hull.hpp"'),
    ('#include "vectorization_geometry/euler/euler_characteristic.hpp"', '#include "topology/euler/euler_characteristic.hpp"'),
    ('#include "vectorization_geometry/medial_axis/medial_axis.hpp"', '#include "geometrify/media_axis/medial_axis.hpp"'),
    ('#include "vectorization_geometry/measurements/turning_function/turning_function.hpp"', '#include "features/turning_function/turning_function.hpp"'),
    ('#include "screen_detection/CCL/connected_components.hpp"', '#include "segmentation/ccl/connected_components.hpp"'),
    ('#include "screen_detection/LOOKUP/template_ncc.hpp"', '#include "segmentation/template_ncc/template_ncc.hpp"'),
    ('#include "../../screen_detection/CCL/connected_components.hpp"', '#include "../../segmentation/ccl/connected_components.hpp"'),
    ('#include "spatial_trees/2d_r_tree/r_tree_2d.hpp"', '#include "structures/2d_r_tree/r_tree_2d.hpp"'),
    ('#include "spatial_trees/vp_tree/vp_tree.hpp"', '#include "structures/vp_tree/vp_tree.hpp"'),
    ('#include "spatial_trees/lsh/fourier_lsh.hpp"', '#include "structures/lsh/fourier_lsh.hpp"'),
    ('#include "lookup_tables/geometry_lut.hpp"', '#include "structures/lookup_tables/geometry_lut.hpp"'),
    ('#include "graphs/half-region-adj/region_adjacency.hpp"', '#include "structures/region_adjacency/region_adjacency.hpp"'),
    ('#include "graphs/slic/slic.hpp"', '#include "structures/slic/slic.hpp"'),
    ('#include "graphs/pslg-color/pslg_color.hpp"', '#include "graphs/pslg/pslg_color.hpp"'),
    ('#include "featurizations/fourier/fourier_descriptors.hpp"', '#include "geometrify/fourier_descriptors/fourier_descriptors.hpp"'),
    ('#include "featurizations/hu_moments/hu_moments.hpp"', '#include "geometrify/hu_moments/hu_moments.hpp"'),
    ('#include "featurizations/rbf/thin_plate_spline.hpp"', '#include "features/rbf/thin_plate_spline.hpp"'),
    ('#include "filters/morph_clean/morph_clean.hpp"', '#include "filters/morph_clean/morph_clean.hpp"'),
    ('#include "filters/hue_gradient/hue_gradient.hpp"', '#include "filters/hue_gradient/hue_gradient.hpp"'),
    ('#include "filters/crease_sink/crease_sink.hpp"', '#include "features/crease_sink/crease_sink.hpp"'),
    ('#include "math/vision_io.hpp"', '#include "datasets/io/vision_io.hpp"'),
]


def should_process(path: Path) -> bool:
    parts = set(path.parts)
    return not parts.intersection(SKIP_DIRS)


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    for old, new in INCLUDE_REPLACEMENTS:
        text = text.replace(old, new)
    # .pixels -> .data for ImageBuffer/GrayImage member access
    text = re.sub(r"\b([a-zA-Z_][\w]*)\.pixels\b", r"\1.data", text)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = 0
    for ext in ("*.hpp", "*.cpp"):
        for path in VIDEO.rglob(ext):
            if not should_process(path):
                continue
            if fix_file(path):
                print(f"fixed: {path.relative_to(REPO)}")
                changed += 1
    print(f"\nUpdated {changed} files.")


if __name__ == "__main__":
    main()
