#pragma once

// Compatibility shim for legacy contour_kit/metrics.hpp includes.
// Prefer datasets/evaluation/evaluation.hpp and image/* modules directly.

#include "contour_compat.hpp"
#include "polygon.hpp"
#include "../../datasets/evaluation/evaluation.hpp"
#include "../image/boundary/boundary.hpp"
#include "../image/geometry/geometry.hpp"
#include "../image/masks/masks.hpp"
#include "../filters/morph_clean/helpers.hpp"

namespace contour {

using math::rasterize_polygon;
using math::polyline_to_boundary;
using math::xor_fill;

inline ImageBuffer rasterize_mask_from_field(const Field& sdf, float iso = 0.0f) {
    ImageBuffer m = make_gray(sdf.width, sdf.height, 0);
    for (int y = 0; y < sdf.height; ++y) {
        for (int x = 0; x < sdf.width; ++x) {
            if (sdf.at(x, y) <= iso) {
                m.at(x, y) = 255;
            }
        }
    }
    return m;
}

inline double mask_iou(const ImageBuffer& a, const ImageBuffer& b, uint8_t thr = 127) {
    return image::mask_iou(a, b, thr);
}

inline ImageBuffer boundary_pixels(const ImageBuffer& mask, uint8_t thr = 127) {
    return image::boundary_pixels(mask, thr);
}

inline double boundary_f1(const ImageBuffer& pred_boundary, const ImageBuffer& gt_boundary, int tol = 2) {
    return image::boundary_f1(pred_boundary, gt_boundary, tol);
}

inline double tightness_ratio(const ImageBuffer& mask, uint8_t thr = 127) {
    return image::tightness_ratio(mask, thr);
}

inline std::vector<Vec2> collect_on(const ImageBuffer& b) {
    return image::collect_on(b);
}

inline std::vector<Vec2> thin_points(std::vector<Vec2> pts, size_t cap = 400) {
    return math::thin_points(std::move(pts), cap);
}

inline double mean_min_distance(const std::vector<Vec2>& a, const std::vector<Vec2>& b) {
    return math::mean_min_distance(a, b);
}

inline double hausdorff(const std::vector<Vec2>& a, const std::vector<Vec2>& b) {
    return math::hausdorff(a, b);
}

inline double mean_boundary_distance(const ImageBuffer& pred_b, const ImageBuffer& gt_b) {
    return datasets::mean_boundary_distance(pred_b, gt_b);
}

using datasets::Score;
using datasets::evaluate_mask;
using datasets::evaluate_polyline;

}  // namespace contour
