#pragma once

#include "../../modules/math/vision_types.hpp"
#include "../../modules/math/polygon.hpp"
#include "../../modules/math/distance.hpp"
#include "../../modules/image/boundary/boundary.hpp"
#include "../../modules/image/masks/masks.hpp"
#include "../../modules/image/geometry/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace datasets {

inline double mean_boundary_distance(const math::ImageBuffer& pred_b, const math::ImageBuffer& gt_b) {
    return math::mean_min_distance(image::collect_on(pred_b), image::collect_on(gt_b));
}

struct Score {
    double iou = 0;
    double boundary_f1 = 0;
    double tightness = 0;
    double latency_ms = 0;
    double megapixels = 0;
    double mean_dist_px = 0;
    double hausdorff_px = 0;
};

inline Score evaluate_mask(const math::ImageBuffer& pred, const math::ImageBuffer& gt, double latency_ms, int tol = 2) {
    Score s;
    s.iou = image::mask_iou(pred, gt);
    s.boundary_f1 = image::boundary_f1(image::boundary_pixels(pred), image::boundary_pixels(gt), tol);
    s.tightness = image::tightness_ratio(gt);
    s.latency_ms = latency_ms;
    s.megapixels = static_cast<double>(gt.width * gt.height) / 1.0e6;
    return s;
}

inline Score evaluate_polyline(const math::Polyline& poly, const math::ImageBuffer& gt, double latency_ms, int tol = 2) {
    auto pred = math::rasterize_polygon(poly.points, gt.width, gt.height);
    Score s = evaluate_mask(pred, gt, latency_ms, tol);
    auto pb = math::polyline_to_boundary(poly.points, gt.width, gt.height, poly.closed);
    s.boundary_f1 = image::boundary_f1(pb, image::boundary_pixels(gt), tol);
    return s;
}

} // namespace datasets
