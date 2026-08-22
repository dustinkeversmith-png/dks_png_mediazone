#pragma once

#include "types.hpp"
#include "../math/polygon.hpp"
#include "../segmentation/convex_hull/helpers.hpp"
#include "../filters/morph_clean/helpers.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace contour {

using math::shoelace;
using math::rasterize_polygon;
using math::polyline_to_boundary;
using math::xor_fill;

inline double mask_iou(const ImageBuffer& a, const ImageBuffer& b, uint8_t thr = 127) {
    const int w = std::min(a.width, b.width);
    const int h = std::min(a.height, b.height);
    int inter = 0, uni = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const bool pa = a.at(x, y) > thr;
            const bool pb = b.at(x, y) > thr;
            inter += (pa && pb);
            uni += (pa || pb);
        }
    }
    return uni > 0 ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0;
}

inline ImageBuffer boundary_pixels(const ImageBuffer& mask, uint8_t thr = 127) {
    ImageBuffer b = make_gray(mask.width, mask.height, 0);
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            if (mask.at(x, y) <= thr) {
                continue;
            }
            bool edge = x == 0 || y == 0 || x == mask.width - 1 || y == mask.height - 1;
            if (!edge) {
                for (int k = 0; k < 4 && !edge; ++k) {
                    static const int dx[4] = {1, -1, 0, 0};
                    static const int dy[4] = {0, 0, 1, -1};
                    if (mask.at(x + dx[k], y + dy[k]) <= thr) {
                        edge = true;
                    }
                }
            }
            if (edge) {
                b.at(x, y) = 255;
            }
        }
    }
    return b;
}

inline double boundary_f1(const ImageBuffer& pred_boundary, const ImageBuffer& gt_boundary, int tol = 2) {
    const auto gt_d = dilate_binary(gt_boundary, tol);
    const auto pr_d = dilate_binary(pred_boundary, tol);
    int pred_n = 0, gt_n = 0, hit_p = 0, hit_g = 0;
    const int w = std::min(pred_boundary.width, gt_boundary.width);
    const int h = std::min(pred_boundary.height, gt_boundary.height);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (pred_boundary.at(x, y) > 0) {
                ++pred_n;
                if (gt_d.at(x, y) > 0) {
                    ++hit_p;
                }
            }
            if (gt_boundary.at(x, y) > 0) {
                ++gt_n;
                if (pr_d.at(x, y) > 0) {
                    ++hit_g;
                }
            }
        }
    }
    const double prec = pred_n > 0 ? static_cast<double>(hit_p) / pred_n : 0.0;
    const double rec = gt_n > 0 ? static_cast<double>(hit_g) / gt_n : 0.0;
    return (prec + rec) > 1e-12 ? 2.0 * prec * rec / (prec + rec) : 0.0;
}

inline double tightness_ratio(const ImageBuffer& mask, uint8_t thr = 127) {
    std::vector<Vec2> pts;
    int area = 0;
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            if (mask.at(x, y) > thr) {
                ++area;
                pts.push_back({static_cast<float>(x), static_cast<float>(y)});
            }
        }
    }
    if (area == 0) {
        return 0.0;
    }
    return static_cast<double>(shoelace(convex_hull(std::move(pts)))) / static_cast<double>(area);
}

inline std::vector<Vec2> collect_on(const ImageBuffer& b) {
    std::vector<Vec2> pts;
    for (int y = 0; y < b.height; ++y) {
        for (int x = 0; x < b.width; ++x) {
            if (b.at(x, y) > 0) {
                pts.push_back({static_cast<float>(x), static_cast<float>(y)});
            }
        }
    }
    return pts;
}

inline std::vector<Vec2> thin_points(std::vector<Vec2> pts, size_t cap = 400) {
    if (pts.size() <= cap) {
        return pts;
    }
    std::vector<Vec2> out;
    out.reserve(cap);
    const float step = static_cast<float>(pts.size()) / static_cast<float>(cap);
    for (size_t i = 0; i < cap; ++i) {
        out.push_back(pts[static_cast<size_t>(i * step)]);
    }
    return out;
}

inline double mean_min_distance(const std::vector<Vec2>& a, const std::vector<Vec2>& b) {
    const auto aa = thin_points(a);
    const auto bb = thin_points(b);
    if (aa.empty() || bb.empty()) {
        return 0.0;
    }
    double acc = 0;
    for (const auto& p : aa) {
        float best = 1e9f;
        for (const auto& q : bb) {
            best = std::min(best, dist(p, q));
        }
        acc += best;
    }
    return acc / static_cast<double>(aa.size());
}

inline double hausdorff(const std::vector<Vec2>& a, const std::vector<Vec2>& b) {
    const auto aa = thin_points(a);
    const auto bb = thin_points(b);
    if (aa.empty() || bb.empty()) {
        return 0.0;
    }
    auto directed = [](const std::vector<Vec2>& u, const std::vector<Vec2>& v) {
        float h = 0;
        for (const auto& p : u) {
            float best = 1e9f;
            for (const auto& q : v) {
                best = std::min(best, dist(p, q));
            }
            h = std::max(h, best);
        }
        return h;
    };
    return std::max(directed(aa, bb), directed(bb, aa));
}

inline double mean_boundary_distance(const ImageBuffer& pred_b, const ImageBuffer& gt_b) {
    return mean_min_distance(collect_on(pred_b), collect_on(gt_b));
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

inline Score evaluate_mask(const ImageBuffer& pred, const ImageBuffer& gt, double latency_ms, int tol = 2) {
    Score s;
    s.iou = mask_iou(pred, gt);
    s.boundary_f1 = boundary_f1(boundary_pixels(pred), boundary_pixels(gt), tol);
    s.tightness = tightness_ratio(gt);
    s.latency_ms = latency_ms;
    s.megapixels = static_cast<double>(gt.width * gt.height) / 1.0e6;
    return s;
}

inline Score evaluate_polyline(const Polyline& poly, const ImageBuffer& gt, double latency_ms, int tol = 2) {
    auto pred = rasterize_polygon(poly.points, gt.width, gt.height);
    Score s = evaluate_mask(pred, gt, latency_ms, tol);
    auto pb = polyline_to_boundary(poly.points, gt.width, gt.height, poly.closed);
    s.boundary_f1 = boundary_f1(pb, boundary_pixels(gt), tol);
    return s;
}

} // namespace contour
