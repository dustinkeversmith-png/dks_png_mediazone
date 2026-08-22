#!/usr/bin/env python3
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# Modular Source Contents
# -------------------------------------------------------------------------

IMAGE_BOUNDARY_HPP = """#pragma once

#include "../../math/vision_types.hpp"
#include "../../filters/morph_clean/helpers.hpp"
#include <algorithm>
#include <vector>

namespace image {

inline math::ImageBuffer boundary_pixels(const math::ImageBuffer& mask, uint8_t thr = 127) {
    math::ImageBuffer b = math::make_gray(mask.width, mask.height, 0);
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

inline double boundary_f1(const math::ImageBuffer& pred_boundary, const math::ImageBuffer& gt_boundary, int tol = 2) {
    const auto gt_d = contour::dilate_binary(gt_boundary, tol);
    const auto pr_d = contour::dilate_binary(pred_boundary, tol);
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

} // namespace image
"""

IMAGE_MASKS_HPP = """#pragma once

#include "../../math/vision_types.hpp"
#include "../../math/geometry.hpp"
#include "../../segmentation/convex_hull/helpers.hpp"
#include <algorithm>
#include <vector>

namespace image {

inline double mask_iou(const math::ImageBuffer& a, const math::ImageBuffer& b, uint8_t thr = 127) {
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

inline double tightness_ratio(const math::ImageBuffer& mask, uint8_t thr = 127) {
    std::vector<math::Vec2> pts;
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
    return static_cast<double>(math::shoelace(contour::convex_hull(std::move(pts)))) / static_cast<double>(area);
}

} // namespace image
"""

IMAGE_GEOMETRY_HPP = """#pragma once

#include "../../math/vision_types.hpp"
#include <vector>

namespace image {

inline std::vector<math::Vec2> collect_on(const math::ImageBuffer& b) {
    std::vector<math::Vec2> pts;
    for (int y = 0; y < b.height; ++y) {
        for (int x = 0; x < b.width; ++x) {
            if (b.at(x, y) > 0) {
                pts.push_back({static_cast<float>(x), static_cast<float>(y)});
            }
        }
    }
    return pts;
}

} // namespace image
"""

MATH_DISTANCE_HPP = """#pragma once

#include "vision_types.hpp"
#include "geometry.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace math {

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

} // namespace math
"""

DATASET_EVALUATION_HPP = """#pragma once

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
"""

METRICS_FORWARD_HPP = """#pragma once

// Forwarding and aggregated inclusion header
#include "../math/vision_types.hpp"
#include "../math/geometry.hpp"
#include "../math/polygon.hpp"
#include "../math/distance.hpp"
#include "../image/boundary/boundary.hpp"
#include "../image/masks/masks.hpp"
#include "../image/geometry/geometry.hpp"
#include "../../datasets/evaluation/evaluation.hpp"

namespace contour {

using namespace math;
using namespace image;
using namespace datasets;

} // namespace contour
"""

# -------------------------------------------------------------------------
# Execution Script
# -------------------------------------------------------------------------

def run_refactor(dry_run: bool = True):
    repo_root = Path.cwd()
    video_dir = repo_root / "src" / "video"

    writes = [
        # image domain subfolders
        (video_dir / "modules" / "image" / "boundary" / "boundary.hpp", IMAGE_BOUNDARY_HPP, "[CREATE] modules/image/boundary/boundary.hpp"),
        (video_dir / "modules" / "image" / "masks" / "masks.hpp", IMAGE_MASKS_HPP, "[CREATE] modules/image/masks/masks.hpp"),
        (video_dir / "modules" / "image" / "geometry" / "geometry.hpp", IMAGE_GEOMETRY_HPP, "[CREATE] modules/image/geometry/geometry.hpp"),

        # math domain distance algorithms
        (video_dir / "modules" / "math" / "distance.hpp", MATH_DISTANCE_HPP, "[CREATE] modules/math/distance.hpp"),

        # datasets evaluation
        (video_dir / "datasets" / "evaluation" / "evaluation.hpp", DATASET_EVALUATION_HPP, "[CREATE] datasets/evaluation/evaluation.hpp"),

        # clean aggregator metrics.hpp
        (video_dir / "modules" / "contour_kit" / "metrics.hpp", METRICS_FORWARD_HPP, "[UPDATE] modules/contour_kit/metrics.hpp"),
    ]

    mode_label = "DRY RUN" if dry_run else "EXECUTING"
    print(f"=== {mode_label}: Modularizing Image Operations & Dataset Evaluation ===\n")

    for dest, content, msg in writes:
        if dry_run:
            print(f"{msg} -> {dest.relative_to(repo_root)}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            print(f"{msg} written.")

    if dry_run:
        print("\nRun with `--apply` to create the domain subfolders and write the files.")
    else:
        print("\nReorganization complete.")

if __name__ == "__main__":
    apply_changes = "--apply" in sys.argv
    run_refactor(dry_run=not apply_changes)