#!/usr/bin/env python3
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# Content templates for extracted helpers
# -------------------------------------------------------------------------

CONVEX_HULL_HELPERS = """#pragma once

#include "types.hpp"
#include <algorithm>
#include <vector>

namespace contour {

inline std::vector<Vec2> convex_hull(std::vector<Vec2> pts) {
    if (pts.size() < 2) {
        return pts;
    }
    std::sort(pts.begin(), pts.end(), [](const Vec2& a, const Vec2& b) {
        return a.x == b.x ? a.y < b.y : a.x < b.x;
    });
    auto cross = [](const Vec2& o, const Vec2& a, const Vec2& b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };
    std::vector<Vec2> lower, upper;
    for (const auto& p : pts) {
        while (lower.size() >= 2 && cross(lower[lower.size() - 2], lower.back(), p) <= 0) {
            lower.pop_back();
        }
        lower.push_back(p);
    }
    for (int i = static_cast<int>(pts.size()) - 1; i >= 0; --i) {
        const auto& p = pts[static_cast<size_t>(i)];
        while (upper.size() >= 2 && cross(upper[upper.size() - 2], upper.back(), p) <= 0) {
            upper.pop_back();
        }
        upper.push_back(p);
    }
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;
}

} // namespace contour
"""

SDF_EDT_HELPERS = """#pragma once

#include "types.hpp"

namespace contour {

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

} // namespace contour
"""

MORPH_CLEAN_HELPERS = """#pragma once

#include "types.hpp"

namespace contour {

inline ImageBuffer dilate_binary(const ImageBuffer& src, int radius) {
    ImageBuffer out = src;
    if (radius <= 0) {
        return out;
    }
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            if (src.at(x, y) == 0) {
                continue;
            }
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx >= 0 && yy >= 0 && xx < src.width && yy < src.height) {
                        out.at(xx, yy) = 255;
                    }
                }
            }
        }
    }
    return out;
}

} // namespace contour
"""

CLEANED_METRICS_HPP = """#pragma once

#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

// Forward / Helper includes from separated modules
#include "../segmentation/convex_hull/helpers.hpp"
#include "../filters/morph_clean/helpers.hpp"
#include "../sdf/edt/helpers.hpp"

namespace contour {

inline float shoelace(const std::vector<Vec2>& p) {
    if (p.size() < 3) {
        return 0.0f;
    }
    double a = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        const Vec2& u = p[i];
        const Vec2& v = p[(i + 1) % p.size()];
        a += static_cast<double>(u.x) * v.y - static_cast<double>(v.x) * u.y;
    }
    return static_cast<float>(std::fabs(a) * 0.5);
}

inline ImageBuffer rasterize_polygon(const std::vector<Vec2>& poly, int w, int h) {
    ImageBuffer mask = make_gray(w, h, 0);
    if (poly.size() < 3) {
        return mask;
    }
    for (int y = 0; y < h; ++y) {
        std::vector<float> xs;
        const float yy = static_cast<float>(y) + 0.5f;
        for (size_t i = 0; i < poly.size(); ++i) {
            const Vec2 a = poly[i];
            const Vec2 b = poly[(i + 1) % poly.size()];
            if ((a.y <= yy && b.y > yy) || (b.y <= yy && a.y > yy)) {
                const float t = (yy - a.y) / (b.y - a.y);
                xs.push_back(a.x + t * (b.x - a.x));
            }
        }
        std::sort(xs.begin(), xs.end());
        for (size_t k = 0; k + 1 < xs.size(); k += 2) {
            const int x0 = std::clamp(static_cast<int>(std::ceil(xs[k])), 0, w);
            const int x1 = std::clamp(static_cast<int>(std::floor(xs[k + 1])), 0, w - 1);
            for (int x = x0; x <= x1; ++x) {
                mask.at(x, y) = 255;
            }
        }
    }
    return mask;
}

inline ImageBuffer xor_fill(const std::vector<Polyline>& loops, int w, int h) {
    ImageBuffer pred = make_gray(w, h, 0);
    for (const auto& p : loops) {
        if (p.points.size() < 3) {
            continue;
        }
        const ImageBuffer m = rasterize_polygon(p.points, w, h);
        for (size_t i = 0; i < pred.data.size(); ++i) {
            pred.data[i] = static_cast<uint8_t>(pred.data[i] ^ m.data[i]);
        }
    }
    return pred;
}

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

inline ImageBuffer polyline_to_boundary(const std::vector<Vec2>& pts, int w, int h, bool closed) {
    ImageBuffer b = make_gray(w, h, 0);
    auto plot = [&](int x, int y) {
        if (x >= 0 && y >= 0 && x < w && y < h) {
            b.at(x, y) = 255;
        }
    };
    auto line = [&](Vec2 a, Vec2 bpt) {
        const int n = std::max(1, static_cast<int>(dist(a, bpt) * 2.0f));
        for (int i = 0; i <= n; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(n);
            plot(static_cast<int>(std::lround(a.x + (bpt.x - a.x) * t)),
                 static_cast<int>(std::lround(a.y + (bpt.y - a.y) * t)));
        }
    };
    if (pts.size() < 2) {
        return b;
    }
    for (size_t i = 1; i < pts.size(); ++i) {
        line(pts[i - 1], pts[i]);
    }
    if (closed) {
        line(pts.back(), pts.front());
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
"""

# -------------------------------------------------------------------------
# Execution Logic
# -------------------------------------------------------------------------

EXTRACT_MAP = [
    {
        "target_path": "src/video/modules/segmentation/convex_hull/helpers.hpp",
        "description": "convex_hull()",
        "content": CONVEX_HULL_HELPERS,
    },
    {
        "target_path": "src/video/modules/sdf/edt/helpers.hpp",
        "description": "rasterize_mask_from_field()",
        "content": SDF_EDT_HELPERS,
    },
    {
        "target_path": "src/video/modules/filters/morph_clean/helpers.hpp",
        "description": "dilate_binary()",
        "content": MORPH_CLEAN_HELPERS,
    },
]

def run_extraction(dry_run: bool = True):
    repo_root = Path.cwd()
    metrics_path = repo_root / "src" / "video" / "modules" / "contour_kit" / "metrics.hpp"

    mode_label = "DRY RUN" if dry_run else "EXECUTING"
    print(f"=== {mode_label}: Modular Helper Extraction ===\n")

    # 1. Write helper files to corresponding module directories
    for item in EXTRACT_MAP:
        dest = repo_root / item["target_path"]
        if dry_run:
            print(f"[EXTRACT] {item['description']} -> {dest.relative_to(repo_root)}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(item["content"], encoding="utf-8")
            print(f"[WRITTEN] {dest.relative_to(repo_root)}")

    # 2. Update metrics.hpp with clean inclusions
    if dry_run:
        print(f"\n[REFACTOR] Clean metrics header at -> {metrics_path.relative_to(repo_root)}")
        print("\nDry run completed. Run with `--apply` to execute file writes.")
    else:
        if metrics_path.exists():
            backup_path = metrics_path.with_suffix(".hpp.bak")
            metrics_path.rename(backup_path)
            print(f"\n[BACKUP] {backup_path.relative_to(repo_root)}")

        metrics_path.write_text(CLEANED_METRICS_HPP, encoding="utf-8")
        print(f"[UPDATED] {metrics_path.relative_to(repo_root)}")
        print("\nExtraction complete.")

if __name__ == "__main__":
    apply_changes = "--apply" in sys.argv
    run_extraction(dry_run=not apply_changes)