#!/usr/bin/env python3
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# New Modular File Contents
# -------------------------------------------------------------------------

MATH_GEOMETRY_HPP = """#pragma once

#include "vision_types.hpp"
#include <cmath>
#include <vector>

namespace math {

constexpr float kPi = 3.14159265358979323846f;

inline Vec2 operator+(const Vec2& a, const Vec2& b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(const Vec2& a, const Vec2& b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(const Vec2& a, float s) { return {a.x * s, a.y * s}; }
inline float dot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }
inline float length(const Vec2& a) { return std::sqrt(dot(a, a)); }
inline float dist2(const Vec2& a, const Vec2& b) { return dot(a - b, a - b); }
inline float dist(const Vec2& a, const Vec2& b) { return std::sqrt(dist2(a, b)); }
inline Vec2 normalize(const Vec2& a) {
    const float n = length(a);
    return n > 1e-8f ? Vec2{a.x / n, a.y / n} : Vec2{0, 0};
}

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

} // namespace math
"""

MATH_POLYGON_HPP = """#pragma once

#include "vision_types.hpp"
#include "geometry.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace math {

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

} // namespace math
"""

CONTOUR_TYPES_HPP = """#pragma once

#include "../math/vision_types.hpp"
#include "../math/geometry.hpp"

namespace contour {

using Vec2 = math::Vec2;
using Rect = math::Rect;
using ImageBuffer = math::ImageBuffer;
using Field = math::Field;
using Polyline = math::Polyline;

using math::make_gray;
using math::make_field;
using math::dist;
using math::dist2;
using math::length;
using math::dot;
using math::normalize;

} // namespace contour
"""

CONTOUR_METRICS_HPP = """#pragma once

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
"""

VISION_TYPES_HPP = """#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

namespace math {

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Polyline {
    std::vector<Vec2> points;
    bool closed = false;
};

struct Rect {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;

    float x1() const { return x + w; }
    float y1() const { return y + h; }
    float area() const { return std::max(0.0f, w) * std::max(0.0f, h); }

    bool intersects(const Rect& o) const {
        return x < o.x1() && x1() > o.x && y < o.y1() && y1() > o.y;
    }

    bool contains(const Rect& o) const {
        return o.x >= x && o.y >= y && o.x1() <= x1() && o.y1() <= y1();
    }

    float iou(const Rect& o) const {
        const float ix = std::max(x, o.x);
        const float iy = std::max(y, o.y);
        const float ix1 = std::min(x1(), o.x1());
        const float iy1 = std::min(y1(), o.y1());
        const float iw = std::max(0.0f, ix1 - ix);
        const float ih = std::max(0.0f, iy1 - iy);
        const float inter = iw * ih;
        const float uni = area() + o.area() - inter;
        return uni > 0.0f ? inter / uni : 0.0f;
    }
};

struct ImageBuffer {
    int width = 0;
    int height = 0;
    int channels = 1;
    std::vector<uint8_t> data;

    bool empty() const { return data.empty() || width <= 0 || height <= 0; }
    size_t index(int x, int y, int c = 0) const {
        return static_cast<size_t>((y * width + x) * channels + c);
    }
    uint8_t at(int x, int y, int c = 0) const {
        if (x < 0 || y < 0 || x >= width || y >= height) {
            return 0;
        }
        return data[index(x, y, c)];
    }
    uint8_t& at(int x, int y, int c = 0) { return data[index(x, y, c)]; }

    float gray(int x, int y) const {
        if (channels == 1) {
            return static_cast<float>(at(x, y, 0));
        }
        const float r = at(x, y, 0);
        const float g = channels > 1 ? at(x, y, 1) : r;
        const float b = channels > 2 ? at(x, y, 2) : r;
        return 0.299f * r + 0.587f * g + 0.114f * b;
    }

    bool fg(int x, int y, uint8_t thr = 128) const {
        return at(x, y, 0) > thr;
    }
};

// Aliased GrayImage for backwards compatibility
using GrayImage = ImageBuffer;

struct Field {
    int width = 0;
    int height = 0;
    std::vector<float> data;
    float at(int x, int y) const {
        if (x < 0 || y < 0 || x >= width || y >= height) {
            return data.empty() ? 0.0f : data.front();
        }
        return data[static_cast<size_t>(y * width + x)];
    }
    float& at(int x, int y) { return data[static_cast<size_t>(y * width + x)]; }
    float sample(float x, float y) const {
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = std::min(width - 1, x0 + 1);
        const int y1 = std::min(height - 1, y0 + 1);
        const float tx = x - static_cast<float>(x0);
        const float ty = y - static_cast<float>(y0);
        const float a = at(std::clamp(x0, 0, width - 1), std::clamp(y0, 0, height - 1));
        const float b = at(x1, std::clamp(y0, 0, height - 1));
        const float c = at(std::clamp(x0, 0, width - 1), y1);
        const float d = at(x1, y1);
        return a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty;
    }
};

inline ImageBuffer make_gray(int w, int h, uint8_t fill = 0) {
    ImageBuffer im;
    im.width = w;
    im.height = h;
    im.channels = 1;
    im.data.assign(static_cast<size_t>(w * h), fill);
    return im;
}

inline Field make_field(int w, int h, float fill = 0.0f) {
    Field f;
    f.width = w;
    f.height = h;
    f.data.assign(static_cast<size_t>(w * h), fill);
    return f;
}

struct DatasetRow {
    std::string file;
    std::string label;
    std::string extra;
    std::vector<std::string> fields;
};

} // namespace math

namespace vision {
using namespace math;
}
"""

def run_reorganization(dry_run: bool = True):
    repo_root = Path.cwd()
    video_modules_dir = repo_root / "src" / "video" / "modules"
    math_dir = video_modules_dir / "math"
    contour_kit_dir = video_modules_dir / "contour_kit"
    edt_dir = video_modules_dir / "sdf" / "edt"

    mode_label = "DRY RUN" if dry_run else "EXECUTING"
    print(f"=== {mode_label}: Math & Geometry Restructuring ===\n")

    # 1. Move distance_transform.hpp to modules/sdf/edt/
    src_dt = math_dir / "distance_transform.hpp"
    dst_dt = edt_dir / "distance_transform.hpp"
    if src_dt.exists():
        if dry_run:
            print(f"[MOVE] {src_dt.relative_to(repo_root)} -> {dst_dt.relative_to(repo_root)}")
        else:
            edt_dir.mkdir(parents=True, exist_ok=True)
            src_dt.rename(dst_dt)
            print(f"[MOVED] {src_dt.relative_to(repo_root)} -> {dst_dt.relative_to(repo_root)}")

    # 2. Write domain-split headers to math/
    writes = [
        (math_dir / "vision_types.hpp", VISION_TYPES_HPP, "[UPDATE] math/vision_types.hpp"),
        (math_dir / "geometry.hpp", MATH_GEOMETRY_HPP, "[CREATE] math/geometry.hpp"),
        (math_dir / "polygon.hpp", MATH_POLYGON_HPP, "[CREATE] math/polygon.hpp"),
        (contour_kit_dir / "types.hpp", CONTOUR_TYPES_HPP, "[UPDATE] contour_kit/types.hpp"),
        (contour_kit_dir / "metrics.hpp", CONTOUR_METRICS_HPP, "[UPDATE] contour_kit/metrics.hpp"),
    ]

    for dest, content, log_msg in writes:
        if dry_run:
            print(f"{log_msg}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            print(f"{log_msg} written.")

    if dry_run:
        print("\nDry run completed. Run with `--apply` to apply the changes.")
    else:
        print("\nRestructuring completed successfully.")

if __name__ == "__main__":
    apply_changes = "--apply" in sys.argv
    run_reorganization(dry_run=not apply_changes)