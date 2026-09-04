#pragma once

#include "../math/vision_types.hpp"
#include "../math/geometry.hpp"
#include "../../segmentation/helpers/ccl/connected_components.hpp"

#include <vector>
#include <cmath>
#include <algorithm>

namespace vision {

inline Vec2 lerp(const Vec2& a, const Vec2& b, float t) {
    return a + (b - a) * t;
}

inline float polyline_length(const std::vector<Vec2>& pts, bool closed) {
    if (pts.size() < 2) {
        return 0.0f;
    }
    float len = 0.0f;
    for (size_t i = 1; i < pts.size(); ++i) {
        len += dist(pts[i - 1], pts[i]);
    }
    if (closed) {
        len += dist(pts.back(), pts.front());
    }
    return len;
}

// Moore-neighbor outer-boundary trace (8-connected) on the largest FG component.
class MooreNeighborTracer {
public:
    struct Contour {
        std::vector<Vec2> points;
        bool closed = false;
        float area = 0.0f;
        float perimeter = 0.0f;
    };

    static GrayImage pad_bg(const GrayImage& image, int pad = 1) {
        GrayImage out;
        out.width = image.width + 2 * pad;
        out.height = image.height + 2 * pad;
        out.data.assign(static_cast<size_t>(out.width * out.height), 0);
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                out.at(x + pad, y + pad) = image.at(x, y);
            }
        }
        return out;
    }

    static GrayImage keep_largest_component(const GrayImage& image, uint8_t thr = 128) {
        auto ccl = ConnectedComponentLabeler::label(image, thr);
        if (ccl.components.empty()) {
            return image;
        }
        int best = ccl.components.front().label;
        int best_area = ccl.components.front().area;
        for (const auto& c : ccl.components) {
            if (c.area > best_area) {
                best_area = c.area;
                best = c.label;
            }
        }
        GrayImage out;
        out.width = image.width;
        out.height = image.height;
        out.data.assign(static_cast<size_t>(image.width * image.height), 0);
        for (size_t i = 0; i < ccl.labels.size() && i < out.data.size(); ++i) {
            if (ccl.labels[i] == best) {
                out.data[i] = 255;
            }
        }
        return out;
    }

    static Contour trace(const GrayImage& image_in, uint8_t thr = 128) {
        Contour contour;
        const GrayImage cleaned = keep_largest_component(image_in, thr);
        const GrayImage image = pad_bg(cleaned, 1);
        constexpr int pad = 1;

        // Start = leftmost FG pixel (top-most on ties).
        int sx = -1, sy = -1;
        for (int x = 0; x < image.width && sx < 0; ++x) {
            for (int y = 0; y < image.height; ++y) {
                if (image.fg(x, y, thr)) {
                    sx = x;
                    sy = y;
                    break;
                }
            }
        }
        if (sx < 0) {
            return contour;
        }

        // Clockwise from E: E, SE, S, SW, W, NW, N, NE
        static const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        static const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

        int x = sx;
        int y = sy;
        int dir = 7;  // arrived as if from west looking east; first search from N (Jacob)

        // Jacob: record first two boundary pixels (p0, p1).
        int p0x = sx, p0y = sy;
        int p1x = -1, p1y = -1;
        const int max_steps = image.width * image.height * 2 + 16;

        for (int step = 0; step < max_steps; ++step) {
            contour.points.push_back({static_cast<float>(x - pad), static_cast<float>(y - pad)});

            // Search from direction rotated 90° left of arrival (i.e. +6 mod 8).
            const int start_dir = (dir + 6) % 8;
            bool found = false;
            for (int k = 0; k < 8; ++k) {
                const int nd = (start_dir + k) % 8;
                const int nx = x + dx[nd];
                const int ny = y + dy[nd];
                if (nx >= 0 && ny >= 0 && nx < image.width && ny < image.height &&
                    image.fg(nx, ny, thr)) {
                    x = nx;
                    y = ny;
                    dir = nd;
                    found = true;
                    break;
                }
            }
            if (!found) {
                break;
            }

            if (step == 0) {
                p1x = x;
                p1y = y;
            } else if (x == p0x && y == p0y && contour.points.size() > 3) {
                // Next neighbor check for full Jacob: also require next == p1.
                // Peek next without committing.
                const int peek_start = (dir + 6) % 8;
                int nx = x, ny = y, ndir = dir;
                bool peek_ok = false;
                for (int k = 0; k < 8; ++k) {
                    const int nd = (peek_start + k) % 8;
                    const int qx = x + dx[nd];
                    const int qy = y + dy[nd];
                    if (qx >= 0 && qy >= 0 && qx < image.width && qy < image.height &&
                        image.fg(qx, qy, thr)) {
                        nx = qx;
                        ny = qy;
                        ndir = nd;
                        peek_ok = true;
                        break;
                    }
                }
                if (peek_ok && nx == p1x && ny == p1y) {
                    contour.closed = true;
                    break;
                }
                // Continue walking if Jacob not satisfied yet.
                (void)ndir;
            }
        }

        if (!contour.points.empty()) {
            std::vector<Vec2> uniq;
            uniq.reserve(contour.points.size());
            uniq.push_back(contour.points.front());
            for (size_t i = 1; i < contour.points.size(); ++i) {
                if (dist2(contour.points[i], uniq.back()) > 1e-6f) {
                    uniq.push_back(contour.points[i]);
                }
            }
            // Drop duplicate close of start.
            if (uniq.size() > 2 && dist2(uniq.front(), uniq.back()) < 1e-6f) {
                uniq.pop_back();
                contour.closed = true;
            }
            contour.points.swap(uniq);
        }

        contour.area = std::fabs(shoelace(contour.points));
        contour.perimeter = polyline_length(contour.points, contour.closed);
        return contour;
    }

    static std::vector<Vec2> resample(const std::vector<Vec2>& src, int n) {
        std::vector<Vec2> out;
        if (src.size() < 2 || n <= 0) {
            return out;
        }
        std::vector<float> acc(src.size(), 0.0f);
        for (size_t i = 1; i < src.size(); ++i) {
            acc[i] = acc[i - 1] + dist(src[i - 1], src[i]);
        }
        const float total = acc.back() + dist(src.back(), src.front());
        if (total <= 1e-6f) {
            return out;
        }
        out.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const float target = (static_cast<float>(i) / static_cast<float>(n)) * total;
            size_t j = 1;
            while (j < acc.size() && acc[j] < target) {
                ++j;
            }
            if (j >= src.size()) {
                const float u = (target - acc.back()) / std::max(1e-6f, total - acc.back());
                out[static_cast<size_t>(i)] = lerp(src.back(), src.front(), u);
            } else {
                const float seg = acc[j] - acc[j - 1];
                const float u = seg > 1e-6f ? (target - acc[j - 1]) / seg : 0.0f;
                out[static_cast<size_t>(i)] = lerp(src[j - 1], src[j], u);
            }
        }
        return out;
    }
};

}  // namespace vision
