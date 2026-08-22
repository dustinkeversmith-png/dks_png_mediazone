#pragma once

#include "../math/vision_types.hpp"

#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

namespace vision {

// Moore-neighbor outer-boundary trace (8-connected).
class MooreNeighborTracer {
public:
    struct Contour {
        std::vector<Vec2> points;
        bool closed = false;
        float area = 0.0f;
        float perimeter = 0.0f;
    };

    static Contour trace(const GrayImage& image, uint8_t thr = 128) {
        Contour contour;
        int sx = -1, sy = -1;
        for (int y = 0; y < image.height && sx < 0; ++y) {
            for (int x = 0; x < image.width; ++x) {
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

        static const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        static const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

        int x = sx;
        int y = sy;
        int dir = 7;  // previous move into the start from the west
        const int max_steps = image.width * image.height * 4 + 8;

        for (int step = 0; step < max_steps; ++step) {
            contour.points.push_back({static_cast<float>(x), static_cast<float>(y)});
            int start_dir = (dir + 6) % 8;  // turn right relative to incoming
            bool found = false;
            for (int k = 0; k < 8; ++k) {
                const int nd = (start_dir + k) % 8;
                const int nx = x + dx[nd];
                const int ny = y + dy[nd];
                if (image.fg(nx, ny, thr)) {
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
            if (x == sx && y == sy && contour.points.size() > 2) {
                contour.closed = true;
                break;
            }
        }

        contour.area = shoelace(contour.points);
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

    static float shoelace(const std::vector<Vec2>& p) {
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

    static float polyline_length(const std::vector<Vec2>& p, bool closed) {
        if (p.size() < 2) {
            return 0.0f;
        }
        float len = 0.0f;
        for (size_t i = 1; i < p.size(); ++i) {
            len += dist(p[i - 1], p[i]);
        }
        if (closed) {
            len += dist(p.back(), p.front());
        }
        return len;
    }

private:
    static Vec2 lerp(const Vec2& a, const Vec2& b, float t) {
        return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
    }
};

}  // namespace vision
