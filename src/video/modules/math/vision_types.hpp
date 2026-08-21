#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

namespace vision {

constexpr float kPi = 3.14159265358979323846f;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

inline float dist2(const Vec2& a, const Vec2& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

inline float dist(const Vec2& a, const Vec2& b) {
    return std::sqrt(dist2(a, b));
}

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

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;

    bool empty() const { return pixels.empty() || width <= 0 || height <= 0; }

    uint8_t at(int x, int y) const {
        if (x < 0 || y < 0 || x >= width || y >= height) {
            return 0;
        }
        return pixels[static_cast<size_t>(y * width + x)];
    }

    uint8_t& at(int x, int y) {
        return pixels[static_cast<size_t>(y * width + x)];
    }

    bool fg(int x, int y, uint8_t thr = 128) const {
        return at(x, y) > thr;
    }

    int count_fg(uint8_t thr = 128) const {
        int n = 0;
        for (uint8_t p : pixels) {
            if (p > thr) {
                ++n;
            }
        }
        return n;
    }
};

struct DatasetRow {
    std::string file;
    std::string label;
    std::string extra;
    std::vector<std::string> fields;
};

}  // namespace vision
