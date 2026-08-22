#pragma once

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
