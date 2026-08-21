#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

namespace contour {

constexpr float kPi = 3.14159265358979323846f;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

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
};

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

struct Polyline {
    std::vector<Vec2> points;
    bool closed = false;
};

struct Rect {
    float x = 0, y = 0, w = 0, h = 0;
    float x1() const { return x + w; }
    float y1() const { return y + h; }
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

}  // namespace contour
