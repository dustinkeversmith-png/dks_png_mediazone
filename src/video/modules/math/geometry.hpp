#pragma once

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
