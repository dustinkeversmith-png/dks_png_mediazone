#pragma once

// Similarity-invariant shape signatures for closed 2D contours.
//
// The signature is three concatenated blocks, each invariant to translation,
// rotation, uniform scale and choice of contour starting point:
//
//   1. Elliptic Fourier Descriptors (Kuhl & Giardina 1982) -- the perimeter as
//      a frequency spectrum. Low harmonics carry the macro-shape, higher ones
//      the detail, so truncating the series is a principled low-pass filter on
//      shape rather than an arbitrary feature choice.
//   2. Hu moment invariants -- seven algebraic combinations of normalized
//      central moments, computed over the filled region.
//   3. Interpretable geometry -- circularity, convexity, elongation and
//      friends. These do not add discriminative power the first two blocks
//      lack, but they are what lets a matched glyph be *named*.
//
// Blocks are scaled by fixed weights when flattened so that plain L2 in the
// flattened space is the intended metric; downstream k-means and VP-tree code
// then needs no notion of block structure.

#include "math/contour_compat.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace descriptors {

using math::Vec2;

// ---------------------------------------------------------------------------
// Elliptic Fourier Descriptors
// ---------------------------------------------------------------------------

struct FourierHarmonic {
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float d = 0.0f;
};

class EllipticFourier {
public:
    static constexpr int kHarmonics = 8;

    // Raw coefficients for harmonics 1..kHarmonics. The DC term is deliberately
    // never computed: dropping it is what makes the series translation-free.
    static std::array<FourierHarmonic, kHarmonics> encode(const std::vector<Vec2>& polygon) {
        std::array<FourierHarmonic, kHarmonics> harmonics{};
        const int n = static_cast<int>(polygon.size());
        if (n < 3) {
            return harmonics;
        }

        std::vector<float> dx(static_cast<size_t>(n));
        std::vector<float> dy(static_cast<size_t>(n));
        std::vector<float> dt(static_cast<size_t>(n));
        std::vector<float> t(static_cast<size_t>(n) + 1, 0.0f);
        for (int i = 0; i < n; ++i) {
            const Vec2& p = polygon[static_cast<size_t>(i)];
            const Vec2& q = polygon[static_cast<size_t>((i + 1) % n)];
            dx[static_cast<size_t>(i)] = q.x - p.x;
            dy[static_cast<size_t>(i)] = q.y - p.y;
            dt[static_cast<size_t>(i)] = std::sqrt(dx[static_cast<size_t>(i)] * dx[static_cast<size_t>(i)] +
                                                   dy[static_cast<size_t>(i)] * dy[static_cast<size_t>(i)]);
            t[static_cast<size_t>(i) + 1] = t[static_cast<size_t>(i)] + dt[static_cast<size_t>(i)];
        }
        const float perimeter = t[static_cast<size_t>(n)];
        if (perimeter <= 1e-6f) {
            return harmonics;
        }

        for (int h = 1; h <= kHarmonics; ++h) {
            const float scale = perimeter / (2.0f * static_cast<float>(h * h) * math::kPi * math::kPi);
            const float omega = 2.0f * static_cast<float>(h) * math::kPi / perimeter;
            float a = 0.0f;
            float b = 0.0f;
            float c = 0.0f;
            float d = 0.0f;
            for (int i = 0; i < n; ++i) {
                const size_t s = static_cast<size_t>(i);
                if (dt[s] <= 1e-9f) {
                    continue;
                }
                const float cos_hi = std::cos(omega * t[s + 1]);
                const float cos_lo = std::cos(omega * t[s]);
                const float sin_hi = std::sin(omega * t[s + 1]);
                const float sin_lo = std::sin(omega * t[s]);
                a += (dx[s] / dt[s]) * (cos_hi - cos_lo);
                b += (dx[s] / dt[s]) * (sin_hi - sin_lo);
                c += (dy[s] / dt[s]) * (cos_hi - cos_lo);
                d += (dy[s] / dt[s]) * (sin_hi - sin_lo);
            }
            harmonics[static_cast<size_t>(h - 1)] = {a * scale, b * scale, c * scale, d * scale};
        }
        return harmonics;
    }

    // Kuhl & Giardina normalization. Rotates the first harmonic's ellipse onto
    // the +x axis and divides through by its semi-major length, which removes
    // starting-point phase, orientation and scale in one step. Everything
    // downstream compares shapes, not poses.
    static std::array<FourierHarmonic, kHarmonics> normalize(
        std::array<FourierHarmonic, kHarmonics> harmonics) {
        const FourierHarmonic& first = harmonics[0];
        const float numerator = 2.0f * (first.a * first.b + first.c * first.d);
        const float denominator =
            first.a * first.a + first.c * first.c - first.b * first.b - first.d * first.d;
        if (std::fabs(numerator) < 1e-12f && std::fabs(denominator) < 1e-12f) {
            return harmonics;
        }
        // Starting-point phase of the fundamental ellipse.
        float theta = 0.5f * std::atan2(numerator, denominator);

        auto rotate_phase = [&harmonics](float phase) {
            std::array<FourierHarmonic, kHarmonics> out{};
            for (int h = 1; h <= kHarmonics; ++h) {
                const FourierHarmonic& in = harmonics[static_cast<size_t>(h - 1)];
                const float cs = std::cos(static_cast<float>(h) * phase);
                const float sn = std::sin(static_cast<float>(h) * phase);
                out[static_cast<size_t>(h - 1)] = {in.a * cs + in.b * sn, -in.a * sn + in.b * cs,
                                                   in.c * cs + in.d * sn, -in.c * sn + in.d * cs};
            }
            return out;
        };

        // Two phases a quarter turn apart both align the ellipse; pick the one
        // that puts the semi-major axis first, otherwise near-square shapes
        // flip between the two conventions and never match each other.
        std::array<FourierHarmonic, kHarmonics> rotated = rotate_phase(theta);
        const float major = rotated[0].a * rotated[0].a + rotated[0].c * rotated[0].c;
        const float minor = rotated[0].b * rotated[0].b + rotated[0].d * rotated[0].d;
        if (minor > major) {
            theta += 0.5f * math::kPi;
            rotated = rotate_phase(theta);
        }

        // Orientation of that semi-major axis in the image plane.
        const float psi = std::atan2(rotated[0].c, rotated[0].a);
        const float cs = std::cos(psi);
        const float sn = std::sin(psi);
        for (FourierHarmonic& harmonic : rotated) {
            const FourierHarmonic in = harmonic;
            harmonic.a = cs * in.a + sn * in.c;
            harmonic.b = cs * in.b + sn * in.d;
            harmonic.c = -sn * in.a + cs * in.c;
            harmonic.d = -sn * in.b + cs * in.d;
        }

        const float magnitude = std::sqrt(rotated[0].a * rotated[0].a + rotated[0].c * rotated[0].c);
        if (magnitude > 1e-9f) {
            const float inv = 1.0f / magnitude;
            for (FourierHarmonic& harmonic : rotated) {
                harmonic.a *= inv;
                harmonic.b *= inv;
                harmonic.c *= inv;
                harmonic.d *= inv;
            }
        }
        return rotated;
    }
};

// ---------------------------------------------------------------------------
// Hu moment invariants
// ---------------------------------------------------------------------------

struct RegionMoments {
    double area = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double mu20 = 0.0;
    double mu11 = 0.0;
    double mu02 = 0.0;
    double mu30 = 0.0;
    double mu21 = 0.0;
    double mu12 = 0.0;
    double mu03 = 0.0;
};

class HuMoments {
public:
    static constexpr int kCount = 7;

    static RegionMoments accumulate(const std::vector<int>& labels, int width, int height,
                                    int target) {
        RegionMoments m;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (labels[static_cast<size_t>(y * width + x)] != target) {
                    continue;
                }
                m.area += 1.0;
                m.cx += x;
                m.cy += y;
            }
        }
        if (m.area <= 0.0) {
            return m;
        }
        m.cx /= m.area;
        m.cy /= m.area;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (labels[static_cast<size_t>(y * width + x)] != target) {
                    continue;
                }
                const double dx = x - m.cx;
                const double dy = y - m.cy;
                m.mu20 += dx * dx;
                m.mu11 += dx * dy;
                m.mu02 += dy * dy;
                m.mu30 += dx * dx * dx;
                m.mu21 += dx * dx * dy;
                m.mu12 += dx * dy * dy;
                m.mu03 += dy * dy * dy;
            }
        }
        return m;
    }

    // Log-compressed so the seven invariants, whose raw magnitudes span many
    // decades, contribute comparably to an L2 distance.
    static std::array<float, kCount> compute(const RegionMoments& m) {
        std::array<float, kCount> hu{};
        if (m.area <= 0.0) {
            return hu;
        }
        const double n20 = m.mu20 / std::pow(m.area, 2.0);
        const double n11 = m.mu11 / std::pow(m.area, 2.0);
        const double n02 = m.mu02 / std::pow(m.area, 2.0);
        const double n30 = m.mu30 / std::pow(m.area, 2.5);
        const double n21 = m.mu21 / std::pow(m.area, 2.5);
        const double n12 = m.mu12 / std::pow(m.area, 2.5);
        const double n03 = m.mu03 / std::pow(m.area, 2.5);

        const double h0 = n20 + n02;
        const double h1 = (n20 - n02) * (n20 - n02) + 4.0 * n11 * n11;
        const double h2 = (n30 - 3.0 * n12) * (n30 - 3.0 * n12) +
                          (3.0 * n21 - n03) * (3.0 * n21 - n03);
        const double h3 = (n30 + n12) * (n30 + n12) + (n21 + n03) * (n21 + n03);
        const double h4 = (n30 - 3.0 * n12) * (n30 + n12) *
                              ((n30 + n12) * (n30 + n12) - 3.0 * (n21 + n03) * (n21 + n03)) +
                          (3.0 * n21 - n03) * (n21 + n03) *
                              (3.0 * (n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03));
        const double h5 = (n20 - n02) * ((n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03)) +
                          4.0 * n11 * (n30 + n12) * (n21 + n03);
        const double h6 = (3.0 * n21 - n03) * (n30 + n12) *
                              ((n30 + n12) * (n30 + n12) - 3.0 * (n21 + n03) * (n21 + n03)) -
                          (n30 - 3.0 * n12) * (n21 + n03) *
                              (3.0 * (n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03));

        const double raw[kCount] = {h0, h1, h2, h3, h4, h5, h6};
        for (int i = 0; i < kCount; ++i) {
            const double v = raw[i];
            hu[static_cast<size_t>(i)] =
                static_cast<float>((v < 0.0 ? -1.0 : 1.0) * std::log10(1.0 + std::fabs(v) * 1e6));
        }
        return hu;
    }
};

// ---------------------------------------------------------------------------
// Interpretable geometry
// ---------------------------------------------------------------------------

struct GeometryTraits {
    float circularity = 0.0f;    // 4*pi*A / P^2, 1.0 for a disc
    float convexity = 0.0f;      // A / A_hull, 1.0 for a convex shape
    float elongation = 0.0f;     // 1 - minor/major from the inertia tensor
    float rectangularity = 0.0f; // A / A_oriented_bbox, 1.0 for a rectangle
    float extent = 0.0f;         // A / A_axis_aligned_bbox
    float corner_density = 0.0f; // simplified vertices per unit normalized perimeter
};

class ShapeGeometry {
public:
    static std::vector<Vec2> convex_hull(std::vector<Vec2> points) {
        if (points.size() < 3) {
            return points;
        }
        std::sort(points.begin(), points.end(), [](const Vec2& a, const Vec2& b) {
            return a.x != b.x ? a.x < b.x : a.y < b.y;
        });
        auto cross = [](const Vec2& o, const Vec2& a, const Vec2& b) {
            return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
        };
        std::vector<Vec2> hull(points.size() * 2);
        size_t k = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            while (k >= 2 && cross(hull[k - 2], hull[k - 1], points[i]) <= 0.0f) {
                --k;
            }
            hull[k++] = points[i];
        }
        for (size_t i = points.size() - 1, lower = k + 1; i > 0; --i) {
            while (k >= lower && cross(hull[k - 2], hull[k - 1], points[i - 1]) <= 0.0f) {
                --k;
            }
            hull[k++] = points[i - 1];
        }
        hull.resize(k > 0 ? k - 1 : 0);
        return hull;
    }

    static float polygon_area(const std::vector<Vec2>& polygon) {
        if (polygon.size() < 3) {
            return 0.0f;
        }
        float sum = 0.0f;
        for (size_t i = 0; i < polygon.size(); ++i) {
            const Vec2& a = polygon[i];
            const Vec2& b = polygon[(i + 1) % polygon.size()];
            sum += a.x * b.y - b.x * a.y;
        }
        return std::fabs(sum) * 0.5f;
    }

    static float polygon_perimeter(const std::vector<Vec2>& polygon) {
        float sum = 0.0f;
        for (size_t i = 0; i < polygon.size(); ++i) {
            const Vec2& a = polygon[i];
            const Vec2& b = polygon[(i + 1) % polygon.size()];
            sum += std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
        }
        return sum;
    }

    static GeometryTraits measure(const std::vector<Vec2>& polygon, const RegionMoments& moments,
                                  int simplified_vertices) {
        GeometryTraits traits;
        const float area = polygon_area(polygon);
        const float perimeter = polygon_perimeter(polygon);
        if (area <= 0.0f || perimeter <= 0.0f) {
            return traits;
        }
        traits.circularity =
            std::clamp(4.0f * math::kPi * area / (perimeter * perimeter), 0.0f, 1.0f);

        const std::vector<Vec2> hull = convex_hull(polygon);
        const float hull_area = polygon_area(hull);
        traits.convexity = hull_area > 0.0f ? std::clamp(area / hull_area, 0.0f, 1.0f) : 0.0f;

        // Principal axes from the inertia tensor give both the elongation and
        // the frame for a tight oriented bounding box, so rectangularity does
        // not depend on how the object happens to sit in the image.
        float axis = 0.0f;
        if (moments.area > 0.0) {
            const double a = moments.mu20 / moments.area;
            const double b = moments.mu11 / moments.area;
            const double c = moments.mu02 / moments.area;
            const double diff = std::sqrt((a - c) * (a - c) + 4.0 * b * b);
            const double major = 0.5 * (a + c + diff);
            const double minor = 0.5 * (a + c - diff);
            if (major > 1e-9) {
                traits.elongation =
                    static_cast<float>(1.0 - std::sqrt(std::max(0.0, minor) / major));
            }
            axis = 0.5f * static_cast<float>(std::atan2(2.0 * b, a - c));
        }

        const float cs = std::cos(-axis);
        const float sn = std::sin(-axis);
        float min_u = 1e30f;
        float max_u = -1e30f;
        float min_v = 1e30f;
        float max_v = -1e30f;
        float min_x = 1e30f;
        float max_x = -1e30f;
        float min_y = 1e30f;
        float max_y = -1e30f;
        for (const Vec2& p : polygon) {
            const float u = cs * p.x - sn * p.y;
            const float v = sn * p.x + cs * p.y;
            min_u = std::min(min_u, u);
            max_u = std::max(max_u, u);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
        const float oriented = (max_u - min_u) * (max_v - min_v);
        const float aligned = (max_x - min_x) * (max_y - min_y);
        traits.rectangularity = oriented > 0.0f ? std::clamp(area / oriented, 0.0f, 1.0f) : 0.0f;
        traits.extent = aligned > 0.0f ? std::clamp(area / aligned, 0.0f, 1.0f) : 0.0f;
        // Vertices per unit of scale-normalized perimeter: a triangle and a
        // scaled-up triangle report the same density, a gear reports far more.
        traits.corner_density =
            std::clamp(static_cast<float>(simplified_vertices) * std::sqrt(area) / perimeter /
                           8.0f,
                       0.0f, 1.0f);
        return traits;
    }
};

// ---------------------------------------------------------------------------
// Combined signature
// ---------------------------------------------------------------------------

struct ShapeDescriptor {
    static constexpr int kFourierDims = (EllipticFourier::kHarmonics - 1) * 4 + 1;
    static constexpr int kHuDims = HuMoments::kCount;
    static constexpr int kGeometryDims = 6;
    static constexpr int kDims = kFourierDims + kHuDims + kGeometryDims;

    std::array<FourierHarmonic, EllipticFourier::kHarmonics> harmonics{};
    std::array<float, HuMoments::kCount> hu{};
    GeometryTraits traits;
    bool valid = false;

    // Block weights. The Fourier block has by far the most dimensions, so
    // without down-weighting it would swamp the other two in an L2 metric.
    static constexpr float kFourierWeight = 1.0f;
    static constexpr float kHuWeight = 0.35f;
    static constexpr float kGeometryWeight = 1.6f;

    std::array<float, kDims> flatten() const {
        std::array<float, kDims> out{};
        int at = 0;
        // Harmonic 1 normalizes to (1, 0, 0, d1); only d1 carries information.
        out[static_cast<size_t>(at++)] = kFourierWeight * harmonics[0].d;
        for (int h = 1; h < EllipticFourier::kHarmonics; ++h) {
            const FourierHarmonic& harmonic = harmonics[static_cast<size_t>(h)];
            out[static_cast<size_t>(at++)] = kFourierWeight * harmonic.a;
            out[static_cast<size_t>(at++)] = kFourierWeight * harmonic.b;
            out[static_cast<size_t>(at++)] = kFourierWeight * harmonic.c;
            out[static_cast<size_t>(at++)] = kFourierWeight * harmonic.d;
        }
        for (float value : hu) {
            out[static_cast<size_t>(at++)] = kHuWeight * value;
        }
        out[static_cast<size_t>(at++)] = kGeometryWeight * traits.circularity;
        out[static_cast<size_t>(at++)] = kGeometryWeight * traits.convexity;
        out[static_cast<size_t>(at++)] = kGeometryWeight * traits.elongation;
        out[static_cast<size_t>(at++)] = kGeometryWeight * traits.rectangularity;
        out[static_cast<size_t>(at++)] = kGeometryWeight * traits.extent;
        out[static_cast<size_t>(at++)] = kGeometryWeight * traits.corner_density;
        return out;
    }

    static float distance(const ShapeDescriptor& a, const ShapeDescriptor& b) {
        return distance(a.flatten(), b.flatten());
    }

    static float distance(const std::array<float, kDims>& a, const std::array<float, kDims>& b) {
        float sum = 0.0f;
        for (int i = 0; i < kDims; ++i) {
            const float d = a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)];
            sum += d * d;
        }
        return std::sqrt(sum);
    }
};

inline ShapeDescriptor describe_shape(const std::vector<Vec2>& polygon,
                                      const RegionMoments& moments, int simplified_vertices) {
    ShapeDescriptor descriptor;
    if (polygon.size() < 3 || moments.area <= 0.0) {
        return descriptor;
    }
    descriptor.harmonics = EllipticFourier::normalize(EllipticFourier::encode(polygon));
    descriptor.hu = HuMoments::compute(moments);
    descriptor.traits = ShapeGeometry::measure(polygon, moments, simplified_vertices);
    descriptor.valid = true;
    return descriptor;
}

// Human-readable archetype for a descriptor, used to name vocabulary entries
// so a classification can be explained rather than just numbered.
inline std::string name_archetype(const GeometryTraits& traits) {
    std::string size_word;
    if (traits.elongation > 0.80f) {
        size_word = "sliver";
    } else if (traits.elongation > 0.55f) {
        size_word = "elongated";
    } else {
        size_word = "compact";
    }

    std::string form_word;
    if (traits.circularity > 0.80f && traits.convexity > 0.90f) {
        form_word = "disc";
    } else if (traits.rectangularity > 0.80f && traits.convexity > 0.88f) {
        form_word = "quad";
    } else if (traits.convexity < 0.70f) {
        form_word = "ragged";
    } else if (traits.corner_density > 0.45f) {
        form_word = "polygon";
    } else {
        form_word = "blob";
    }
    return size_word + "-" + form_word;
}

}  // namespace descriptors
