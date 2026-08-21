#pragma once

#include "../math/vision_types.hpp"
#include "../boundary_tracing/moore_neighbor.hpp"

#include <array>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

namespace vision {

// Geometry-aware lookup: quantized (aspect ratio, compactness = 4πA/P²).
class GeometryLUT {
public:
    static constexpr int kAspectBins = 8;
    static constexpr int kCompactBins = 8;

    struct Stats {
        float aspect = 1.0f;
        float compactness = 1.0f;
        float area = 0.0f;
        float perimeter = 0.0f;
        Rect bbox;
    };

    std::array<std::array<std::string, kCompactBins>, kAspectBins> table{};

    static Stats measure(const GrayImage& image, uint8_t thr = 128) {
        Stats s;
        int minx = image.width, miny = image.height, maxx = -1, maxy = -1;
        int area = 0;
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                if (!image.fg(x, y, thr)) {
                    continue;
                }
                ++area;
                minx = std::min(minx, x);
                miny = std::min(miny, y);
                maxx = std::max(maxx, x);
                maxy = std::max(maxy, y);
            }
        }
        s.area = static_cast<float>(area);
        if (area == 0) {
            return s;
        }
        s.bbox = {static_cast<float>(minx), static_cast<float>(miny),
                  static_cast<float>(maxx - minx + 1), static_cast<float>(maxy - miny + 1)};
        s.aspect = s.bbox.w / std::max(1.0f, s.bbox.h);
        auto contour = MooreNeighborTracer::trace(image, thr);
        s.perimeter = std::max(1.0f, contour.perimeter);
        s.compactness = (4.0f * kPi * s.area) / (s.perimeter * s.perimeter);
        return s;
    }

    static int aspect_bin(float aspect) {
        const float loga = std::log(std::max(0.05f, aspect));
        const int b = static_cast<int>(std::floor((loga + 2.0f) / 4.0f * kAspectBins));
        return std::clamp(b, 0, kAspectBins - 1);
    }

    static int compact_bin(float c) {
        const int b = static_cast<int>(std::floor(std::clamp(c, 0.0f, 0.999f) * kCompactBins));
        return std::clamp(b, 0, kCompactBins - 1);
    }

    void insert(const Stats& s, const std::string& label) {
        table[static_cast<size_t>(aspect_bin(s.aspect))][static_cast<size_t>(compact_bin(s.compactness))] = label;
    }

    std::string query(const Stats& s) const {
        return table[static_cast<size_t>(aspect_bin(s.aspect))][static_cast<size_t>(compact_bin(s.compactness))];
    }
};

}  // namespace vision
