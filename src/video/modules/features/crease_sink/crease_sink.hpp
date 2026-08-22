#pragma once

#include "contour_kit/types.hpp"
#include "featurizations/laplace_gaussian/laplace_gaussian.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace contour {

// Dark crease / valley filter: grayscale black top-hat + LoG sink + 1D NMS spine.
// Collapses the two parallel walls of a fold into a single topological edge.
class CreaseSink {
public:
    int radius = 2;
    float sigma = 1.6f;

    static ImageBuffer gray_dilate(const ImageBuffer& src, int r) {
        ImageBuffer out = make_gray(src.width, src.height, 0);
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                float m = 0;
                for (int dy = -r; dy <= r; ++dy) {
                    for (int dx = -r; dx <= r; ++dx) {
                        const int xx = std::clamp(x + dx, 0, src.width - 1);
                        const int yy = std::clamp(y + dy, 0, src.height - 1);
                        m = std::max(m, src.gray(xx, yy));
                    }
                }
                out.at(x, y) = static_cast<uint8_t>(std::clamp(m, 0.0f, 255.0f));
            }
        }
        return out;
    }

    static ImageBuffer gray_erode(const ImageBuffer& src, int r) {
        ImageBuffer out = make_gray(src.width, src.height, 255);
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                float m = 255;
                for (int dy = -r; dy <= r; ++dy) {
                    for (int dx = -r; dx <= r; ++dx) {
                        const int xx = std::clamp(x + dx, 0, src.width - 1);
                        const int yy = std::clamp(y + dy, 0, src.height - 1);
                        m = std::min(m, src.gray(xx, yy));
                    }
                }
                out.at(x, y) = static_cast<uint8_t>(std::clamp(m, 0.0f, 255.0f));
            }
        }
        return out;
    }

    ImageBuffer black_tophat(const ImageBuffer& src) const {
        const ImageBuffer closed = gray_erode(gray_dilate(src, radius), radius);
        ImageBuffer hat = make_gray(src.width, src.height, 0);
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                const float v = static_cast<float>(closed.at(x, y)) - src.gray(x, y);
                hat.at(x, y) = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
            }
        }
        return hat;
    }

    Field valley_response(const ImageBuffer& src) const {
        LaplaceGaussian log;
        log.sigma = sigma;
        Field lo = log.response(src);
        const ImageBuffer hat = black_tophat(src);
        Field out = make_field(src.width, src.height, 0);
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                // LoG center is negative on a dark blob after mean-subtract; invert so sinks > 0.
                out.at(x, y) = std::max(0.0f, -lo.at(x, y)) + hat.at(x, y);
            }
        }
        return out;
    }

    static ImageBuffer nms_spine(const Field& valley, float thr_frac = 0.12f) {
        ImageBuffer out = make_gray(valley.width, valley.height, 0);
        float vmax = 1e-6f;
        for (float v : valley.data) {
            vmax = std::max(vmax, v);
        }
        const float thr = thr_frac * vmax;
        for (int y = 1; y < valley.height - 1; ++y) {
            for (int x = 1; x < valley.width - 1; ++x) {
                const float v = valley.at(x, y);
                if (v < thr) {
                    continue;
                }
                const float gx = valley.at(x + 1, y) - valley.at(x - 1, y);
                const float gy = valley.at(x, y + 1) - valley.at(x, y - 1);
                const float g = std::hypot(gx, gy);
                int ax = 1, ay = 0;
                if (g > 1e-6f) {
                    ax = static_cast<int>(std::lround(gx / g));
                    ay = static_cast<int>(std::lround(gy / g));
                    ax = std::clamp(ax, -1, 1);
                    ay = std::clamp(ay, -1, 1);
                    if (ax == 0 && ay == 0) {
                        ax = 1;
                    }
                }
                const float a = valley.at(x + ax, y + ay);
                const float b = valley.at(x - ax, y - ay);
                if (v >= a && v >= b) {
                    out.at(x, y) = 255;
                }
            }
        }
        return out;
    }

    ImageBuffer detect(const ImageBuffer& src) const { return nms_spine(valley_response(src)); }
};

}  // namespace contour
