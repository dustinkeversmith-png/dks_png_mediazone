#pragma once

#include "math/contour_compat.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace contour {

class LaplaceGaussian {
public:
    float sigma = 1.4f;

    Field response(const ImageBuffer& src) const {
        const int r = std::max(2, static_cast<int>(std::ceil(3.0f * sigma)));
        std::vector<float> k(static_cast<size_t>((2 * r + 1) * (2 * r + 1)));
        float sum = 0;
        const float s2 = sigma * sigma;
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                const float d2 = static_cast<float>(dx * dx + dy * dy);
                const float val = (d2 - 2.0f * s2) / (s2 * s2) * std::exp(-d2 / (2.0f * s2));
                k[static_cast<size_t>((dy + r) * (2 * r + 1) + (dx + r))] = val;
                sum += val;
            }
        }
        const float mean = sum / static_cast<float>(k.size());
        for (float& c : k) {
            c -= mean;
        }
        Field out = make_field(src.width, src.height, 0);
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                float acc = 0;
                for (int dy = -r; dy <= r; ++dy) {
                    for (int dx = -r; dx <= r; ++dx) {
                        const int xx = std::clamp(x + dx, 0, src.width - 1);
                        const int yy = std::clamp(y + dy, 0, src.height - 1);
                        acc += k[static_cast<size_t>((dy + r) * (2 * r + 1) + (dx + r))] * src.gray(xx, yy);
                    }
                }
                out.at(x, y) = acc;
            }
        }
        return out;
    }

    ImageBuffer zero_crossings(const ImageBuffer& src) const {
        Field log = response(src);
        ImageBuffer zc = make_gray(src.width, src.height, 0);
        for (int y = 1; y < src.height - 1; ++y) {
            for (int x = 1; x < src.width - 1; ++x) {
                const float v = log.at(x, y);
                if ((v > 0) != (log.at(x - 1, y) > 0) || (v > 0) != (log.at(x, y - 1) > 0)) {
                    zc.at(x, y) = 255;
                }
            }
        }
        return zc;
    }
};

}  // namespace contour
