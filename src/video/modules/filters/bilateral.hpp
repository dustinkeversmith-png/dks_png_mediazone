#pragma once

#include "contour_kit/types.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace contour {

class BilateralFilter {
public:
    int radius = 3;
    float sigma_s = 2.0f;
    float sigma_r = 28.0f;

    ImageBuffer apply(const ImageBuffer& src) const {
        ImageBuffer out = make_gray(src.width, src.height);
        const float inv_s = 1.0f / (2.0f * sigma_s * sigma_s);
        const float inv_r = 1.0f / (2.0f * sigma_r * sigma_r);
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                const float ic = src.gray(x, y);
                float acc = 0, wsum = 0;
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int xx = std::clamp(x + dx, 0, src.width - 1);
                        const int yy = std::clamp(y + dy, 0, src.height - 1);
                        const float iv = src.gray(xx, yy);
                        const float ws = std::exp(-(dx * dx + dy * dy) * inv_s);
                        const float wr = std::exp(-((iv - ic) * (iv - ic)) * inv_r);
                        const float w = ws * wr;
                        acc += w * iv;
                        wsum += w;
                    }
                }
                out.at(x, y) = static_cast<uint8_t>(std::clamp(acc / std::max(wsum, 1e-6f), 0.0f, 255.0f));
            }
        }
        return out;
    }
};

}  // namespace contour
