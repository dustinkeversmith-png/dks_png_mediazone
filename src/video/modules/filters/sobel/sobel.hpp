#pragma once

#include "math/contour_compat.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace contour {

class SobelFilter {
public:
    Field mag;
    Field gx;
    Field gy;

    void compute(const ImageBuffer& im) {
        mag = make_field(im.width, im.height);
        gx = make_field(im.width, im.height);
        gy = make_field(im.width, im.height);
        for (int y = 0; y < im.height; ++y) {
            for (int x = 0; x < im.width; ++x) {
                auto I = [&](int xx, int yy) {
                    return im.gray(std::clamp(xx, 0, im.width - 1), std::clamp(yy, 0, im.height - 1));
                };
                const float xg = (I(x + 1, y - 1) + 2 * I(x + 1, y) + I(x + 1, y + 1)) -
                                 (I(x - 1, y - 1) + 2 * I(x - 1, y) + I(x - 1, y + 1));
                const float yg = (I(x - 1, y + 1) + 2 * I(x, y + 1) + I(x + 1, y + 1)) -
                                 (I(x - 1, y - 1) + 2 * I(x, y - 1) + I(x + 1, y - 1));
                gx.at(x, y) = xg * 0.125f;
                gy.at(x, y) = yg * 0.125f;
                mag.at(x, y) = std::hypot(gx.at(x, y), gy.at(x, y));
            }
        }
    }

    Field laplacian(const ImageBuffer& im) const {
        Field lap = make_field(im.width, im.height);
        for (int y = 0; y < im.height; ++y) {
            for (int x = 0; x < im.width; ++x) {
                auto I = [&](int xx, int yy) {
                    return im.gray(std::clamp(xx, 0, im.width - 1), std::clamp(yy, 0, im.height - 1));
                };
                lap.at(x, y) = I(x + 1, y) + I(x - 1, y) + I(x, y + 1) + I(x, y - 1) - 4 * I(x, y);
            }
        }
        return lap;
    }

    Field edge_energy() const {
        Field e = make_field(mag.width, mag.height);
        float mmax = 1e-6f;
        for (float v : mag.data) {
            mmax = std::max(mmax, v);
        }
        for (size_t i = 0; i < mag.data.size(); ++i) {
            const float n = mag.data[i] / mmax;
            e.data[i] = -n * n;  // E_ext = -|∇I|² (normalized)
        }
        return e;
    }
};

}  // namespace contour
