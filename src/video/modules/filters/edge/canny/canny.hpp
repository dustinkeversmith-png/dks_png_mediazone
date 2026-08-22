#pragma once

#include "math/contour_compat.hpp"
#include "filters/sobel/sobel.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <queue>

namespace contour {

class Canny {
public:
    float low = 0.08f;
    float high = 0.24f;

    ImageBuffer detect(const ImageBuffer& src) const {
        SobelFilter sobel;
        sobel.compute(src);
        Field nms = make_field(src.width, src.height, 0);
        float mmax = 1e-6f;
        for (float v : sobel.mag.data) {
            mmax = std::max(mmax, v);
        }
        for (int y = 1; y < src.height - 1; ++y) {
            for (int x = 1; x < src.width - 1; ++x) {
                const float gx = sobel.gx.at(x, y);
                const float gy = sobel.gy.at(x, y);
                const float mag = sobel.mag.at(x, y);
                float a = 0, b = 0;
                const float ax = std::fabs(gx);
                const float ay = std::fabs(gy);
                if (ax > ay) {
                    a = sobel.mag.at(x + 1, y);
                    b = sobel.mag.at(x - 1, y);
                } else {
                    a = sobel.mag.at(x, y + 1);
                    b = sobel.mag.at(x, y - 1);
                }
                if (mag >= a && mag >= b) {
                    nms.at(x, y) = mag / mmax;
                }
            }
        }
        ImageBuffer strong = make_gray(src.width, src.height, 0);
        ImageBuffer weak = make_gray(src.width, src.height, 0);
        std::queue<std::pair<int, int>> q;
        for (int y = 1; y < src.height - 1; ++y) {
            for (int x = 1; x < src.width - 1; ++x) {
                const float v = nms.at(x, y);
                if (v >= high) {
                    strong.at(x, y) = 255;
                    q.push({x, y});
                } else if (v >= low) {
                    weak.at(x, y) = 255;
                }
            }
        }
        while (!q.empty()) {
            const auto [x, y] = q.front();
            q.pop();
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= src.width || yy >= src.height) {
                        continue;
                    }
                    if (weak.at(xx, yy) && !strong.at(xx, yy)) {
                        strong.at(xx, yy) = 255;
                        q.push({xx, yy});
                    }
                }
            }
        }
        return strong;
    }
};

}  // namespace contour
