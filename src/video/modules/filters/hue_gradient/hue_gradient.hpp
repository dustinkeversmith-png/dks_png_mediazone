#pragma once

#include "math/contour_compat.hpp"
#include "filters/lab_color/lab_color_space.hpp"
#include <algorithm>
#include <cmath>

namespace contour {

// Perceptual color/hue gradient: CIE76 ΔE_ab across 3×1 / 1×3 stencils.
class HueGradient {
public:
    Field mag;
    Field gx;
    Field gy;

    void compute(const ImageBuffer& im) {
        mag = make_field(im.width, im.height, 0);
        gx = make_field(im.width, im.height, 0);
        gy = make_field(im.width, im.height, 0);
        for (int y = 0; y < im.height; ++y) {
            for (int x = 0; x < im.width; ++x) {
                const Lab c = LabColor::at(im, x, y);
                const Lab l = LabColor::at(im, std::max(0, x - 1), y);
                const Lab r = LabColor::at(im, std::min(im.width - 1, x + 1), y);
                const Lab u = LabColor::at(im, x, std::max(0, y - 1));
                const Lab d = LabColor::at(im, x, std::min(im.height - 1, y + 1));
                const float dx = std::sqrt(std::max(0.0f, LabColor::delta2(r, l))) * 0.5f;
                const float dy = std::sqrt(std::max(0.0f, LabColor::delta2(d, u))) * 0.5f;
                gx.at(x, y) = dx;
                gy.at(x, y) = dy;
                mag.at(x, y) = std::hypot(dx, dy);
            }
        }
    }
};

}  // namespace contour
