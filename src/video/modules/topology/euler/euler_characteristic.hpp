#pragma once

#include "../../segmentation/ccl/connected_components.hpp"

#include <algorithm>
#include <vector>

namespace vision {

class EulerCharacteristic {
public:
    struct Result {
        int components = 0;
        int holes = 0;
        int chi = 0;
    };

    static Result compute(const GrayImage& image, uint8_t thr = 128) {
        Result r;
        auto fg = ConnectedComponentLabeler::label(image, thr);
        r.components = static_cast<int>(fg.components.size());

        GrayImage inv = image;
        for (uint8_t& p : inv.data) {
            p = p > thr ? 0 : 255;
        }
        auto bg = ConnectedComponentLabeler::label(inv, thr);

        int holes = 0;
        for (const auto& c : bg.components) {
            const bool touches_border =
                c.bbox.x <= 0.5f || c.bbox.y <= 0.5f ||
                c.bbox.x1() >= static_cast<float>(image.width) - 0.5f ||
                c.bbox.y1() >= static_cast<float>(image.height) - 0.5f;
            if (!touches_border) {
                ++holes;
            }
        }
        r.holes = holes;
        r.chi = r.components - r.holes;
        return r;
    }
};

}  // namespace vision
