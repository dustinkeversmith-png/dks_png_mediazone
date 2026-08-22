#pragma once

#include "types.hpp"

namespace contour {

inline ImageBuffer dilate_binary(const ImageBuffer& src, int radius) {
    ImageBuffer out = src;
    if (radius <= 0) {
        return out;
    }
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            if (src.at(x, y) == 0) {
                continue;
            }
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx >= 0 && yy >= 0 && xx < src.width && yy < src.height) {
                        out.at(xx, yy) = 255;
                    }
                }
            }
        }
    }
    return out;
}

} // namespace contour
