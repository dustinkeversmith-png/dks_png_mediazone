#pragma once

#include "contour_kit/types.hpp"
#include <algorithm>
#include <vector>

namespace contour {

class MorphClean {
public:
    static ImageBuffer erode(const ImageBuffer& src, uint8_t thr = 127) {
        ImageBuffer out = make_gray(src.width, src.height, 0);
        for (int y = 1; y < src.height - 1; ++y) {
            for (int x = 1; x < src.width - 1; ++x) {
                bool keep = true;
                for (int dy = -1; dy <= 1 && keep; ++dy) {
                    for (int dx = -1; dx <= 1 && keep; ++dx) {
                        if (src.at(x + dx, y + dy) <= thr) {
                            keep = false;
                        }
                    }
                }
                out.at(x, y) = keep ? 255 : 0;
            }
        }
        return out;
    }

    static ImageBuffer dilate(const ImageBuffer& src, uint8_t thr = 127) {
        ImageBuffer out = make_gray(src.width, src.height, 0);
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                if (src.at(x, y) <= thr) {
                    continue;
                }
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
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

    static ImageBuffer opening(const ImageBuffer& src) { return dilate(erode(src)); }
    static ImageBuffer closing(const ImageBuffer& src) { return erode(dilate(src)); }

    // Closing only: seal 1-px gaps without eroding hair/wires (opening would).
    static ImageBuffer clean(const ImageBuffer& src) { return closing(src); }
};

}  // namespace contour
