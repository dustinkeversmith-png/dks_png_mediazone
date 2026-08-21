#pragma once

#include "../../math/distance_transform.hpp"

#include <vector>
#include <cmath>
#include <algorithm>

namespace vision {

class MedialAxis {
public:
    struct Result {
        GrayImage skeleton;
        int skeleton_pixels = 0;
        float mean_radius = 0.0f;
    };

    static Result extract(const GrayImage& image, uint8_t thr = 128, float min_radius = 0.75f) {
        Result r;
        r.skeleton.width = image.width;
        r.skeleton.height = image.height;
        r.skeleton.pixels.assign(image.pixels.size(), 0);

        GrayImage interior = image;
        for (size_t i = 0; i < interior.pixels.size(); ++i) {
            interior.pixels[i] = image.pixels[i] > thr ? static_cast<uint8_t>(0) : static_cast<uint8_t>(255);
        }
        const auto dt = FelzenszwalbDistanceTransform::edt(interior, thr);
        double radius_sum = 0.0;
        const int w = image.width;
        const int h = image.height;
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                if (!image.fg(x, y, thr)) {
                    continue;
                }
                const float v = dt[static_cast<size_t>(y * w + x)];
                if (v < min_radius) {
                    continue;
                }
                bool ridge = true;
                int ge = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const float n = dt[static_cast<size_t>((y + dy) * w + (x + dx))];
                        if (n > v + 0.25f) {
                            ridge = false;
                        }
                        if (n <= v + 0.25f) {
                            ++ge;
                        }
                    }
                }
                if (ridge && ge >= 4) {
                    r.skeleton.at(x, y) = 255;
                    ++r.skeleton_pixels;
                    radius_sum += v;
                }
            }
        }
        r.mean_radius = r.skeleton_pixels > 0
                            ? static_cast<float>(radius_sum / r.skeleton_pixels)
                            : 0.0f;
        return r;
    }
};

}  // namespace vision
