#pragma once

#include "contour_kit/types.hpp"
#include "featurizations/sdf/8SSEDT.hpp"
#include <algorithm>
#include <cmath>

namespace contour {

// Euclidean distance transform (Felzenszwalb–Huttenlocher 1D envelopes).
// unsigned: distance to nearest seed/foreground. signed: negative inside a mask.
class EuclideanDistanceTransform {
public:
    static Field distance_to_zero(const Field& seeds) {
        Field sq = ExactSDF::squared_edt(seeds);
        Field out = make_field(sq.width, sq.height, 0);
        for (size_t i = 0; i < sq.data.size(); ++i) {
            const float v = sq.data[i];
            out.data[i] = v >= ExactSDF::kInf * 0.5f ? ExactSDF::kInf : std::sqrt(std::max(0.0f, v));
        }
        return out;
    }

    static Field to_foreground(const ImageBuffer& mask, uint8_t thr = 127) {
        Field seeds = make_field(mask.width, mask.height, ExactSDF::kInf);
        for (int y = 0; y < mask.height; ++y) {
            for (int x = 0; x < mask.width; ++x) {
                if (mask.at(x, y) > thr) {
                    seeds.at(x, y) = 0.0f;
                }
            }
        }
        return distance_to_zero(seeds);
    }

    static Field to_background(const ImageBuffer& mask, uint8_t thr = 127) {
        Field seeds = make_field(mask.width, mask.height, ExactSDF::kInf);
        for (int y = 0; y < mask.height; ++y) {
            for (int x = 0; x < mask.width; ++x) {
                if (mask.at(x, y) <= thr) {
                    seeds.at(x, y) = 0.0f;
                }
            }
        }
        return distance_to_zero(seeds);
    }

    static Field signed_from_mask(const ImageBuffer& mask, uint8_t thr = 127) {
        return ExactSDF::from_mask(mask, thr);
    }
};

}  // namespace contour
