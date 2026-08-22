#pragma once

#include "math/contour_compat.hpp"
#include "filters/sobel/sobel.hpp"
#include "filters/edge/canny/canny.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

namespace contour {

// Gradient Vector Flow (Xu & Prince): diffuse Canny/edge gradients to widen capture range.
class GradientVectorFlow {
public:
    float mu = 0.2f;
    float dt = 0.15f;
    int iterations = 20;
    Field u;
    Field v;

    void compute(const ImageBuffer& edges_or_gray, bool already_edges = false) {
        ImageBuffer edge = edges_or_gray;
        if (!already_edges) {
            Canny canny;
            edge = canny.detect(edges_or_gray);
        }
        SobelFilter sobel;
        sobel.compute(edge);
        Field fx = sobel.gx;
        Field fy = sobel.gy;
        const int w = edge.width;
        const int h = edge.height;
        Field mag2 = make_field(w, h);
        float gmax = 1e-6f;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                mag2.at(x, y) = fx.at(x, y) * fx.at(x, y) + fy.at(x, y) * fy.at(x, y);
                gmax = std::max(gmax, mag2.at(x, y));
            }
        }
        const float s = std::sqrt(gmax);
        for (size_t i = 0; i < mag2.data.size(); ++i) {
            mag2.data[i] /= gmax;
            fx.data[i] /= s;
            fy.data[i] /= s;
        }
        u = fx;
        v = fy;
        for (int it = 0; it < iterations; ++it) {
            Field nu = u;
            Field nv = v;
            for (int y = 1; y < h - 1; ++y) {
                for (int x = 1; x < w - 1; ++x) {
                    const float lap_u =
                        u.at(x + 1, y) + u.at(x - 1, y) + u.at(x, y + 1) + u.at(x, y - 1) - 4 * u.at(x, y);
                    const float lap_v =
                        v.at(x + 1, y) + v.at(x - 1, y) + v.at(x, y + 1) + v.at(x, y - 1) - 4 * v.at(x, y);
                    const float g2 = mag2.at(x, y);
                    float uu = u.at(x, y) + dt * (mu * lap_u - (u.at(x, y) - fx.at(x, y)) * g2);
                    float vv = v.at(x, y) + dt * (mu * lap_v - (v.at(x, y) - fy.at(x, y)) * g2);
                    nu.at(x, y) = std::clamp(uu, -2.0f, 2.0f);
                    nv.at(x, y) = std::clamp(vv, -2.0f, 2.0f);
                }
            }
            std::swap(u, nu);
            std::swap(v, nv);
        }
    }

    Vec2 sample(float x, float y) const { return {u.sample(x, y), v.sample(x, y)}; }
};

}  // namespace contour
