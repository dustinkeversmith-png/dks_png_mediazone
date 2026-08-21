#pragma once

#include "../contour_kit/types.hpp"
#include "../filters/sobel.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace contour {

class SnakeActiveContour {
public:
    float alpha = 0.08f;   // tension
    float beta = 0.15f;    // rigidity
    float gamma = 1.2f;    // external force
    float dt = 0.15f;
    int iterations = 80;
    int n_points = 96;

    Polyline evolve(const ImageBuffer& image, const Rect& bbox) const {
        SobelFilter sobel;
        sobel.compute(image);
        Field energy = sobel.edge_energy();
        Field ex = make_field(image.width, image.height);
        Field ey = make_field(image.width, image.height);
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                const float e0 = energy.at(x, y);
                ex.at(x, y) = energy.at(std::min(image.width - 1, x + 1), y) - e0;
                ey.at(x, y) = energy.at(x, std::min(image.height - 1, y + 1)) - e0;
            }
        }

        std::vector<Vec2> v(static_cast<size_t>(n_points));
        const float cx = bbox.x + bbox.w * 0.5f;
        const float cy = bbox.y + bbox.h * 0.5f;
        const float rx = std::max(4.0f, bbox.w * 0.5f);
        const float ry = std::max(4.0f, bbox.h * 0.5f);
        for (int i = 0; i < n_points; ++i) {
            const float t = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(n_points);
            v[static_cast<size_t>(i)] = {cx + rx * std::cos(t), cy + ry * std::sin(t)};
        }

        auto wrap = [&](int i) {
            const int n = static_cast<int>(v.size());
            return (i % n + n) % n;
        };
        for (int it = 0; it < iterations; ++it) {
            std::vector<Vec2> nv = v;
            for (int i = 0; i < static_cast<int>(v.size()); ++i) {
                const Vec2& vm2 = v[static_cast<size_t>(wrap(i - 2))];
                const Vec2& vm1 = v[static_cast<size_t>(wrap(i - 1))];
                const Vec2& vc = v[static_cast<size_t>(i)];
                const Vec2& vp1 = v[static_cast<size_t>(wrap(i + 1))];
                const Vec2& vp2 = v[static_cast<size_t>(wrap(i + 2))];
                const Vec2 tension = (vm1 + vp1) * 0.5f - vc;  // discrete v''
                const Vec2 rigid = vm2 * -0.25f + vm1 + vc * -1.5f + vp1 + vp2 * -0.25f;
                const int x = std::clamp(static_cast<int>(vc.x), 0, image.width - 1);
                const int y = std::clamp(static_cast<int>(vc.y), 0, image.height - 1);
                const Vec2 ext{-ex.at(x, y), -ey.at(x, y)};  // descend E_ext
                Vec2 f = tension * alpha - rigid * beta + ext * gamma;
                nv[static_cast<size_t>(i)] = vc + f * dt;
                nv[static_cast<size_t>(i)].x = std::clamp(nv[static_cast<size_t>(i)].x, 1.0f,
                                                          static_cast<float>(image.width - 2));
                nv[static_cast<size_t>(i)].y = std::clamp(nv[static_cast<size_t>(i)].y, 1.0f,
                                                          static_cast<float>(image.height - 2));
            }
            v.swap(nv);
        }
        Polyline p;
        p.points = std::move(v);
        p.closed = true;
        return p;
    }
};

}  // namespace contour
