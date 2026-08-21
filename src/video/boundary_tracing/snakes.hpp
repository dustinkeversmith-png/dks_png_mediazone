#pragma once

#include "../contour_kit/types.hpp"
#include "../contour_kit/metrics.hpp"
#include "../bbox/bbox_auto.hpp"
#include "../filters/bilateral.hpp"
#include "../featurizations/edge/canny.hpp"
#include "../featurizations/gvh/gvh.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace contour {

class SnakeActiveContour {
public:
    float alpha = 0.06f;
    float beta = 0.10f;
    float gamma = 1.6f;
    float dt = 0.18f;
    int iterations = 70;
    int n_points = 96;

    Polyline evolve(const ImageBuffer& image, const Rect& bbox) const {
        BilateralFilter bf;
        const ImageBuffer smooth = bf.apply(image);
        Canny canny;
        canny.low = 0.12f;
        canny.high = 0.34f;
        const ImageBuffer edges = canny.detect(smooth);
        GradientVectorFlow gvf;
        gvf.compute(edges, true);

        const Rect init = {bbox.x + bbox.w * 0.08f, bbox.y + bbox.h * 0.08f, bbox.w * 0.84f,
                           bbox.h * 0.84f};
        std::vector<Vec2> hull = {
            {init.x, init.y},
            {init.x1(), init.y},
            {init.x1(), init.y1()},
            {init.x, init.y1()},
        };
        hull = convex_hull(hull);
        std::vector<Vec2> v(static_cast<size_t>(n_points));
        float perim = 0;
        std::vector<float> acc(hull.size() + 1, 0);
        for (size_t i = 0; i < hull.size(); ++i) {
            perim += dist(hull[i], hull[(i + 1) % hull.size()]);
            acc[i + 1] = perim;
        }
        for (int i = 0; i < n_points; ++i) {
            float t = (static_cast<float>(i) / static_cast<float>(n_points)) * perim;
            size_t e = 1;
            while (e < acc.size() && acc[e] < t) {
                ++e;
            }
            const Vec2 a = hull[(e - 1) % hull.size()];
            const Vec2 b = hull[e % hull.size()];
            const float seg = std::max(1e-4f, acc[e] - acc[e - 1]);
            const float u = (t - acc[e - 1]) / seg;
            v[static_cast<size_t>(i)] = a + (b - a) * u;
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
                const Vec2 tension = (vm1 + vp1) * 0.5f - vc;
                const Vec2 rigid = vm2 * -0.25f + vm1 + vc * -1.5f + vp1 + vp2 * -0.25f;
                const Vec2 ext = gvf.sample(vc.x, vc.y);
                Vec2 f = tension * alpha - rigid * beta + ext * gamma;
                if (!std::isfinite(f.x) || !std::isfinite(f.y)) {
                    f = {0, 0};
                }
                nv[static_cast<size_t>(i)] = vc + f * dt;
                nv[static_cast<size_t>(i)].x =
                    std::clamp(nv[static_cast<size_t>(i)].x, 1.0f, static_cast<float>(image.width - 2));
                nv[static_cast<size_t>(i)].y =
                    std::clamp(nv[static_cast<size_t>(i)].y, 1.0f, static_cast<float>(image.height - 2));
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
