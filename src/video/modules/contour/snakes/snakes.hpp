#pragma once

#include "math/contour_compat.hpp"
#include "math/contour_metrics.hpp"
#include "segmentation/bbox_auto/bbox_auto.hpp"
#include "filters/bilateral/bilateral.hpp"
#include "filters/edge/canny/canny.hpp"
#include "filters/gvf/gvh.hpp"
#include "filters/lab_color/lab_color_space.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace contour {

class SnakeActiveContour {
public:
    float alpha = 0.001f;
    float beta = 0.01f;
    float gamma = 2.4f;
    float kappa = 0.35f;  // region balloon magnitude; sign from Lab FG/BG
    float dt = 0.22f;
    int iterations = 140;
    int n_points = 192;

    Polyline evolve(const ImageBuffer& image, const Rect& bbox) const {
        BilateralFilter bf;
        bf.radius = 2;
        const ImageBuffer smooth = bf.apply(image);
        Canny canny;
        canny.low = 0.10f;
        canny.high = 0.28f;
        const ImageBuffer edges = canny.detect(smooth);
        GradientVectorFlow gvf;
        gvf.iterations = 64;
        gvf.mu = 0.2f;
        gvf.dt = 0.15f;
        gvf.compute(edges, true);

        Lab mu_bg{}, mu_fg{};
        int nbg = 0, nfg = 0;
        const Rect inner = {bbox.x + bbox.w * 0.22f, bbox.y + bbox.h * 0.22f, bbox.w * 0.56f,
                            bbox.h * 0.56f};
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                const Lab p = LabColor::at(image, x, y);
                const bool in_box = x >= bbox.x && x < bbox.x1() && y >= bbox.y && y < bbox.y1();
                const bool in_inner = x >= inner.x && x < inner.x1() && y >= inner.y && y < inner.y1();
                if (!in_box) {
                    mu_bg.L += p.L;
                    mu_bg.a += p.a;
                    mu_bg.b += p.b;
                    ++nbg;
                } else if (in_inner) {
                    mu_fg.L += p.L;
                    mu_fg.a += p.a;
                    mu_fg.b += p.b;
                    ++nfg;
                }
            }
        }
        if (nbg > 0) {
            mu_bg.L /= nbg;
            mu_bg.a /= nbg;
            mu_bg.b /= nbg;
        }
        if (nfg > 0) {
            mu_fg.L /= nfg;
            mu_fg.a /= nfg;
            mu_fg.b /= nfg;
        }

        // Start on an inset bbox (touches object extrema) and contract onto GVF edges.
        const Rect init = {bbox.x + bbox.w * 0.02f, bbox.y + bbox.h * 0.02f, bbox.w * 0.96f,
                           bbox.h * 0.96f};
        std::vector<Vec2> hull = {
            {init.x, init.y},
            {init.x1(), init.y},
            {init.x1(), init.y1()},
            {init.x, init.y1()},
        };
        std::vector<Vec2> v(static_cast<size_t>(n_points));
        float perim0 = 0;
        std::vector<float> acc0(5, 0);
        for (size_t i = 0; i < 4; ++i) {
            perim0 += dist(hull[i], hull[(i + 1) % 4]);
            acc0[i + 1] = perim0;
        }
        for (int i = 0; i < n_points; ++i) {
            float t = (static_cast<float>(i) / static_cast<float>(n_points)) * perim0;
            size_t e = 1;
            while (e < acc0.size() && acc0[e] < t) {
                ++e;
            }
            const Vec2 a = hull[(e - 1) % 4];
            const Vec2 b = hull[e % 4];
            const float seg = std::max(1e-4f, acc0[e] - acc0[e - 1]);
            v[static_cast<size_t>(i)] = a + (b - a) * ((t - acc0[e - 1]) / seg);
        }

        auto wrap = [&](int i) {
            const int n = static_cast<int>(v.size());
            return (i % n + n) % n;
        };
        auto resample = [&]() {
            float perim = 0;
            std::vector<float> acc(v.size() + 1, 0);
            for (size_t i = 0; i < v.size(); ++i) {
                perim += dist(v[i], v[(i + 1) % v.size()]);
                acc[i + 1] = perim;
            }
            if (perim < 8.0f) {
                return;
            }
            std::vector<Vec2> nv(v.size());
            for (size_t i = 0; i < v.size(); ++i) {
                float t = (static_cast<float>(i) / static_cast<float>(v.size())) * perim;
                size_t e = 1;
                while (e < acc.size() && acc[e] < t) {
                    ++e;
                }
                const Vec2 a = v[(e - 1) % v.size()];
                const Vec2 b = v[e % v.size()];
                const float seg = std::max(1e-4f, acc[e] - acc[e - 1]);
                const float u = (t - acc[e - 1]) / seg;
                nv[i] = a + (b - a) * u;
            }
            v.swap(nv);
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
                const Vec2 tang = vp1 - vm1;
                const Vec2 outward = normalize({tang.y, -tang.x});  // CCW ellipse → outward
                const Vec2 ext = gvf.sample(vc.x, vc.y);
                const float g = length(ext);
                const int ix = std::clamp(static_cast<int>(std::lround(vc.x)), 0, image.width - 1);
                const int iy = std::clamp(static_cast<int>(std::lround(vc.y)), 0, image.height - 1);
                const Lab p = LabColor::at(image, ix, iy);
                const float dfg = LabColor::delta2(p, mu_fg);
                const float dbg = LabColor::delta2(p, mu_bg);
                // Positive when the sample looks like FG → expand; BG → contract.
                const float region = std::tanh((dbg - dfg) / 400.0f);
                const float balloon = kappa * region / (1.0f + 2.0f * g);
                Vec2 f = tension * alpha - rigid * beta + ext * gamma + outward * balloon;
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
            if ((it + 1) % 12 == 0) {
                resample();
            }
        }
        Polyline p;
        p.points = std::move(v);
        p.closed = true;
        return p;
    }
};

}  // namespace contour
