#pragma once

#include "../../boundary_tracing/moore_neighbor.hpp"

#include <fftw3.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace vision {

class FourierDescriptors {
public:
    int sample_count = 64;
    int keep_bins = 16;

    std::vector<float> compute(const GrayImage& image) const {
        auto contour = MooreNeighborTracer::trace(image);
        if (contour.points.size() < 8) {
            return std::vector<float>(static_cast<size_t>(keep_bins), 0.0f);
        }
        auto samples = MooreNeighborTracer::resample(contour.points, sample_count);
        return compute_from_points(samples);
    }

    std::vector<float> compute_from_points(const std::vector<Vec2>& samples) const {
        const int n = static_cast<int>(samples.size());
        std::vector<float> desc(static_cast<size_t>(keep_bins), 0.0f);
        if (n < 8) {
            return desc;
        }

        fftwf_complex* in = fftwf_alloc_complex(static_cast<size_t>(n));
        fftwf_complex* out = fftwf_alloc_complex(static_cast<size_t>(n));
        if (in == nullptr || out == nullptr) {
            fftwf_free(in);
            fftwf_free(out);
            return desc;
        }

        float mx = 0.0f, my = 0.0f;
        for (const auto& p : samples) {
            mx += p.x;
            my += p.y;
        }
        mx /= static_cast<float>(n);
        my /= static_cast<float>(n);
        for (int i = 0; i < n; ++i) {
            in[i][0] = samples[static_cast<size_t>(i)].x - mx;
            in[i][1] = samples[static_cast<size_t>(i)].y - my;
        }

        fftwf_plan plan = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(plan);

        // Drop DC (translation). Normalize by |F1| for scale. Magnitudes → rotation invariance.
        const float f1 = std::hypot(out[1][0], out[1][1]);
        const float norm = f1 > 1e-6f ? f1 : 1.0f;
        const int bins = std::min(keep_bins, n - 1);
        for (int k = 0; k < bins; ++k) {
            const int idx = k + 1;
            desc[static_cast<size_t>(k)] = std::hypot(out[idx][0], out[idx][1]) / norm;
        }

        fftwf_destroy_plan(plan);
        fftwf_free(in);
        fftwf_free(out);
        return desc;
    }

    static float l2(const std::vector<float>& a, const std::vector<float>& b) {
        const size_t n = std::min(a.size(), b.size());
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(a[i] - b[i]);
            s += d * d;
        }
        return static_cast<float>(std::sqrt(s));
    }

    static float dot(const std::vector<float>& a, const std::vector<float>& b) {
        const size_t n = std::min(a.size(), b.size());
        double s = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (size_t i = 0; i < n; ++i) {
            s += static_cast<double>(a[i]) * b[i];
            na += static_cast<double>(a[i]) * a[i];
            nb += static_cast<double>(b[i]) * b[i];
        }
        const double denom = std::sqrt(na * nb);
        return denom > 1e-12 ? static_cast<float>(s / denom) : 0.0f;
    }
};

}  // namespace vision
