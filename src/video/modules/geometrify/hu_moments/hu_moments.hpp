#pragma once

#include "../../math/vision_types.hpp"

#include <array>
#include <cmath>
#include <algorithm>

namespace vision {

class HuMoments {
public:
    struct Result {
        std::array<double, 7> hu{};
        std::array<double, 4> flusser{};
        double eta20 = 0, eta02 = 0, eta11 = 0, eta30 = 0, eta03 = 0, eta21 = 0, eta12 = 0;
        double cx = 0, cy = 0, area = 0;
    };

    static Result compute(const GrayImage& image, uint8_t thr = 128) {
        Result r;
        double m00 = 0, m10 = 0, m01 = 0;
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                if (!image.fg(x, y, thr)) {
                    continue;
                }
                m00 += 1.0;
                m10 += x;
                m01 += y;
            }
        }
        r.area = m00;
        if (m00 < 1.0) {
            return r;
        }
        r.cx = m10 / m00;
        r.cy = m01 / m00;

        double mu20 = 0, mu02 = 0, mu11 = 0, mu30 = 0, mu03 = 0, mu21 = 0, mu12 = 0;
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                if (!image.fg(x, y, thr)) {
                    continue;
                }
                const double dx = x - r.cx;
                const double dy = y - r.cy;
                mu20 += dx * dx;
                mu02 += dy * dy;
                mu11 += dx * dy;
                mu30 += dx * dx * dx;
                mu03 += dy * dy * dy;
                mu21 += dx * dx * dy;
                mu12 += dx * dy * dy;
            }
        }

        const double n20 = 2.0, n02 = 2.0, n11 = 2.0;
        auto eta = [&](double mu, double p, double q) {
            return mu / std::pow(m00, 1.0 + (p + q) / 2.0);
        };
        r.eta20 = eta(mu20, 2, 0);
        r.eta02 = eta(mu02, 0, 2);
        r.eta11 = eta(mu11, 1, 1);
        r.eta30 = eta(mu30, 3, 0);
        r.eta03 = eta(mu03, 0, 3);
        r.eta21 = eta(mu21, 2, 1);
        r.eta12 = eta(mu12, 1, 2);
        (void)n20;
        (void)n02;
        (void)n11;

        const double n20v = r.eta20, n02v = r.eta02, n11v = r.eta11;
        const double n30v = r.eta30, n03v = r.eta03, n21v = r.eta21, n12v = r.eta12;

        r.hu[0] = n20v + n02v;
        r.hu[1] = (n20v - n02v) * (n20v - n02v) + 4.0 * n11v * n11v;
        r.hu[2] = (n30v - 3.0 * n12v) * (n30v - 3.0 * n12v) + (3.0 * n21v - n03v) * (3.0 * n21v - n03v);
        r.hu[3] = (n30v + n12v) * (n30v + n12v) + (n21v + n03v) * (n21v + n03v);
        r.hu[4] = (n30v - 3.0 * n12v) * (n30v + n12v) *
                      ((n30v + n12v) * (n30v + n12v) - 3.0 * (n21v + n03v) * (n21v + n03v)) +
                  (3.0 * n21v - n03v) * (n21v + n03v) *
                      (3.0 * (n30v + n12v) * (n30v + n12v) - (n21v + n03v) * (n21v + n03v));
        r.hu[5] = (n20v - n02v) * ((n30v + n12v) * (n30v + n12v) - (n21v + n03v) * (n21v + n03v)) +
                  4.0 * n11v * (n30v + n12v) * (n21v + n03v);
        r.hu[6] = (3.0 * n21v - n03v) * (n30v + n12v) *
                      ((n30v + n12v) * (n30v + n12v) - 3.0 * (n21v + n03v) * (n21v + n03v)) -
                  (n30v - 3.0 * n12v) * (n21v + n03v) *
                      (3.0 * (n30v + n12v) * (n30v + n12v) - (n21v + n03v) * (n21v + n03v));

        // Flusser affine invariants (first four from central moments)
        r.flusser[0] = (n20v * n02v - n11v * n11v);
        r.flusser[1] = (n30v * n30v * n03v * n03v - 6.0 * n30v * n21v * n12v * n03v +
                        4.0 * n30v * n12v * n12v * n12v + 4.0 * n21v * n21v * n21v * n03v -
                        3.0 * n21v * n21v * n12v * n12v);
        r.flusser[2] = n20v * (n21v * n03v - n12v * n12v) - n11v * (n30v * n03v - n21v * n12v) +
                       n02v * (n30v * n12v - n21v * n21v);
        r.flusser[3] = (n20v * n20v * n20v * n03v * n03v - 6.0 * n20v * n20v * n11v * n12v * n03v -
                        6.0 * n20v * n20v * n02v * n21v * n03v + 9.0 * n20v * n20v * n02v * n12v * n12v +
                        12.0 * n20v * n11v * n11v * n21v * n03v + 6.0 * n20v * n11v * n02v * n30v * n03v -
                        18.0 * n20v * n11v * n02v * n21v * n12v - 8.0 * n11v * n11v * n11v * n30v * n03v -
                        6.0 * n20v * n02v * n02v * n30v * n12v + 9.0 * n20v * n02v * n02v * n21v * n21v +
                        12.0 * n11v * n11v * n02v * n30v * n12v - 6.0 * n11v * n02v * n02v * n30v * n21v +
                        n02v * n02v * n02v * n30v * n30v);

        return r;
    }

    static std::vector<float> log_abs_vector(const Result& r) {
        std::vector<float> v(7);
        for (int i = 0; i < 7; ++i) {
            v[static_cast<size_t>(i)] = static_cast<float>(std::log(1.0 + std::fabs(r.hu[static_cast<size_t>(i)])));
        }
        return v;
    }

    static float l2(const std::vector<float>& a, const std::vector<float>& b) {
        double s = 0.0;
        const size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(a[i] - b[i]);
            s += d * d;
        }
        return static_cast<float>(std::sqrt(s));
    }
};

}  // namespace vision
