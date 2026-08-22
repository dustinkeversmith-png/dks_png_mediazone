#pragma once

#include "contour/moore_neighborhood/moore_neighbor.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace vision {

class TurningFunction {
public:
    int bins = 128;

    std::vector<float> compute(const GrayImage& image) const {
        auto contour = MooreNeighborTracer::trace(image);
        return compute_from_points(contour.points);
    }

    std::vector<float> compute_from_points(const std::vector<Vec2>& pts) const {
        std::vector<float> theta(static_cast<size_t>(bins), 0.0f);
        if (pts.size() < 3) {
            return theta;
        }
        std::vector<float> angles;
        std::vector<float> acc_len;
        float total = 0.0f;
        angles.reserve(pts.size());
        acc_len.reserve(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            const Vec2& a = pts[i];
            const Vec2& b = pts[(i + 1) % pts.size()];
            const float seg = dist(a, b);
            if (seg < 1e-5f) {
                continue;
            }
            total += seg;
            acc_len.push_back(total);
            angles.push_back(std::atan2(b.y - a.y, b.x - a.x));
        }
        if (angles.empty() || total <= 1e-6f) {
            return theta;
        }
        // unwrap
        for (size_t i = 1; i < angles.size(); ++i) {
            while (angles[i] - angles[i - 1] > kPi) {
                angles[i] -= 2.0f * kPi;
            }
            while (angles[i] - angles[i - 1] < -kPi) {
                angles[i] += 2.0f * kPi;
            }
        }
        const float origin = angles.front();
        size_t j = 0;
        for (int i = 0; i < bins; ++i) {
            const float s = (static_cast<float>(i) + 0.5f) / static_cast<float>(bins) * total;
            while (j + 1 < acc_len.size() && acc_len[j] < s) {
                ++j;
            }
            theta[static_cast<size_t>(i)] = angles[std::min(j, angles.size() - 1)] - origin;
        }
        return theta;
    }

    static float circular_l2(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) {
            return 1.0e9f;
        }
        const int n = static_cast<int>(a.size());
        float best = std::numeric_limits<float>::max();
        for (int shift = 0; shift < n; shift += std::max(1, n / 32)) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) {
                const double d = static_cast<double>(a[static_cast<size_t>(i)] -
                                                     b[static_cast<size_t>((i + shift) % n)]);
                s += d * d;
            }
            best = std::min(best, static_cast<float>(std::sqrt(s / n)));
        }
        return best;
    }
};

}  // namespace vision
