#pragma once

#include "contour/moore_neighborhood/moore_neighbor.hpp"

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace vision {

// Thin-plate spline RBF: φ(r) = r² ln r, solved with Eigen.
class ThinPlateSplineRBF {
public:
    double lambda = 1e-3;
    Eigen::VectorXd weights_x;
    Eigen::VectorXd weights_y;
    std::vector<Vec2> centers;

    static double kernel(double r) {
        if (r < 1e-12) {
            return 0.0;
        }
        return r * r * std::log(r);
    }

    bool fit(const std::vector<Vec2>& points, const std::vector<Vec2>& targets) {
        const int n = static_cast<int>(std::min(points.size(), targets.size()));
        if (n < 3) {
            return false;
        }
        centers.assign(points.begin(), points.begin() + n);
        Eigen::MatrixXd A(n, n);
        Eigen::VectorXd bx(n), by(n);
        for (int i = 0; i < n; ++i) {
            bx(i) = targets[static_cast<size_t>(i)].x;
            by(i) = targets[static_cast<size_t>(i)].y;
            for (int j = 0; j < n; ++j) {
                const double r = dist(points[static_cast<size_t>(i)], points[static_cast<size_t>(j)]);
                A(i, j) = kernel(r);
            }
            A(i, i) += lambda;
        }
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
        weights_x = qr.solve(bx);
        weights_y = qr.solve(by);
        return true;
    }

    bool fit_identity(const std::vector<Vec2>& contour, int subsample = 32) {
        if (contour.size() < 8) {
            return false;
        }
        std::vector<Vec2> pts;
        const int step = std::max(1, static_cast<int>(contour.size()) / subsample);
        for (size_t i = 0; i < contour.size(); i += static_cast<size_t>(step)) {
            pts.push_back(contour[i]);
        }
        return fit(pts, pts);
    }

    Vec2 evaluate(const Vec2& p) const {
        Vec2 out{0.0f, 0.0f};
        for (int i = 0; i < static_cast<int>(centers.size()); ++i) {
            const double k = kernel(dist(p, centers[static_cast<size_t>(i)]));
            out.x += static_cast<float>(weights_x(i) * k);
            out.y += static_cast<float>(weights_y(i) * k);
        }
        return out;
    }

    float reconstruction_rmse() const {
        if (centers.empty()) {
            return 1.0e9f;
        }
        double s = 0.0;
        for (size_t i = 0; i < centers.size(); ++i) {
            const Vec2 y = evaluate(centers[i]);
            s += dist2(y, centers[i]);
        }
        return static_cast<float>(std::sqrt(s / static_cast<double>(centers.size())));
    }
};

}  // namespace vision
