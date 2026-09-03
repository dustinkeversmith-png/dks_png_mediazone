#pragma once

#include "math/contour_compat.hpp"
#include "sdf/chamfer/chamfer.hpp"
#include "contour/marching_squares/marching_squares.hpp"
#include "segmentation/ccl/connected_components.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace contour {

// Chan–Vese minimal partitioning (Mumford–Shah without edges).
// Evolves a level-set φ to minimize region variance inside/outside the zero level,
// allowing implicit splits/merges without gradient edges or manual seeds.
class ChanVeseMinPartition {
public:
    float mu = 0.2f;       // length penalty
    float nu = 0.0f;       // area penalty
    float lambda1 = 1.0f;  // inside fidelity
    float lambda2 = 1.0f;  // outside fidelity
    float dt = 0.45f;
    float eps = 1.0f;
    int iterations = 80;
    int reinit_every = 10;
    int checker_period = 8;  // seed checkerboard half-period in pixels

    Field phi;
    float c1 = 0.0f;
    float c2 = 0.0f;

    struct Result {
        ImageBuffer partition;   // 255 = inside (φ<=0), 0 = outside
        Polyline contour;        // largest zero-level loop
        int n_regions = 0;       // CCL count on partition
        float energy = 0.0f;
    };

    Result segment(const ImageBuffer& image) {
        phi = make_field(image.width, image.height);
        // Soft center-bias init (still seed-free): negative toward image center.
        // Checkerboard alone often collapses to illumination partitions on DIS5K.
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                const float nx = (2.0f * x / std::max(1, image.width - 1)) - 1.0f;
                const float ny = (2.0f * y / std::max(1, image.height - 1)) - 1.0f;
                const float radial = std::sqrt(nx * nx + ny * ny);
                const int cx = (x / checker_period) & 1;
                const int cy = (y / checker_period) & 1;
                const float checker = ((cx ^ cy) == 0) ? -1.0f : 1.0f;
                phi.at(x, y) = 1.5f * (radial - 0.55f) + 0.35f * checker;
            }
        }
        reinitialize();

        std::vector<float> I(static_cast<size_t>(image.width * image.height));
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                I[static_cast<size_t>(y * image.width + x)] = image.gray(x, y) / 255.0f;
            }
        }

        float last_energy = 0.0f;
        for (int it = 0; it < iterations; ++it) {
            double c1n = 0, c1d = 0, c2n = 0, c2d = 0;
            for (int y = 0; y < image.height; ++y) {
                for (int x = 0; x < image.width; ++x) {
                    const float h = heaviside(-phi.at(x, y));
                    const float pix = I[static_cast<size_t>(y * image.width + x)];
                    c1n += pix * h;
                    c1d += h;
                    c2n += pix * (1.0f - h);
                    c2d += (1.0f - h);
                }
            }
            c1 = c1d > 1e-6 ? static_cast<float>(c1n / c1d) : 0.0f;
            c2 = c2d > 1e-6 ? static_cast<float>(c2n / c2d) : 1.0f;

            Field nphi = phi;
            double energy = 0.0;
            for (int y = 1; y < image.height - 1; ++y) {
                for (int x = 1; x < image.width - 1; ++x) {
                    const float px = phi.at(x, y);
                    const float dx = (phi.at(x + 1, y) - phi.at(x - 1, y)) * 0.5f;
                    const float dy = (phi.at(x, y + 1) - phi.at(x, y - 1)) * 0.5f;
                    const float dxx = phi.at(x + 1, y) - 2 * px + phi.at(x - 1, y);
                    const float dyy = phi.at(x, y + 1) - 2 * px + phi.at(x, y - 1);
                    const float dxy = (phi.at(x + 1, y + 1) - phi.at(x + 1, y - 1) -
                                       phi.at(x - 1, y + 1) + phi.at(x - 1, y - 1)) *
                                      0.25f;
                    const float den = std::pow(dx * dx + dy * dy + 1e-6f, 1.5f);
                    const float kappa = (dxx * dy * dy - 2 * dx * dy * dxy + dyy * dx * dx) / den;
                    const float pix = I[static_cast<size_t>(y * image.width + x)];
                    const float e1 = (pix - c1) * (pix - c1);
                    const float e2 = (pix - c2) * (pix - c2);
                    const float force = mu * kappa - nu - lambda1 * e1 + lambda2 * e2;
                    nphi.at(x, y) = px + dt * dirac(px) * force;

                    const float h = heaviside(-px);
                    energy += static_cast<double>(mu) * std::sqrt(dx * dx + dy * dy + 1e-6f);
                    energy += static_cast<double>(lambda1) * e1 * h;
                    energy += static_cast<double>(lambda2) * e2 * (1.0f - h);
                }
            }
            std::swap(phi, nphi);
            last_energy = static_cast<float>(energy);
            if ((it + 1) % reinit_every == 0) {
                reinitialize();
            }
        }

        Result r;
        r.partition = make_gray(image.width, image.height, 0);
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                r.partition.at(x, y) = phi.at(x, y) <= 0.0f ? 255 : 0;
            }
        }
        MarchingSquares ms;
        r.contour = ms.largest_closed(phi, 0.0f);
        const auto ccl = vision::ConnectedComponentLabeler::label(r.partition);
        r.n_regions = static_cast<int>(ccl.components.size());
        r.energy = last_energy;
        return r;
    }

private:
    static constexpr float kPi = 3.14159265358979323846f;

    float heaviside(float z) const {
        return 0.5f * (1.0f + (2.0f / kPi) * std::atan(z / eps));
    }
    float dirac(float z) const {
        return (eps / kPi) / (z * z + eps * eps);
    }
    void reinitialize() {
        ImageBuffer mask = make_gray(phi.width, phi.height, 0);
        for (int y = 0; y < phi.height; ++y) {
            for (int x = 0; x < phi.width; ++x) {
                mask.at(x, y) = phi.at(x, y) <= 0.0f ? 255 : 0;
            }
        }
        phi = ChamferSDF::from_mask(mask);
    }
};

}  // namespace contour
