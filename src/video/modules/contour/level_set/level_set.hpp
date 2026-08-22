#pragma once

#include "math/contour_compat.hpp"
#include "sdf/chamfer/chamfer.hpp"
#include "contour/marching_squares/marching_squares.hpp"
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace contour {

class ChanVeseLevelSet {
public:
    float mu = 0.15f;      // length penalty
    float nu = 0.0f;       // area penalty
    float lambda1 = 1.0f;
    float lambda2 = 1.0f;
    float dt = 0.4f;
    float eps = 1.2f;
    int iterations = 50;
    int reinit_every = 10;

    Field phi;

    Polyline segment(const ImageBuffer& image, const Rect& init_box) {
        phi = make_field(image.width, image.height);
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                const bool inside = x >= init_box.x && x < init_box.x1() && y >= init_box.y && y < init_box.y1();
                phi.at(x, y) = inside ? -2.0f : 2.0f;
            }
        }
        reinitialize();
        std::vector<float> I(static_cast<size_t>(image.width * image.height));
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                I[static_cast<size_t>(y * image.width + x)] = image.gray(x, y) / 255.0f;
            }
        }
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
            const float c1 = c1d > 1e-6 ? static_cast<float>(c1n / c1d) : 0.0f;
            const float c2 = c2d > 1e-6 ? static_cast<float>(c2n / c2d) : 1.0f;

            Field nphi = phi;
            for (int y = 1; y < image.height - 1; ++y) {
                for (int x = 1; x < image.width - 1; ++x) {
                    const float px = phi.at(x, y);
                    const float dx = (phi.at(x + 1, y) - phi.at(x - 1, y)) * 0.5f;
                    const float dy = (phi.at(x, y + 1) - phi.at(x, y - 1)) * 0.5f;
                    const float dxx = phi.at(x + 1, y) - 2 * px + phi.at(x - 1, y);
                    const float dyy = phi.at(x, y + 1) - 2 * px + phi.at(x, y - 1);
                    const float dxy = (phi.at(x + 1, y + 1) - phi.at(x + 1, y - 1) -
                                       phi.at(x - 1, y + 1) + phi.at(x - 1, y - 1)) * 0.25f;
                    const float den = std::pow(dx * dx + dy * dy + 1e-6f, 1.5f);
                    const float kappa = (dxx * dy * dy - 2 * dx * dy * dxy + dyy * dx * dx) / den;
                    const float pix = I[static_cast<size_t>(y * image.width + x)];
                    const float force = mu * kappa - nu - lambda1 * (pix - c1) * (pix - c1) +
                                        lambda2 * (pix - c2) * (pix - c2);
                    nphi.at(x, y) = px + dt * dirac(px) * force;
                }
            }
            std::swap(phi, nphi);
            if ((it + 1) % reinit_every == 0) {
                reinitialize();
            }
        }
        MarchingSquares ms;
        return ms.largest_closed(phi, 0.0f);
    }

private:
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
