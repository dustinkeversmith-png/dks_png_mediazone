#pragma once

#include "math/contour_compat.hpp"
#include "sdf/chamfer/chamfer.hpp"
#include "contour/marching_squares/marching_squares.hpp"
#include "segmentation/ccl/connected_components.hpp"
#include "filters/bilateral/bilateral.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace contour {

// Lightweight CLAHE-style tile equalization for Chan–Vese prep.
inline ImageBuffer clahe_gray(const ImageBuffer& src, int tile = 8, float clip = 2.5f) {
    ImageBuffer out = make_gray(src.width, src.height, 0);
    if (src.empty()) {
        return out;
    }
    const int tw = std::max(8, src.width / tile);
    const int th = std::max(8, src.height / tile);
    for (int ty = 0; ty < src.height; ty += th) {
        for (int tx = 0; tx < src.width; tx += tw) {
            const int x1 = std::min(src.width, tx + tw);
            const int y1 = std::min(src.height, ty + th);
            int hist[256] = {};
            int n = 0;
            for (int y = ty; y < y1; ++y) {
                for (int x = tx; x < x1; ++x) {
                    ++hist[static_cast<int>(src.gray(x, y))];
                    ++n;
                }
            }
            if (n <= 0) {
                continue;
            }
            const int clip_limit = std::max(1, static_cast<int>(clip * n / 256.0f));
            int clipped = 0;
            for (int i = 0; i < 256; ++i) {
                if (hist[i] > clip_limit) {
                    clipped += hist[i] - clip_limit;
                    hist[i] = clip_limit;
                }
            }
            const int redist = clipped / 256;
            for (int i = 0; i < 256; ++i) {
                hist[i] += redist;
            }
            int cdf[256];
            cdf[0] = hist[0];
            for (int i = 1; i < 256; ++i) {
                cdf[i] = cdf[i - 1] + hist[i];
            }
            const int cdf_min = [&]() {
                for (int i = 0; i < 256; ++i) {
                    if (cdf[i] > 0) {
                        return cdf[i];
                    }
                }
                return 0;
            }();
            const float den = std::max(1, n - cdf_min);
            for (int y = ty; y < y1; ++y) {
                for (int x = tx; x < x1; ++x) {
                    const int v = static_cast<int>(src.gray(x, y));
                    const float eq = (cdf[v] - cdf_min) * 255.0f / den;
                    out.at(x, y) = static_cast<uint8_t>(std::clamp(eq, 0.0f, 255.0f));
                }
            }
        }
    }
    return out;
}

// Chan–Vese minimal partitioning (Mumford–Shah without edges).
class ChanVeseMinPartition {
public:
    float mu = 0.02f;       // low length penalty (avoid Manhattan lock / refused legs)
    float nu = 0.0f;
    float lambda1 = 1.0f;   // inside
    float lambda2 = 2.0f;   // outside (push outward)
    float dt = 0.5f;
    float eps = 1.0f;
    int iterations = 100;
    int reinit_every = 12;
    int bubble_period = 10;  // multi-bubble grid period
    float margin = 0.08f;    // large outer init inset

    Field phi;
    float c1 = 0.0f;
    float c2 = 0.0f;

    struct Result {
        ImageBuffer partition;
        Polyline contour;
        int n_regions = 0;
        float energy = 0.0f;
    };

    Result segment(const ImageBuffer& image_in) {
        // Prep: bilateral denoise + CLAHE so texture doesn't warp regional means.
        BilateralFilter bilat;
        bilat.radius = 2;
        bilat.sigma_s = 1.5f;
        bilat.sigma_r = 24.0f;
        ImageBuffer image = bilat.apply(image_in);
        image = clahe_gray(image, 8, 2.0f);

        phi = make_field(image.width, image.height);
        // Large outer rectangle (inside=negative) + multi-bubble grid → expand, don't collapse.
        const float x0 = image.width * margin;
        const float y0 = image.height * margin;
        const float x1 = image.width * (1.0f - margin);
        const float y1 = image.height * (1.0f - margin);
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                const bool in_box = x >= x0 && x < x1 && y >= y0 && y < y1;
                const int cx = (x / bubble_period) & 1;
                const int cy = (y / bubble_period) & 1;
                const float bubble = ((cx ^ cy) == 0) ? -1.2f : 1.2f;
                phi.at(x, y) = in_box ? (-2.0f + 0.25f * bubble) : 2.5f;
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
                    const float den = std::pow(dx * dx + dy * dy + eps * eps, 1.5f);
                    const float kappa = (dxx * dy * dy - 2 * dx * dy * dxy + dyy * dx * dx) / den;
                    const float pix = I[static_cast<size_t>(y * image.width + x)];
                    const float e1 = (pix - c1) * (pix - c1);
                    const float e2 = (pix - c2) * (pix - c2);
                    const float force = mu * kappa - nu - lambda1 * e1 + lambda2 * e2;
                    nphi.at(x, y) = px + dt * dirac(px) * force;

                    const float h = heaviside(-px);
                    energy += static_cast<double>(mu) * std::sqrt(dx * dx + dy * dy + eps * eps);
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
