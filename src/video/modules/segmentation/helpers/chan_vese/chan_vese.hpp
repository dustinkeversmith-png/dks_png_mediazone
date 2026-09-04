#pragma once

#include "math/contour_compat.hpp"
#include "sdf/chamfer/chamfer.hpp"
#include "contour/marching_squares/marching_squares.hpp"
#include "segmentation/helpers/ccl/connected_components.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace contour {

// Lightweight CLAHE-style tile equalization.
// The LUT is built from the clipped histogram and blended bilinearly between
// tile centres; tiles too small to hold a meaningful histogram fall back to
// identity so flat regions are not crushed to black.
inline ImageBuffer clahe_gray(const ImageBuffer& src, int tiles = 8, float clip = 2.5f) {
    ImageBuffer out = make_gray(src.width, src.height, 0);
    if (src.empty()) {
        return out;
    }
    const int nx = std::max(1, std::min(tiles, src.width / 16));
    const int ny = std::max(1, std::min(tiles, src.height / 16));
    std::vector<std::array<uint8_t, 256>> luts(static_cast<size_t>(nx * ny));
    for (int ty = 0; ty < ny; ++ty) {
        for (int tx = 0; tx < nx; ++tx) {
            auto& lut = luts[static_cast<size_t>(ty * nx + tx)];
            const int x0 = tx * src.width / nx;
            const int x1 = (tx + 1) * src.width / nx;
            const int y0 = ty * src.height / ny;
            const int y1 = (ty + 1) * src.height / ny;
            int hist[256] = {};
            int n = 0;
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    ++hist[static_cast<int>(src.gray(x, y))];
                    ++n;
                }
            }
            if (n < 256) {
                for (int v = 0; v < 256; ++v) {
                    lut[v] = static_cast<uint8_t>(v);
                }
                continue;
            }
            const int limit = std::max(1, static_cast<int>(clip * n / 256.0f));
            int excess = 0;
            for (int v = 0; v < 256; ++v) {
                if (hist[v] > limit) {
                    excess += hist[v] - limit;
                    hist[v] = limit;
                }
            }
            // Spread the clipped mass evenly, keeping the remainder: integer
            // division alone loses nearly all of it on small tiles, which
            // flattens the LUT and crushes the tile to black.
            const int share = excess / 256;
            const int rest = excess - share * 256;
            for (int v = 0; v < 256; ++v) {
                hist[v] += share + (v < rest ? 1 : 0);
            }
            int cdf = 0;
            for (int v = 0; v < 256; ++v) {
                cdf += hist[v];
                lut[v] = static_cast<uint8_t>(std::clamp(cdf * 255 / n, 0, 255));
            }
        }
    }
    const float tw = static_cast<float>(src.width) / static_cast<float>(nx);
    const float th = static_cast<float>(src.height) / static_cast<float>(ny);
    for (int y = 0; y < src.height; ++y) {
        const float fy = (static_cast<float>(y) + 0.5f) / th - 0.5f;
        const int ty0 = std::clamp(static_cast<int>(std::floor(fy)), 0, ny - 1);
        const int ty1 = std::clamp(ty0 + 1, 0, ny - 1);
        const float wy = std::clamp(fy - static_cast<float>(ty0), 0.0f, 1.0f);
        for (int x = 0; x < src.width; ++x) {
            const float fx = (static_cast<float>(x) + 0.5f) / tw - 0.5f;
            const int tx0 = std::clamp(static_cast<int>(std::floor(fx)), 0, nx - 1);
            const int tx1 = std::clamp(tx0 + 1, 0, nx - 1);
            const float wx = std::clamp(fx - static_cast<float>(tx0), 0.0f, 1.0f);
            const int v = static_cast<int>(src.gray(x, y));
            const float a = luts[static_cast<size_t>(ty0 * nx + tx0)][v];
            const float b = luts[static_cast<size_t>(ty0 * nx + tx1)][v];
            const float c = luts[static_cast<size_t>(ty1 * nx + tx0)][v];
            const float d = luts[static_cast<size_t>(ty1 * nx + tx1)][v];
            const float top = a + (b - a) * wx;
            const float bot = c + (d - c) * wx;
            out.at(x, y) = static_cast<uint8_t>(std::clamp(top + (bot - top) * wy, 0.0f, 255.0f));
        }
    }
    return out;
}

// Chan–Vese minimal partitioning (piecewise-constant Mumford–Shah, no edges).
//
// `n_levelsets` level set functions carve the frame into 1 << n_levelsets
// phases (the Vese–Chan multiphase framework), so a single run separates
// several appearance classes instead of one foreground/background cut. The data
// term is vector-valued (Chan–Sandberg–Vese) so objects that differ in colour
// but not in luma still split. Each phase is then cut into connected
// components and neighbouring components with matching means are re-merged, so
// every distinct object in the frame ends up with its own instance label.
//
// Numerics follow the original paper: a *global* regularized Dirac keeps every
// pixel mobile (that is what lets new contours appear in the middle of the
// frame), φ is left un-reinitialized and merely clamped, and each step is
// normalized so the largest displacement is exactly `dt` whatever the contrast.
class ChanVeseMinPartition {
public:
    float mu = 0.12f;        // length penalty (the data term is variance-normalized)
    float nu = 0.0f;         // area penalty
    float lambda = 1.0f;     // data fidelity
    float dt = 0.45f;        // max |Δφ| per iteration, in φ units
    float eps = 1.2f;        // Heaviside / Dirac width, in pixels
    int iterations = 220;
    int scales = 3;          // coarse-to-fine pyramid levels (1 = single scale)
    int n_levelsets = 2;     // 2 → 4 phases, 3 → 8 phases
    int init_period = 11;    // checkerboard cell size of the first level set
    float phi_clamp = 3.0f;  // keeps the Dirac from starving far from a contour
    // A bilateral prefilter steadies the region means but erases 1–2 px
    // filaments (chain links, spectacle wire), which are exactly the objects
    // hardest to recover. Off by default; raise it for noisy footage.
    int denoise_passes = 0;
    // Edge indicator on the length term (geodesic active contour): boundary
    // costs less where the image gradient is strong, so the minimal partition
    // snaps to object outlines instead of settling into smooth blobs.
    // 0 disables and leaves the plain Chan–Vese curvature flow.
    float edge_beta = 0.20f;

    // Post-processing: phases → object instances.
    float merge_tol = 0.03f;       // merge phases whose means are this close (0..1)
    float rag_tol = 0.05f;         // merge touching instances whose means are this close
    float min_area_frac = 0.002f;  // absorb instances smaller than this share of the frame
    int min_area_px = 16;
    // Boundary connectivity (Zhu et al. 2014): frame-margin contact over
    // sqrt(area). Backdrop spreads along the margin and scores high; a compact
    // object scores near zero even when it runs off the edge of the frame.
    float bnd_con_bg = 2.0f;
    // Contact is measured against a margin band, not the outermost pixel ring:
    // a letterbox matte or a dark vignette otherwise owns the whole ring and
    // leaves the real backdrop looking like an interior object.
    float border_band_frac = 0.005f;
    // Backdrop that splits into pieces (a wall plus its shadow) leaves pieces
    // with low border contact. A piece that touches the border at all and looks
    // like the backdrop is backdrop too. 0 disables.
    float bg_color_tol = 0.06f;

    Field phi;               // == phis.front(), kept for existing callers
    std::vector<Field> phis;
    float c1 = 0.0f;         // mean luma of the object instances
    float c2 = 0.0f;         // mean luma of the background
    int iterations_run = 0;

    struct Object {
        int label = 0;        // 1..K, matches `labels` / `label_image`
        int phase = 0;
        int area = 0;
        math::Rect bbox;
        std::array<float, 3> mean{{0.0f, 0.0f, 0.0f}};
        int border_px = 0;          // pixels this instance owns in the frame margin band
        float bnd_con = 0.0f;       // border_px / sqrt(area)
        bool background = false;
    };

    struct Result {
        ImageBuffer partition;    // 255 = object, 0 = background
        ImageBuffer label_image;  // instance ids, spread out for 8-bit viewing
        ImageBuffer phase_image;  // phase ids, spread out for 8-bit viewing
        std::vector<int> labels;  // per-pixel instance id (1..K)
        std::vector<Object> objects;
        std::vector<Polyline> contours;  // one loop per foreground object
        Polyline contour;                // loop of the largest foreground object
        int n_regions = 0;   // instances including background
        int n_objects = 0;   // foreground instances
        int n_phases = 0;    // phases surviving the mean merge
        float energy = 0.0f; // Mumford–Shah energy per pixel
    };

    Result segment(const ImageBuffer& image_in) {
        Result r;
        const int w = image_in.width;
        const int h = image_in.height;
        if (image_in.empty()) {
            return r;
        }
        const int nc = std::clamp(image_in.channels, 1, 3);
        const size_t n_px = static_cast<size_t>(w) * static_cast<size_t>(h);

        std::vector<float> I(n_px * static_cast<size_t>(nc));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < nc; ++c) {
                    I[(static_cast<size_t>(y) * w + x) * nc + c] = image_in.at(x, y, c) / 255.0f;
                }
            }
        }
        for (int p = 0; p < denoise_passes; ++p) {
            I = bilateral(I, w, h, nc);
        }

        std::vector<int> phase(n_px, 0);
        const float energy = evolve(I, w, h, nc, phase);

        std::vector<int> cls;
        const int n_phases = merge_phases(I, phase, n_px, nc, cls);
        std::vector<int> inst;
        int n_labels = label_instances(cls, n_phases, w, h, inst);
        const int min_area =
            std::max(min_area_px, static_cast<int>(min_area_frac * static_cast<float>(n_px)));
        absorb_small(inst, n_labels, w, h, min_area);
        merge_adjacent(inst, n_labels, w, h, I, nc, rag_tol);

        r.objects = measure(inst, n_labels, cls, w, h, I, nc);
        r.n_objects = classify_background(r.objects, inst, w, h, nc);
        finish(r, inst, I, w, h, nc, n_labels, n_phases, energy);
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
    // Neumann boundary: clamp, rather than Field::at's front()-on-out-of-range.
    static float px(const Field& f, int x, int y) {
        return f.data[static_cast<size_t>(std::clamp(y, 0, f.height - 1)) * f.width +
                      static_cast<size_t>(std::clamp(x, 0, f.width - 1))];
    }
    float curvature(const Field& f, int x, int y) const {
        const float c = px(f, x, y);
        const float dx = 0.5f * (px(f, x + 1, y) - px(f, x - 1, y));
        const float dy = 0.5f * (px(f, x, y + 1) - px(f, x, y - 1));
        const float dxx = px(f, x + 1, y) - 2.0f * c + px(f, x - 1, y);
        const float dyy = px(f, x, y + 1) - 2.0f * c + px(f, x, y - 1);
        const float dxy = 0.25f * (px(f, x + 1, y + 1) - px(f, x + 1, y - 1) - px(f, x - 1, y + 1) +
                                   px(f, x - 1, y - 1));
        const float den = std::pow(dx * dx + dy * dy + 1e-6f, 1.5f);
        return (dxx * dy * dy - 2.0f * dx * dy * dxy + dyy * dx * dx) / den;
    }
    static float mean_distance(const std::array<float, 3>& a, const std::array<float, 3>& b, int nc) {
        float s = 0.0f;
        for (int c = 0; c < nc; ++c) {
            const float d = a[static_cast<size_t>(c)] - b[static_cast<size_t>(c)];
            s += d * d;
        }
        return std::sqrt(s / static_cast<float>(nc));
    }
    static float channel_variance(const std::vector<float>& I, size_t n_px, int nc) {
        float total = 0.0f;
        for (int c = 0; c < nc; ++c) {
            double sum = 0.0;
            double sum2 = 0.0;
            for (size_t k = 0; k < n_px; ++k) {
                const double v = I[k * nc + c];
                sum += v;
                sum2 += v * v;
            }
            const double m = sum / static_cast<double>(n_px);
            total += static_cast<float>(sum2 / static_cast<double>(n_px) - m * m);
        }
        return total / static_cast<float>(nc);
    }
    // Colour bilateral: one range weight from the whole pixel vector, so the
    // channels stay aligned and edges between equal-luma colours survive.
    static std::vector<float> bilateral(const std::vector<float>& src, int w, int h, int nc,
                                        int radius = 2, float sigma_s = 1.8f,
                                        float sigma_r = 0.10f) {
        std::vector<float> out(src.size());
        const float inv_s = 1.0f / (2.0f * sigma_s * sigma_s);
        const float inv_r = 1.0f / (2.0f * sigma_r * sigma_r * static_cast<float>(nc));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t k = static_cast<size_t>(y) * w + x;
                float acc[3] = {0.0f, 0.0f, 0.0f};
                float wsum = 0.0f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int xx = std::clamp(x + dx, 0, w - 1);
                        const int yy = std::clamp(y + dy, 0, h - 1);
                        const size_t kk = static_cast<size_t>(yy) * w + xx;
                        float d2 = 0.0f;
                        for (int c = 0; c < nc; ++c) {
                            const float t = src[kk * nc + c] - src[k * nc + c];
                            d2 += t * t;
                        }
                        const float wgt =
                            std::exp(-static_cast<float>(dx * dx + dy * dy) * inv_s - d2 * inv_r);
                        for (int c = 0; c < nc; ++c) {
                            acc[c] += wgt * src[kk * nc + c];
                        }
                        wsum += wgt;
                    }
                }
                for (int c = 0; c < nc; ++c) {
                    out[k * nc + c] = acc[c] / std::max(wsum, 1e-6f);
                }
            }
        }
        return out;
    }

    // g = 1 / (1 + (|∇I| / beta)^2) over the vector-valued image, so a boundary
    // placed on an image edge costs almost nothing.
    Field edge_indicator(const std::vector<float>& I, int w, int h, int nc) const {
        Field g = make_field(w, h, 1.0f);
        if (edge_beta <= 0.0f) {
            return g;
        }
        const float inv_beta2 = 1.0f / (edge_beta * edge_beta);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float mag2 = 0.0f;
                for (int c = 0; c < nc; ++c) {
                    const size_t xp = (static_cast<size_t>(y) * w + std::min(w - 1, x + 1)) * nc + c;
                    const size_t xm = (static_cast<size_t>(y) * w + std::max(0, x - 1)) * nc + c;
                    const size_t yp = (static_cast<size_t>(std::min(h - 1, y + 1)) * w + x) * nc + c;
                    const size_t ym = (static_cast<size_t>(std::max(0, y - 1)) * w + x) * nc + c;
                    const float dx = 0.5f * (I[xp] - I[xm]);
                    const float dy = 0.5f * (I[yp] - I[ym]);
                    mag2 += dx * dx + dy * dy;
                }
                g.at(x, y) = 1.0f / (1.0f + mag2 / static_cast<float>(nc) * inv_beta2);
            }
        }
        return g;
    }

    // --- level set evolution ------------------------------------------------
    static std::vector<float> downsample2(const std::vector<float>& I, int w, int h, int nc, int& ow,
                                          int& oh) {
        ow = std::max(1, w / 2);
        oh = std::max(1, h / 2);
        std::vector<float> out(static_cast<size_t>(ow) * oh * nc, 0.0f);
        for (int y = 0; y < oh; ++y) {
            for (int x = 0; x < ow; ++x) {
                for (int c = 0; c < nc; ++c) {
                    float s = 0.0f;
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            const int sx = std::min(w - 1, 2 * x + dx);
                            const int sy = std::min(h - 1, 2 * y + dy);
                            s += I[(static_cast<size_t>(sy) * w + sx) * nc + c];
                        }
                    }
                    out[(static_cast<size_t>(y) * ow + x) * nc + c] = s * 0.25f;
                }
            }
        }
        return out;
    }
    static Field upsample_to(const Field& src, int w, int h) {
        Field out = make_field(w, h);
        const float fx = static_cast<float>(src.width) / static_cast<float>(w);
        const float fy = static_cast<float>(src.height) / static_cast<float>(h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                out.at(x, y) = src.sample((static_cast<float>(x) + 0.5f) * fx - 0.5f,
                                          (static_cast<float>(y) + 0.5f) * fy - 0.5f);
            }
        }
        return out;
    }

    void init_levelsets(int w, int h, int L) {
        phis.assign(static_cast<size_t>(L), make_field(w, h));
        for (int i = 0; i < L; ++i) {
            // Distinct periods and offsets per function, so every one of the
            // 2^L phases starts with a non-empty support; identical
            // checkerboards would leave the extra phases permanently empty.
            const float period =
                static_cast<float>(init_period) * std::pow(1.7f, static_cast<float>(i));
            const float off = period * 0.37f * static_cast<float>(i);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const float sx = std::sin(kPi * (static_cast<float>(x) + off) / period);
                    const float sy = std::sin(kPi * (static_cast<float>(y) + off) / period);
                    phis[static_cast<size_t>(i)].at(x, y) = sx * sy;
                }
            }
        }
    }

    // Coarse-to-fine: settle the big structures on a cheap small image, then
    // carry φ up as the initialization for the next level. Fewer local minima
    // than one pass at full size, and cheaper.
    float evolve(const std::vector<float>& I0, int w0, int h0, int nc, std::vector<int>& phase) {
        const int S = std::clamp(scales, 1, 4);
        std::vector<std::vector<float>> pyr{I0};
        std::vector<int> pw{w0};
        std::vector<int> ph{h0};
        for (int s = 1; s < S; ++s) {
            int ow = 0;
            int oh = 0;
            if (pw.back() < 32 || ph.back() < 32) {
                break;
            }
            pyr.push_back(downsample2(pyr.back(), pw.back(), ph.back(), nc, ow, oh));
            pw.push_back(ow);
            ph.push_back(oh);
        }
        const int L = std::clamp(n_levelsets, 1, 3);
        const int top = static_cast<int>(pyr.size()) - 1;
        init_levelsets(pw[static_cast<size_t>(top)], ph[static_cast<size_t>(top)], L);

        iterations_run = 0;
        float energy = 0.0f;
        for (int s = top; s >= 0; --s) {
            if (s != top) {
                for (auto& f : phis) {
                    f = upsample_to(f, pw[static_cast<size_t>(s)], ph[static_cast<size_t>(s)]);
                }
            }
            const int iters = s == top ? iterations : std::max(40, iterations / 2);
            energy = run_scale(pyr[static_cast<size_t>(s)], pw[static_cast<size_t>(s)],
                               ph[static_cast<size_t>(s)], nc, L, iters, phase);
        }
        phi = phis.front();
        return energy;
    }

    float run_scale(const std::vector<float>& I, int w, int h, int nc, int L, int iterations,
                    std::vector<int>& phase) {
        const size_t n_px = static_cast<size_t>(w) * static_cast<size_t>(h);
        const float var = std::max(1e-4f, channel_variance(I, n_px, nc));
        const int P = 1 << L;
        phase.assign(n_px, 0);

        const Field g = edge_indicator(I, w, h, nc);
        std::vector<float> H(n_px * static_cast<size_t>(L));
        std::vector<float> chi(n_px * static_cast<size_t>(P));
        std::vector<std::array<float, 3>> means(static_cast<size_t>(P));
        std::vector<float> e(static_cast<size_t>(P));
        std::vector<std::vector<float>> step(static_cast<size_t>(L), std::vector<float>(n_px, 0.0f));
        std::vector<int> prev_phase(n_px, -1);

        double last_energy = 0.0;
        int settled = 0;
        int it = 0;
        for (; it < iterations; ++it) {
            for (int i = 0; i < L; ++i) {
                const Field& f = phis[static_cast<size_t>(i)];
                for (size_t k = 0; k < n_px; ++k) {
                    H[k * L + i] = heaviside(f.data[k]);
                }
            }
            std::array<std::array<double, 3>, 8> num{};
            std::array<double, 8> den{};
            for (size_t k = 0; k < n_px; ++k) {
                for (int p = 0; p < P; ++p) {
                    float m = 1.0f;
                    for (int i = 0; i < L; ++i) {
                        const float hi = H[k * L + i];
                        m *= (p >> i) & 1 ? hi : 1.0f - hi;
                    }
                    chi[k * P + p] = m;
                    den[static_cast<size_t>(p)] += m;
                    for (int c = 0; c < nc; ++c) {
                        num[static_cast<size_t>(p)][static_cast<size_t>(c)] +=
                            static_cast<double>(m) * I[k * nc + c];
                    }
                }
            }
            for (int p = 0; p < P; ++p) {
                for (int c = 0; c < nc; ++c) {
                    means[static_cast<size_t>(p)][static_cast<size_t>(c)] =
                        den[static_cast<size_t>(p)] > 1e-6
                            ? static_cast<float>(num[static_cast<size_t>(p)][static_cast<size_t>(c)] /
                                                 den[static_cast<size_t>(p)])
                            : 0.5f;
                }
            }

            double energy = 0.0;
            float max_step = 0.0f;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const size_t k = static_cast<size_t>(y) * w + x;
                    for (int p = 0; p < P; ++p) {
                        float d = 0.0f;
                        for (int c = 0; c < nc; ++c) {
                            const float t =
                                I[k * nc + c] - means[static_cast<size_t>(p)][static_cast<size_t>(c)];
                            d += t * t;
                        }
                        e[static_cast<size_t>(p)] = d / (static_cast<float>(nc) * var);
                        energy += static_cast<double>(e[static_cast<size_t>(p)]) * chi[k * P + p];
                    }
                    for (int i = 0; i < L; ++i) {
                        // Descent for φ_i: every phase pulls with the sign of
                        // its i-th bit, weighted by the membership of the other
                        // level sets (Vese–Chan, generalized to L functions).
                        float data = 0.0f;
                        for (int p = 0; p < P; ++p) {
                            float wgt = 1.0f;
                            for (int j = 0; j < L; ++j) {
                                if (j == i) {
                                    continue;
                                }
                                const float hj = H[k * L + j];
                                wgt *= (p >> j) & 1 ? hj : 1.0f - hj;
                            }
                            data += ((p >> i) & 1 ? 1.0f : -1.0f) * e[static_cast<size_t>(p)] * wgt;
                        }
                        const Field& f = phis[static_cast<size_t>(i)];
                        const float gx = 0.5f * (px(f, x + 1, y) - px(f, x - 1, y));
                        const float gy = 0.5f * (px(f, x, y + 1) - px(f, x, y - 1));
                        const float mag = std::sqrt(gx * gx + gy * gy + 1e-6f);
                        // mu * div(g ∇φ/|∇φ|) = mu * (g·κ + ∇g·∇φ/|∇φ|)
                        const float gv = g.data[k];
                        const float dgx = 0.5f * (px(g, x + 1, y) - px(g, x - 1, y));
                        const float dgy = 0.5f * (px(g, x, y + 1) - px(g, x, y - 1));
                        const float length = gv * curvature(f, x, y) + (dgx * gx + dgy * gy) / mag;
                        const float force = mu * length - nu - lambda * data;
                        const float d_phi = dirac(f.data[k]);
                        step[static_cast<size_t>(i)][k] = d_phi * force;
                        max_step = std::max(max_step, std::fabs(step[static_cast<size_t>(i)][k]));
                        energy += static_cast<double>(mu) * gv * d_phi * mag;
                    }
                }
            }
            const float scale = dt / std::max(max_step, 1e-9f);
            for (int i = 0; i < L; ++i) {
                Field& f = phis[static_cast<size_t>(i)];
                for (size_t k = 0; k < n_px; ++k) {
                    f.data[k] = std::clamp(f.data[k] + scale * step[static_cast<size_t>(i)][k],
                                           -phi_clamp, phi_clamp);
                }
            }

            int flipped = 0;
            for (size_t k = 0; k < n_px; ++k) {
                int idx = 0;
                for (int i = 0; i < L; ++i) {
                    idx |= (phis[static_cast<size_t>(i)].data[k] > 0.0f ? 1 : 0) << i;
                }
                phase[k] = idx;
                if (idx != prev_phase[k]) {
                    ++flipped;
                }
            }
            prev_phase = phase;
            last_energy = energy / static_cast<double>(n_px);
            // Convergence is tested on the labelling, not on φ: the normalized
            // step holds |Δφ| at dt forever, so φ never stops moving.
            settled = flipped <= static_cast<int>(n_px / 2000) ? settled + 1 : 0;
            if (settled >= 8) {
                ++it;
                break;
            }
        }
        iterations_run += it;
        return static_cast<float>(last_energy);
    }

    // --- phases → instances --------------------------------------------------
    // Phases whose fitted means agree describe the same material; collapse them
    // before instancing so one object split across two phases stays one object.
    int merge_phases(const std::vector<float>& I, const std::vector<int>& phase, size_t n_px, int nc,
                     std::vector<int>& cls) const {
        const int P = 1 << std::clamp(n_levelsets, 1, 3);
        std::array<std::array<double, 3>, 8> num{};
        std::array<double, 8> den{};
        for (size_t k = 0; k < n_px; ++k) {
            const size_t p = static_cast<size_t>(phase[k]);
            den[p] += 1.0;
            for (int c = 0; c < nc; ++c) {
                num[p][static_cast<size_t>(c)] += I[k * nc + c];
            }
        }
        std::vector<std::array<float, 3>> mean(static_cast<size_t>(P), {{0.5f, 0.5f, 0.5f}});
        for (int p = 0; p < P; ++p) {
            if (den[static_cast<size_t>(p)] <= 0.0) {
                continue;
            }
            for (int c = 0; c < nc; ++c) {
                mean[static_cast<size_t>(p)][static_cast<size_t>(c)] =
                    static_cast<float>(num[static_cast<size_t>(p)][static_cast<size_t>(c)] /
                                       den[static_cast<size_t>(p)]);
            }
        }
        std::vector<int> root(static_cast<size_t>(P));
        for (int p = 0; p < P; ++p) {
            root[static_cast<size_t>(p)] = p;
        }
        auto find = [&root](int a) {
            while (root[static_cast<size_t>(a)] != a) {
                root[static_cast<size_t>(a)] =
                    root[static_cast<size_t>(root[static_cast<size_t>(a)])];
                a = root[static_cast<size_t>(a)];
            }
            return a;
        };
        for (int a = 0; a < P; ++a) {
            for (int b = a + 1; b < P; ++b) {
                if (den[static_cast<size_t>(a)] <= 0.0 || den[static_cast<size_t>(b)] <= 0.0) {
                    continue;
                }
                if (mean_distance(mean[static_cast<size_t>(a)], mean[static_cast<size_t>(b)], nc) <
                    merge_tol) {
                    const int ra = find(a);
                    const int rb = find(b);
                    root[static_cast<size_t>(std::max(ra, rb))] = std::min(ra, rb);
                }
            }
        }
        std::vector<int> id(static_cast<size_t>(P), -1);
        int n = 0;
        for (int p = 0; p < P; ++p) {
            const int rp = find(p);
            if (id[static_cast<size_t>(rp)] < 0) {
                id[static_cast<size_t>(rp)] = n++;
            }
            id[static_cast<size_t>(p)] = id[static_cast<size_t>(rp)];
        }
        cls.resize(n_px);
        for (size_t k = 0; k < n_px; ++k) {
            cls[k] = id[static_cast<size_t>(phase[k])];
        }
        return n;
    }

    static int label_instances(const std::vector<int>& cls, int n_cls, int w, int h,
                               std::vector<int>& inst) {
        const size_t n_px = static_cast<size_t>(w) * static_cast<size_t>(h);
        inst.assign(n_px, 0);
        int next = 0;
        for (int m = 0; m < n_cls; ++m) {
            ImageBuffer mask = make_gray(w, h, 0);
            bool any = false;
            for (size_t k = 0; k < n_px; ++k) {
                if (cls[k] == m) {
                    mask.data[k] = 255;
                    any = true;
                }
            }
            if (!any) {
                continue;
            }
            const auto ccl = vision::ConnectedComponentLabeler::label(mask);
            for (size_t k = 0; k < n_px; ++k) {
                if (ccl.labels[k] > 0) {
                    inst[k] = next + ccl.labels[k];
                }
            }
            next += static_cast<int>(ccl.components.size());
        }
        return next;
    }

    struct Edge {
        int a = 0;
        int b = 0;
        int shared = 0;
    };

    // Region adjacency graph over instance labels, with the shared 4-neighbour
    // boundary length on each edge.
    static std::vector<Edge> adjacency(const std::vector<int>& inst, int n_labels, int w, int h) {
        std::vector<Edge> edges;
        std::vector<std::vector<size_t>> index(static_cast<size_t>(n_labels + 1));
        auto bump = [&](int a, int b) {
            if (a == b || a <= 0 || b <= 0) {
                return;
            }
            if (a > b) {
                std::swap(a, b);
            }
            for (size_t ei : index[static_cast<size_t>(a)]) {
                if (edges[ei].b == b) {
                    ++edges[ei].shared;
                    return;
                }
            }
            index[static_cast<size_t>(a)].push_back(edges.size());
            edges.push_back({a, b, 1});
        };
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int id = inst[static_cast<size_t>(y) * w + x];
                if (x + 1 < w) {
                    bump(id, inst[static_cast<size_t>(y) * w + x + 1]);
                }
                if (y + 1 < h) {
                    bump(id, inst[static_cast<size_t>(y + 1) * w + x]);
                }
            }
        }
        return edges;
    }

    static std::vector<int> areas_of(const std::vector<int>& inst, int n_labels) {
        std::vector<int> area(static_cast<size_t>(n_labels + 1), 0);
        for (int id : inst) {
            if (id > 0) {
                ++area[static_cast<size_t>(id)];
            }
        }
        return area;
    }

    // Rewrites `inst` through the union-find roots and compacts ids to 1..K.
    static void apply_union(std::vector<int>& inst, int& n_labels, std::vector<int>& root) {
        auto find = [&root](int a) {
            while (root[static_cast<size_t>(a)] != a) {
                root[static_cast<size_t>(a)] =
                    root[static_cast<size_t>(root[static_cast<size_t>(a)])];
                a = root[static_cast<size_t>(a)];
            }
            return a;
        };
        std::vector<int> remap(root.size(), 0);
        int next = 0;
        for (size_t k = 0; k < inst.size(); ++k) {
            if (inst[k] <= 0) {
                continue;
            }
            const int rp = find(inst[k]);
            if (remap[static_cast<size_t>(rp)] == 0) {
                remap[static_cast<size_t>(rp)] = ++next;
            }
            inst[k] = remap[static_cast<size_t>(rp)];
        }
        n_labels = next;
    }

    // Speckles are dissolved into whichever larger neighbour they share the most
    // boundary with, so the partition stays a clean cover of the frame.
    static void absorb_small(std::vector<int>& inst, int& n_labels, int w, int h, int min_area) {
        for (int pass = 0; pass < 4 && n_labels > 1; ++pass) {
            const auto area = areas_of(inst, n_labels);
            const auto edges = adjacency(inst, n_labels, w, h);
            std::vector<int> best(static_cast<size_t>(n_labels + 1), 0);
            std::vector<int> best_shared(static_cast<size_t>(n_labels + 1), 0);
            auto consider = [&](int small, int host) {
                if (area[static_cast<size_t>(small)] >= min_area ||
                    area[static_cast<size_t>(host)] <= area[static_cast<size_t>(small)]) {
                    return false;
                }
                return true;
            };
            for (const auto& e : edges) {
                if (consider(e.a, e.b) && e.shared > best_shared[static_cast<size_t>(e.a)]) {
                    best_shared[static_cast<size_t>(e.a)] = e.shared;
                    best[static_cast<size_t>(e.a)] = e.b;
                }
                if (consider(e.b, e.a) && e.shared > best_shared[static_cast<size_t>(e.b)]) {
                    best_shared[static_cast<size_t>(e.b)] = e.shared;
                    best[static_cast<size_t>(e.b)] = e.a;
                }
            }
            std::vector<int> root(static_cast<size_t>(n_labels + 1));
            for (int i = 0; i <= n_labels; ++i) {
                root[static_cast<size_t>(i)] = i;
            }
            bool changed = false;
            for (int id = 1; id <= n_labels; ++id) {
                if (best[static_cast<size_t>(id)] > 0) {
                    root[static_cast<size_t>(id)] = best[static_cast<size_t>(id)];
                    changed = true;
                }
            }
            if (!changed) {
                return;
            }
            apply_union(inst, n_labels, root);
        }
    }

    // Piecewise-constant fitting cuts along φ, not along object outlines, so one
    // object often arrives as several touching pieces of the same material.
    // Merging adjacent instances by fitted mean puts them back together while
    // leaving separate objects of the same colour as separate instances.
    static void merge_adjacent(std::vector<int>& inst, int& n_labels, int w, int h,
                               const std::vector<float>& I, int nc, float tol) {
        for (int pass = 0; pass < 6 && n_labels > 1; ++pass) {
            std::vector<double> area(static_cast<size_t>(n_labels + 1), 0.0);
            std::vector<std::array<double, 3>> sum(static_cast<size_t>(n_labels + 1));
            for (size_t k = 0; k < inst.size(); ++k) {
                const int id = inst[k];
                if (id <= 0) {
                    continue;
                }
                area[static_cast<size_t>(id)] += 1.0;
                for (int c = 0; c < nc; ++c) {
                    sum[static_cast<size_t>(id)][static_cast<size_t>(c)] += I[k * nc + c];
                }
            }
            std::vector<int> root(static_cast<size_t>(n_labels + 1));
            for (int i = 0; i <= n_labels; ++i) {
                root[static_cast<size_t>(i)] = i;
            }
            auto find = [&root](int a) {
                while (root[static_cast<size_t>(a)] != a) {
                    root[static_cast<size_t>(a)] =
                        root[static_cast<size_t>(root[static_cast<size_t>(a)])];
                    a = root[static_cast<size_t>(a)];
                }
                return a;
            };
            auto mean_of = [&](int id) {
                std::array<float, 3> m{{0.0f, 0.0f, 0.0f}};
                if (area[static_cast<size_t>(id)] > 0.0) {
                    for (int c = 0; c < nc; ++c) {
                        m[static_cast<size_t>(c)] =
                            static_cast<float>(sum[static_cast<size_t>(id)][static_cast<size_t>(c)] /
                                               area[static_cast<size_t>(id)]);
                    }
                }
                return m;
            };
            auto edges = adjacency(inst, n_labels, w, h);
            // Closest pair first, so a chain of near-identical regions collapses
            // without dragging in a region that only matches transitively.
            std::sort(edges.begin(), edges.end(), [&](const Edge& l, const Edge& r) {
                return mean_distance(mean_of(l.a), mean_of(l.b), nc) <
                       mean_distance(mean_of(r.a), mean_of(r.b), nc);
            });
            bool changed = false;
            for (const auto& e : edges) {
                const int ra = find(e.a);
                const int rb = find(e.b);
                if (ra == rb) {
                    continue;
                }
                if (mean_distance(mean_of(ra), mean_of(rb), nc) >= tol) {
                    continue;
                }
                const int keep = area[static_cast<size_t>(ra)] >= area[static_cast<size_t>(rb)] ? ra : rb;
                const int drop = keep == ra ? rb : ra;
                root[static_cast<size_t>(drop)] = keep;
                area[static_cast<size_t>(keep)] += area[static_cast<size_t>(drop)];
                for (int c = 0; c < nc; ++c) {
                    sum[static_cast<size_t>(keep)][static_cast<size_t>(c)] +=
                        sum[static_cast<size_t>(drop)][static_cast<size_t>(c)];
                }
                changed = true;
            }
            if (!changed) {
                return;
            }
            apply_union(inst, n_labels, root);
        }
    }

    std::vector<Object> measure(const std::vector<int>& inst, int n_labels,
                                const std::vector<int>& cls, int w, int h,
                                const std::vector<float>& I, int nc) const {
        std::vector<Object> objs(static_cast<size_t>(n_labels));
        std::vector<std::array<double, 3>> acc(static_cast<size_t>(n_labels));
        std::vector<int> min_x(static_cast<size_t>(n_labels), w);
        std::vector<int> min_y(static_cast<size_t>(n_labels), h);
        std::vector<int> max_x(static_cast<size_t>(n_labels), -1);
        std::vector<int> max_y(static_cast<size_t>(n_labels), -1);
        const int band = std::max(
            1, static_cast<int>(border_band_frac * static_cast<float>(std::min(w, h)) + 0.5f));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t k = static_cast<size_t>(y) * w + x;
                const int id = inst[k] - 1;
                if (id < 0) {
                    continue;
                }
                auto& o = objs[static_cast<size_t>(id)];
                o.label = id + 1;
                o.phase = cls[k];
                ++o.area;
                for (int c = 0; c < nc; ++c) {
                    acc[static_cast<size_t>(id)][static_cast<size_t>(c)] += I[k * nc + c];
                }
                min_x[static_cast<size_t>(id)] = std::min(min_x[static_cast<size_t>(id)], x);
                min_y[static_cast<size_t>(id)] = std::min(min_y[static_cast<size_t>(id)], y);
                max_x[static_cast<size_t>(id)] = std::max(max_x[static_cast<size_t>(id)], x);
                max_y[static_cast<size_t>(id)] = std::max(max_y[static_cast<size_t>(id)], y);
                if (x < band || y < band || x >= w - band || y >= h - band) {
                    ++o.border_px;
                }
            }
        }
        for (int id = 0; id < n_labels; ++id) {
            auto& o = objs[static_cast<size_t>(id)];
            if (o.area <= 0) {
                continue;
            }
            for (int c = 0; c < nc; ++c) {
                o.mean[static_cast<size_t>(c)] =
                    static_cast<float>(acc[static_cast<size_t>(id)][static_cast<size_t>(c)] / o.area);
            }
            o.bbox = {static_cast<float>(min_x[static_cast<size_t>(id)]),
                      static_cast<float>(min_y[static_cast<size_t>(id)]),
                      static_cast<float>(max_x[static_cast<size_t>(id)] -
                                         min_x[static_cast<size_t>(id)] + 1),
                      static_cast<float>(max_y[static_cast<size_t>(id)] -
                                         min_y[static_cast<size_t>(id)] + 1)};
            o.bnd_con = static_cast<float>(o.border_px) / std::sqrt(static_cast<float>(o.area));
        }
        return objs;
    }

    // Returns the number of foreground objects.
    int classify_background(std::vector<Object>& objs, const std::vector<int>& inst, int w, int h,
                            int nc) const {
        for (auto& o : objs) {
            o.background = o.bnd_con >= bnd_con_bg;
        }
        if (objs.empty()) {
            return 0;
        }
        if (bg_color_tol > 0.0f) {
            std::array<double, 3> acc{};
            double total = 0.0;
            for (const auto& o : objs) {
                if (!o.background) {
                    continue;
                }
                total += o.area;
                for (int c = 0; c < nc; ++c) {
                    acc[static_cast<size_t>(c)] +=
                        static_cast<double>(o.area) * o.mean[static_cast<size_t>(c)];
                }
            }
            if (total > 0.0) {
                std::array<float, 3> backdrop{{0.0f, 0.0f, 0.0f}};
                for (int c = 0; c < nc; ++c) {
                    backdrop[static_cast<size_t>(c)] =
                        static_cast<float>(acc[static_cast<size_t>(c)] / total);
                }
                for (auto& o : objs) {
                    if (!o.background && o.border_px > 0 &&
                        mean_distance(o.mean, backdrop, nc) < bg_color_tol) {
                        o.background = true;
                    }
                }
            }
        }
        if (std::none_of(objs.begin(), objs.end(), [](const Object& o) { return o.background; })) {
            // Nothing looks like backdrop: demote the most border-hugging region
            // so the atom still reports a foreground/background split.
            size_t bg = 0;
            for (size_t i = 1; i < objs.size(); ++i) {
                if (objs[i].bnd_con > objs[bg].bnd_con) {
                    bg = i;
                }
            }
            objs[bg].background = true;
        }
        // A background region walled off from the frame border is a hole in an
        // object (the gap inside a spectacle rim, a link of a chain), and DIS5K
        // style masks count it as part of the object.
        const auto edges = adjacency(inst, static_cast<int>(objs.size()), w, h);
        for (int pass = 0; pass < 3; ++pass) {
            bool changed = false;
            for (auto& o : objs) {
                if (!o.background || o.border_px > 0) {
                    continue;
                }
                bool enclosed = true;
                int host_area = 0;
                for (const auto& e : edges) {
                    const int other = e.a == o.label ? e.b : (e.b == o.label ? e.a : 0);
                    if (other == 0) {
                        continue;
                    }
                    const auto& nb = objs[static_cast<size_t>(other - 1)];
                    if (nb.background) {
                        enclosed = false;
                        break;
                    }
                    host_area = std::max(host_area, nb.area);
                }
                // A hole is smaller than the thing it is a hole in; without that
                // guard a thin object ringing a wide gap swallows the backdrop.
                if (enclosed && host_area > 0 && o.area < host_area) {
                    o.background = false;
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }
        }
        return static_cast<int>(
            std::count_if(objs.begin(), objs.end(), [](const Object& o) { return !o.background; }));
    }

    void finish(Result& r, const std::vector<int>& inst, const std::vector<float>& I, int w, int h,
                int nc, int n_labels, int n_phases, float energy) {
        const size_t n_px = static_cast<size_t>(w) * static_cast<size_t>(h);
        r.labels = inst;
        r.partition = make_gray(w, h, 0);
        r.label_image = make_gray(w, h, 0);
        r.phase_image = make_gray(w, h, 0);
        double fg_sum = 0.0;
        double fg_n = 0.0;
        double bg_sum = 0.0;
        double bg_n = 0.0;
        for (size_t k = 0; k < n_px; ++k) {
            const int id = inst[k] - 1;
            if (id < 0) {
                continue;
            }
            const auto& o = r.objects[static_cast<size_t>(id)];
            r.partition.data[k] = o.background ? 0 : 255;
            r.label_image.data[k] =
                static_cast<uint8_t>(40 + (static_cast<unsigned>(id + 1) * 47u) % 200u);
            r.phase_image.data[k] =
                static_cast<uint8_t>(n_phases > 1 ? 255 * o.phase / (n_phases - 1) : 255);
            float g = 0.0f;
            for (int c = 0; c < nc; ++c) {
                g += I[k * nc + c];
            }
            g /= static_cast<float>(nc);
            (o.background ? bg_sum : fg_sum) += g;
            (o.background ? bg_n : fg_n) += 1.0;
        }
        c1 = fg_n > 0.0 ? static_cast<float>(fg_sum / fg_n) : 0.0f;
        c2 = bg_n > 0.0 ? static_cast<float>(bg_sum / bg_n) : 0.0f;

        MarchingSquares ms;
        int biggest_area = -1;
        for (const auto& o : r.objects) {
            if (o.background || o.area <= 0) {
                continue;
            }
            ImageBuffer mask = make_gray(w, h, 0);
            for (size_t k = 0; k < n_px; ++k) {
                mask.data[k] = inst[k] == o.label ? 255 : 0;
            }
            Polyline loop = ms.largest_closed(ChamferSDF::from_mask(mask), 0.0f);
            if (loop.points.size() < 3) {
                continue;
            }
            if (o.area > biggest_area) {
                biggest_area = o.area;
                r.contour = loop;
            }
            r.contours.push_back(std::move(loop));
        }

        r.n_regions = n_labels;
        r.n_phases = n_phases;
        r.energy = energy;
    }
};

}  // namespace contour
