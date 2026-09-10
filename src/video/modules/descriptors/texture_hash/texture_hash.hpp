#pragma once

// Deterministic surface-material fingerprints for an arbitrary region mask.
//
// Four complementary views of "what is this made of", none of which need the
// region to be a rectangle:
//
//   1. Rotation-invariant uniform LBP -- micro-pattern census. Compares each
//      pixel only to its own neighbours, so it survives any monotonic
//      illumination change.
//   2. 64-bit DCT perceptual hash -- the spatial frequency layout, compared
//      with a single popcount. Median thresholding makes it invariant to
//      contrast scaling as well as brightness offset.
//   3. GLCM scalars + structure tensor -- roughness, regularity and grain
//      direction.
//   4. CIELAB covariance -- how the colour itself varies across the surface,
//      which separates a plaid weave from a single hue under folded shadow.
//
// Together these answer "are these two polygons cut from the same material?"
// without answering "are they the same shape?", which is deliberately the
// shape descriptor's job.

#include "filters/lab_color/lab_color_space.hpp"
#include "math/contour_compat.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace descriptors {

using math::ImageBuffer;

// A region expressed as a label map plus the id to select, which is what both
// the segmentation and the ground-truth ingest paths already produce.
struct RegionView {
    const std::vector<int>* labels = nullptr;
    int width = 0;
    int height = 0;
    int target = 0;
    int min_x = 0;
    int min_y = 0;
    int max_x = 0;
    int max_y = 0;
    int area = 0;

    bool contains(int x, int y) const {
        return x >= 0 && y >= 0 && x < width && y < height &&
               (*labels)[static_cast<size_t>(y * width + x)] == target;
    }
};

inline RegionView make_region_view(const std::vector<int>& labels, int width, int height,
                                   int target) {
    RegionView view;
    view.labels = &labels;
    view.width = width;
    view.height = height;
    view.target = target;
    view.min_x = width;
    view.min_y = height;
    view.max_x = -1;
    view.max_y = -1;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (labels[static_cast<size_t>(y * width + x)] != target) {
                continue;
            }
            view.min_x = std::min(view.min_x, x);
            view.max_x = std::max(view.max_x, x);
            view.min_y = std::min(view.min_y, y);
            view.max_y = std::max(view.max_y, y);
            ++view.area;
        }
    }
    return view;
}

struct TextureSignature {
    uint64_t spatial_hash = 0;
    float hash_confidence = 0.0f;   // 0 when the surface is too flat to fingerprint
    std::array<uint8_t, 10> lbp{};  // riu2 histogram, quantized to sum 255
    float roughness = 0.0f;         // GLCM contrast, normalized to [0, 1]
    float energy = 0.0f;            // GLCM angular second moment
    float homogeneity = 0.0f;       // GLCM inverse difference moment
    float anisotropy = 0.0f;        // 0 isotropic, 1 single grain direction
    float orientation = 0.0f;       // dominant grain angle in radians
    std::array<float, 6> lab_covariance{};  // LL, aa, bb, La, Lb, ab
    float color_roughness = 0.0f;           // trace of the covariance
    std::array<float, 3> mean_lab{};
    bool valid = false;
};

// ---------------------------------------------------------------------------
// Rotation-invariant uniform Local Binary Patterns
// ---------------------------------------------------------------------------

class LocalBinaryPattern {
public:
    // The classic uniform LBP has 59 bins for P=8, but those bins still encode
    // *which* neighbour the micro-edge points at, so a rotated patch lands in a
    // different bin. Collapsing each rotation class to its popcount (Ojala's
    // riu2) is the actually rotation-invariant form and needs only 10 bins:
    // nine uniform classes plus one catch-all for noisy, non-uniform pixels.
    static constexpr int kBins = 10;

    static uint8_t code_at(const ImageBuffer& luma, int x, int y) {
        static constexpr int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        static constexpr int dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};
        const float center = luma.gray(x, y);
        uint8_t code = 0;
        for (int i = 0; i < 8; ++i) {
            const int xx = std::clamp(x + dx[i], 0, luma.width - 1);
            const int yy = std::clamp(y + dy[i], 0, luma.height - 1);
            if (luma.gray(xx, yy) >= center) {
                code |= static_cast<uint8_t>(1u << i);
            }
        }
        return code;
    }

    static int transitions(uint8_t code) {
        int count = 0;
        for (int i = 0; i < 8; ++i) {
            const int a = (code >> i) & 1;
            const int b = (code >> ((i + 1) % 8)) & 1;
            count += a != b ? 1 : 0;
        }
        return count;
    }

    static int riu2_bin(uint8_t code) {
        if (transitions(code) > 2) {
            return kBins - 1;
        }
        int ones = 0;
        for (int i = 0; i < 8; ++i) {
            ones += (code >> i) & 1;
        }
        return ones;  // 0..8
    }

    static std::array<uint8_t, kBins> histogram(const ImageBuffer& luma, const RegionView& region) {
        std::array<int, kBins> counts{};
        int total = 0;
        for (int y = region.min_y; y <= region.max_y; ++y) {
            for (int x = region.min_x; x <= region.max_x; ++x) {
                if (!region.contains(x, y)) {
                    continue;
                }
                ++counts[static_cast<size_t>(riu2_bin(code_at(luma, x, y)))];
                ++total;
            }
        }
        std::array<uint8_t, kBins> out{};
        if (total <= 0) {
            return out;
        }
        for (int i = 0; i < kBins; ++i) {
            out[static_cast<size_t>(i)] = static_cast<uint8_t>(std::clamp(
                static_cast<int>(std::lround(255.0 * counts[static_cast<size_t>(i)] / total)), 0,
                255));
        }
        return out;
    }

    static float chi_square(const std::array<uint8_t, kBins>& a,
                            const std::array<uint8_t, kBins>& b) {
        float sum = 0.0f;
        for (int i = 0; i < kBins; ++i) {
            const float p = static_cast<float>(a[static_cast<size_t>(i)]);
            const float q = static_cast<float>(b[static_cast<size_t>(i)]);
            const float denominator = p + q;
            if (denominator > 0.0f) {
                sum += (p - q) * (p - q) / denominator;
            }
        }
        return sum / 255.0f;  // roughly [0, 2]
    }
};

// ---------------------------------------------------------------------------
// DCT perceptual hash
// ---------------------------------------------------------------------------

class PerceptualHash {
public:
    static constexpr int kPatch = 32;

    // Outside-region pixels are filled with the region's own mean rather than
    // zero. A zero fill would stamp the silhouette into the frequency spectrum
    // and the "hash" would end up describing shape, which is the one thing it
    // must not do.
    static std::array<float, kPatch * kPatch> resample(const ImageBuffer& luma,
                                                       const RegionView& region) {
        std::array<float, kPatch * kPatch> patch{};
        const int w = std::max(1, region.max_x - region.min_x + 1);
        const int h = std::max(1, region.max_y - region.min_y + 1);
        double sum = 0.0;
        int count = 0;
        for (int y = region.min_y; y <= region.max_y; ++y) {
            for (int x = region.min_x; x <= region.max_x; ++x) {
                if (region.contains(x, y)) {
                    sum += luma.gray(x, y);
                    ++count;
                }
            }
        }
        const float fill = count > 0 ? static_cast<float>(sum / count) : 0.0f;
        for (int py = 0; py < kPatch; ++py) {
            for (int px = 0; px < kPatch; ++px) {
                const int sx = region.min_x + (px * w) / kPatch;
                const int sy = region.min_y + (py * h) / kPatch;
                patch[static_cast<size_t>(py * kPatch + px)] =
                    region.contains(sx, sy) ? luma.gray(sx, sy) : fill;
            }
        }
        return patch;
    }

    static std::array<float, kPatch * kPatch> dct2(const std::array<float, kPatch * kPatch>& in) {
        // Separable DCT-II with a precomputed basis; kPatch is small and fixed
        // so the O(n^3) form costs less than setting up an FFT.
        static const std::array<float, kPatch * kPatch> basis = build_basis();
        std::array<float, kPatch * kPatch> rows{};
        for (int y = 0; y < kPatch; ++y) {
            for (int u = 0; u < kPatch; ++u) {
                float sum = 0.0f;
                for (int x = 0; x < kPatch; ++x) {
                    sum += in[static_cast<size_t>(y * kPatch + x)] *
                           basis[static_cast<size_t>(u * kPatch + x)];
                }
                rows[static_cast<size_t>(y * kPatch + u)] = sum;
            }
        }
        std::array<float, kPatch * kPatch> out{};
        for (int u = 0; u < kPatch; ++u) {
            for (int v = 0; v < kPatch; ++v) {
                float sum = 0.0f;
                for (int y = 0; y < kPatch; ++y) {
                    sum += rows[static_cast<size_t>(y * kPatch + u)] *
                           basis[static_cast<size_t>(v * kPatch + y)];
                }
                out[static_cast<size_t>(v * kPatch + u)] = sum;
            }
        }
        return out;
    }

    struct Result {
        uint64_t bits = 0;
        // How far the retained coefficients sit above the noise a uint8 image
        // quantizes in. A near-uniform surface has no spatial frequency content
        // to fingerprint, so its bits are a coin flip and must not be allowed
        // to masquerade as evidence of a material.
        float confidence = 0.0f;
    };

    static Result hash(const ImageBuffer& luma, const RegionView& region) {
        const std::array<float, kPatch * kPatch> spectrum = dct2(resample(luma, region));
        // Zigzag order walks outward from DC through increasing spatial
        // frequency; entries 1..64 are the low-to-mid band, and skipping entry
        // 0 discards the average brightness.
        std::array<float, 64> band{};
        int taken = 0;
        for (int diagonal = 0; diagonal < 2 * kPatch - 1 && taken < 65; ++diagonal) {
            const int lo = std::max(0, diagonal - kPatch + 1);
            const int hi = std::min(diagonal, kPatch - 1);
            for (int i = lo; i <= hi && taken < 65; ++i) {
                const int u = (diagonal % 2 == 0) ? diagonal - i : i;
                const int v = (diagonal % 2 == 0) ? i : diagonal - i;
                if (taken > 0) {
                    band[static_cast<size_t>(taken - 1)] =
                        spectrum[static_cast<size_t>(v * kPatch + u)];
                }
                ++taken;
            }
        }
        std::array<float, 64> sorted = band;
        std::nth_element(sorted.begin(), sorted.begin() + 32, sorted.end());
        const float median = sorted[32];
        Result result;
        float deviation = 0.0f;
        for (int i = 0; i < 64; ++i) {
            if (band[static_cast<size_t>(i)] > median) {
                result.bits |= (1ull << i);
            }
            deviation += std::fabs(band[static_cast<size_t>(i)] - median);
        }
        // Rounding to uint8 injects roughly +/-0.5 of a grey level, which an
        // orthonormal DCT spreads into coefficients of about 0.3; a mean
        // deviation well past that means the bits describe real structure.
        result.confidence = std::clamp((deviation / 64.0f - 0.6f) / 2.4f, 0.0f, 1.0f);
        return result;
    }

    static int hamming(uint64_t a, uint64_t b) {
        uint64_t diff = a ^ b;
        int count = 0;
        while (diff != 0) {
            diff &= diff - 1;
            ++count;
        }
        return count;
    }

private:
    static std::array<float, kPatch * kPatch> build_basis() {
        std::array<float, kPatch * kPatch> basis{};
        for (int u = 0; u < kPatch; ++u) {
            const float alpha = u == 0 ? std::sqrt(1.0f / kPatch) : std::sqrt(2.0f / kPatch);
            for (int x = 0; x < kPatch; ++x) {
                basis[static_cast<size_t>(u * kPatch + x)] =
                    alpha * std::cos(math::kPi * (2.0f * x + 1.0f) * u / (2.0f * kPatch));
            }
        }
        return basis;
    }
};

// ---------------------------------------------------------------------------
// Full signature
// ---------------------------------------------------------------------------

inline TextureSignature analyze_texture(const ImageBuffer& rgb, const ImageBuffer& luma,
                                        const RegionView& region) {
    TextureSignature signature;
    if (region.area <= 0 || region.max_x < region.min_x) {
        return signature;
    }
    const PerceptualHash::Result hashed = PerceptualHash::hash(luma, region);
    signature.spatial_hash = hashed.bits;
    signature.hash_confidence = hashed.confidence;
    signature.lbp = LocalBinaryPattern::histogram(luma, region);

    // GLCM over 8 quantized grey levels, averaged across the four axis and
    // diagonal offsets so the scalars themselves are direction-agnostic; the
    // structure tensor below is what reports direction.
    constexpr int kLevels = 8;
    std::array<double, kLevels * kLevels> glcm{};
    double pairs = 0.0;
    static constexpr int ox[4] = {1, 1, 0, -1};
    static constexpr int oy[4] = {0, 1, 1, 1};
    auto level_at = [&](int x, int y) {
        return std::clamp(static_cast<int>(luma.gray(x, y)) / 32, 0, kLevels - 1);
    };
    for (int y = region.min_y; y <= region.max_y; ++y) {
        for (int x = region.min_x; x <= region.max_x; ++x) {
            if (!region.contains(x, y)) {
                continue;
            }
            const int i = level_at(x, y);
            for (int o = 0; o < 4; ++o) {
                const int xx = x + ox[o];
                const int yy = y + oy[o];
                if (!region.contains(xx, yy)) {
                    continue;
                }
                const int j = level_at(xx, yy);
                glcm[static_cast<size_t>(i * kLevels + j)] += 1.0;
                glcm[static_cast<size_t>(j * kLevels + i)] += 1.0;
                pairs += 2.0;
            }
        }
    }
    if (pairs > 0.0) {
        double contrast = 0.0;
        double energy = 0.0;
        double homogeneity = 0.0;
        for (int i = 0; i < kLevels; ++i) {
            for (int j = 0; j < kLevels; ++j) {
                const double p = glcm[static_cast<size_t>(i * kLevels + j)] / pairs;
                const double d = i - j;
                contrast += d * d * p;
                energy += p * p;
                homogeneity += p / (1.0 + std::fabs(d));
            }
        }
        const double max_contrast = (kLevels - 1) * (kLevels - 1);
        signature.roughness = static_cast<float>(contrast / max_contrast);
        signature.energy = static_cast<float>(energy);
        signature.homogeneity = static_cast<float>(homogeneity);
    }

    // Gradient structure tensor: its eigenvalue split is the standard coherence
    // measure, and its eigenvector is the grain direction.
    double jxx = 0.0;
    double jxy = 0.0;
    double jyy = 0.0;
    for (int y = region.min_y; y <= region.max_y; ++y) {
        for (int x = region.min_x; x <= region.max_x; ++x) {
            if (!region.contains(x, y)) {
                continue;
            }
            const float gx = luma.gray(std::min(luma.width - 1, x + 1), y) -
                             luma.gray(std::max(0, x - 1), y);
            const float gy = luma.gray(x, std::min(luma.height - 1, y + 1)) -
                             luma.gray(x, std::max(0, y - 1));
            jxx += gx * gx;
            jxy += gx * gy;
            jyy += gy * gy;
        }
    }
    const double trace = jxx + jyy;
    if (trace > 1e-9) {
        const double diff = std::sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy);
        signature.anisotropy = static_cast<float>(std::clamp(diff / trace, 0.0, 1.0));
        signature.orientation = 0.5f * static_cast<float>(std::atan2(2.0 * jxy, jxx - jyy));
    }

    // CIELAB covariance over the interior.
    double sum_l = 0.0;
    double sum_a = 0.0;
    double sum_b = 0.0;
    int count = 0;
    for (int y = region.min_y; y <= region.max_y; ++y) {
        for (int x = region.min_x; x <= region.max_x; ++x) {
            if (!region.contains(x, y)) {
                continue;
            }
            const contour::Lab lab = contour::LabColor::at(rgb, x, y);
            sum_l += lab.L;
            sum_a += lab.a;
            sum_b += lab.b;
            ++count;
        }
    }
    if (count > 0) {
        const double mean_l = sum_l / count;
        const double mean_a = sum_a / count;
        const double mean_b = sum_b / count;
        signature.mean_lab = {static_cast<float>(mean_l), static_cast<float>(mean_a),
                              static_cast<float>(mean_b)};
        double cll = 0.0;
        double caa = 0.0;
        double cbb = 0.0;
        double cla = 0.0;
        double clb = 0.0;
        double cab = 0.0;
        for (int y = region.min_y; y <= region.max_y; ++y) {
            for (int x = region.min_x; x <= region.max_x; ++x) {
                if (!region.contains(x, y)) {
                    continue;
                }
                const contour::Lab lab = contour::LabColor::at(rgb, x, y);
                const double dl = lab.L - mean_l;
                const double da = lab.a - mean_a;
                const double db = lab.b - mean_b;
                cll += dl * dl;
                caa += da * da;
                cbb += db * db;
                cla += dl * da;
                clb += dl * db;
                cab += da * db;
            }
        }
        const double inv = 1.0 / count;
        signature.lab_covariance = {
            static_cast<float>(cll * inv), static_cast<float>(caa * inv),
            static_cast<float>(cbb * inv), static_cast<float>(cla * inv),
            static_cast<float>(clb * inv), static_cast<float>(cab * inv)};
        signature.color_roughness =
            static_cast<float>(std::sqrt((cll + caa + cbb) * inv));
    }
    signature.valid = true;
    return signature;
}

// Material agreement in [0, 1]. Separate from shape distance on purpose: the
// classifier asks the two questions independently and the DAG combines them.
inline float material_distance(const TextureSignature& a, const TextureSignature& b) {
    if (!a.valid || !b.valid) {
        return 1.0f;
    }
    const float hash_term =
        static_cast<float>(PerceptualHash::hamming(a.spatial_hash, b.spatial_hash)) / 64.0f;
    const float lbp_term = std::clamp(LocalBinaryPattern::chi_square(a.lbp, b.lbp) * 0.5f, 0.0f,
                                      1.0f);
    const float rough_term = std::clamp(std::fabs(a.roughness - b.roughness) * 3.0f, 0.0f, 1.0f);
    const float grain_term = std::clamp(std::fabs(a.anisotropy - b.anisotropy), 0.0f, 1.0f);
    // Only trust the frequency fingerprint as far as both surfaces have one.
    // Its share of the budget is handed to the LBP census, which stays
    // meaningful on flat surfaces because it is purely ordinal.
    const float trust = std::min(a.hash_confidence, b.hash_confidence);
    const float hash_weight = 0.40f * trust;
    const float lbp_weight = 0.35f + 0.40f * (1.0f - trust);
    return std::clamp(hash_weight * hash_term + lbp_weight * lbp_term + 0.15f * rough_term +
                          0.10f * grain_term,
                      0.0f, 1.0f);
}

inline float lab_delta_e(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    const float dl = a[0] - b[0];
    const float da = a[1] - b[1];
    const float db = a[2] - b[2];
    return std::sqrt(dl * dl + da * da + db * db);
}

}  // namespace descriptors
