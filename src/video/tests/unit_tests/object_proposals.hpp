#pragma once

// Object proposals for CCL demos: one box per separable visual region.
// Bright scenes: dark silhouettes + black-tophat rods + snow ground.
// Mid/dark: chroma seeds + gradient-watershed basins (bears / clutter).
// Not GT-matched.

#include "math/contour_compat.hpp"
#include "filters/edge/canny/canny.hpp"
#include "filters/lab_color/lab_color_space.hpp"
#include "filters/bilateral/bilateral.hpp"
#include "segmentation/ccl/connected_components.hpp"
#include "segmentation/watershed/watershed.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

namespace vision {

inline GrayImage morph_dilate(const GrayImage& src, int r) {
    GrayImage out = src;
    if (r <= 0) {
        return out;
    }
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            if (src.at(x, y) == 0) {
                continue;
            }
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx >= 0 && yy >= 0 && xx < src.width && yy < src.height) {
                        out.at(xx, yy) = 255;
                    }
                }
            }
        }
    }
    return out;
}

inline GrayImage morph_erode(const GrayImage& src, int r) {
    GrayImage inv = src;
    for (uint8_t& p : inv.data) {
        p = p ? 0 : 255;
    }
    inv = morph_dilate(inv, r);
    for (uint8_t& p : inv.data) {
        p = p ? 0 : 255;
    }
    return inv;
}

inline GrayImage morph_close(const GrayImage& src, int r) {
    return morph_erode(morph_dilate(src, r), r);
}

inline GrayImage morph_open(const GrayImage& src, int r) {
    return morph_dilate(morph_erode(src, r), r);
}

inline GrayImage gray_dilate(const GrayImage& src, int r) {
    GrayImage out = make_gray(src.width, src.height, 0);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            float m = 0.0f;
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    const int xx = std::clamp(x + dx, 0, src.width - 1);
                    const int yy = std::clamp(y + dy, 0, src.height - 1);
                    m = std::max(m, src.gray(xx, yy));
                }
            }
            out.at(x, y) = static_cast<uint8_t>(std::clamp(m, 0.0f, 255.0f));
        }
    }
    return out;
}

inline GrayImage gray_erode(const GrayImage& src, int r) {
    GrayImage out = make_gray(src.width, src.height, 255);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            float m = 255.0f;
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    const int xx = std::clamp(x + dx, 0, src.width - 1);
                    const int yy = std::clamp(y + dy, 0, src.height - 1);
                    m = std::min(m, src.gray(xx, yy));
                }
            }
            out.at(x, y) = static_cast<uint8_t>(std::clamp(m, 0.0f, 255.0f));
        }
    }
    return out;
}

inline GrayImage black_tophat(const GrayImage& src, int r) {
    const GrayImage closed = gray_erode(gray_dilate(src, r), r);
    GrayImage hat = make_gray(src.width, src.height, 0);
    for (size_t i = 0; i < hat.data.size(); ++i) {
        const float v = static_cast<float>(closed.data[i]) - static_cast<float>(src.data[i]);
        hat.data[i] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
    }
    return hat;
}

inline GrayImage or_mask(const GrayImage& a, const GrayImage& b) {
    GrayImage out = a;
    for (size_t i = 0; i < out.data.size(); ++i) {
        if (b.data[i]) {
            out.data[i] = 255;
        }
    }
    return out;
}

inline GrayImage chrominance_saliency(const GrayImage& rgb_or_gray) {
    GrayImage out = make_gray(rgb_or_gray.width, rgb_or_gray.height, 0);
    float maxv = 1e-6f;
    std::vector<float> tmp(out.data.size(), 0.0f);
    for (int y = 0; y < rgb_or_gray.height; ++y) {
        for (int x = 0; x < rgb_or_gray.width; ++x) {
            const auto lab = contour::LabColor::at(rgb_or_gray, x, y);
            float s = std::fabs(lab.a) + std::fabs(lab.b);
            if (lab.a > 12.0f && lab.L > 12.0f && lab.L < 88.0f) {
                s += 2.0f * lab.a;
            }
            tmp[static_cast<size_t>(y * rgb_or_gray.width + x)] = s;
            maxv = std::max(maxv, s);
        }
    }
    for (size_t i = 0; i < out.data.size(); ++i) {
        out.data[i] = static_cast<uint8_t>(std::clamp(tmp[i] / maxv * 255.0f, 0.0f, 255.0f));
    }
    return out;
}

inline GrayImage sky_mask(const GrayImage& rgb_or_gray, float l_thr = 88.0f, float c_thr = 14.0f) {
    GrayImage out = make_gray(rgb_or_gray.width, rgb_or_gray.height, 0);
    const int y_sky = std::max(1, rgb_or_gray.height * 42 / 100);  // only upper band
    for (int y = 0; y < y_sky; ++y) {
        for (int x = 0; x < rgb_or_gray.width; ++x) {
            const auto lab = contour::LabColor::at(rgb_or_gray, x, y);
            const float c = std::fabs(lab.a) + std::fabs(lab.b);
            if (lab.L >= l_thr && c <= c_thr) {
                out.at(x, y) = 255;
            }
        }
    }
    return out;
}

struct ObjectProposal {
    GrayImage binary;
    GrayImage body;
    GrayImage thin;
    GrayImage ground;
    GrayImage edges;
    GrayImage saliency;
    GrayImage tophat;
    GrayImage smooth;  // for intensity watershed splits
};

inline uint8_t percentile_u8(std::vector<uint8_t> v, double q) {
    if (v.empty()) {
        return 128;
    }
    q = std::clamp(q, 0.0, 1.0);
    const size_t k = static_cast<size_t>(q * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
    return v[k];
}

inline GrayImage smooth_luma(const GrayImage& luma) {
    contour::BilateralFilter bilat;
    bilat.radius = 4;
    bilat.sigma_s = 2.5f;
    bilat.sigma_r = 32.0f;
    return bilat.apply(luma);
}

inline void paint_label(GrayImage& dst, const std::vector<int>& labels, int W, int H, int lab) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (labels[static_cast<size_t>(y * W + x)] == lab) {
                dst.at(x, y) = 255;
            }
        }
    }
}

// Grow regions from a seed grid where neighbors stay within luma tolerance.
// Separates midtone objects (bears) that morphology cannot peel apart.
inline std::vector<GrayImage> intensity_grid_segments(const GrayImage& smooth, int step,
                                                      float tol, int min_area, int max_area) {
    std::vector<GrayImage> out;
    const int W = smooth.width;
    const int H = smooth.height;
    std::vector<int> labels(static_cast<size_t>(W * H), 0);
    int next = 1;
    auto flood = [&](int sx, int sy) {
        const size_t si = static_cast<size_t>(sy * W + sx);
        if (labels[si] != 0) {
            return;
        }
        const float seed_v = smooth.gray(sx, sy);
        std::vector<int> st;
        st.push_back(static_cast<int>(si));
        labels[si] = next;
        int area = 0;
        while (!st.empty()) {
            const int idx = st.back();
            st.pop_back();
            ++area;
            const int x = idx % W;
            const int y = idx / W;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= W || yy >= H) {
                        continue;
                    }
                    const size_t ni = static_cast<size_t>(yy * W + xx);
                    if (labels[ni] != 0) {
                        continue;
                    }
                    if (std::fabs(smooth.gray(xx, yy) - seed_v) <= tol) {
                        labels[ni] = next;
                        st.push_back(static_cast<int>(ni));
                    }
                }
            }
        }
        if (area >= min_area && area <= max_area) {
            GrayImage piece = make_gray(W, H, 0);
            for (size_t i = 0; i < labels.size(); ++i) {
                if (labels[i] == next) {
                    piece.data[i] = 255;
                }
            }
            out.push_back(std::move(piece));
        } else {
            // Free rejected pixels so a later seed can claim them.
            for (size_t i = 0; i < labels.size(); ++i) {
                if (labels[i] == next) {
                    labels[i] = 0;
                }
            }
        }
        ++next;
    };
    for (int y = step / 2; y < H; y += step) {
        for (int x = step / 2; x < W; x += step) {
            flood(x, y);
        }
    }
    return out;
}

inline GrayImage keep_elongated(const GrayImage& src, int min_area, float min_aspect) {
    GrayImage out = make_gray(src.width, src.height, 0);
    auto ccl = ConnectedComponentLabeler::label(src);
    const int W = src.width;
    const int H = src.height;
    for (const auto& c : ccl.components) {
        if (c.area < min_area) {
            continue;
        }
        const float aspect = (c.bbox.h > 1.0f) ? (c.bbox.w / c.bbox.h) : 99.0f;
        const float long_aspect =
            std::max(aspect, (c.bbox.w > 1.0f) ? (c.bbox.h / c.bbox.w) : 99.0f);
        if (long_aspect < min_aspect) {
            continue;
        }
        paint_label(out, ccl.labels, W, H, c.label);
    }
    return out;
}

// Grow regions that stay similar in luma and do not cross thickened edges.
inline GrayImage edge_aware_regions(const GrayImage& smooth, const GrayImage& edges, float tol,
                                    int step, int min_area, int max_area) {
    const int W = smooth.width;
    const int H = smooth.height;
    GrayImage barrier = morph_dilate(edges, 1);
    std::vector<int> labels(static_cast<size_t>(W * H), 0);
    int next = 1;
    GrayImage out = make_gray(W, H, 0);

    auto flood = [&](int sx, int sy) {
        const size_t si = static_cast<size_t>(sy * W + sx);
        if (labels[si] != 0 || barrier.data[si]) {
            return;
        }
        const float seed_v = smooth.gray(sx, sy);
        std::vector<int> st;
        st.push_back(static_cast<int>(si));
        labels[si] = next;
        int area = 0;
        int minx = sx, maxx = sx, miny = sy, maxy = sy;
        while (!st.empty()) {
            const int idx = st.back();
            st.pop_back();
            ++area;
            const int x = idx % W;
            const int y = idx / W;
            minx = std::min(minx, x);
            maxx = std::max(maxx, x);
            miny = std::min(miny, y);
            maxy = std::max(maxy, y);
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= W || yy >= H) {
                        continue;
                    }
                    const size_t ni = static_cast<size_t>(yy * W + xx);
                    if (labels[ni] != 0 || barrier.data[ni]) {
                        continue;
                    }
                    if (std::fabs(smooth.gray(xx, yy) - seed_v) <= tol) {
                        labels[ni] = next;
                        st.push_back(static_cast<int>(ni));
                    }
                }
            }
        }
        const float bw = static_cast<float>(maxx - minx + 1);
        const float bh = static_cast<float>(maxy - miny + 1);
        const float short_side = std::min(bw, bh);
        const float long_aspect = std::max(bw, bh) / std::max(1.0f, short_side);
        const bool ok = area >= min_area && area <= max_area &&
                        !(short_side <= 5.0f && long_aspect >= 5.0f);
        if (ok) {
            GrayImage piece = make_gray(W, H, 0);
            for (size_t i = 0; i < labels.size(); ++i) {
                if (labels[i] == next) {
                    piece.data[i] = 255;
                }
            }
            // 1px gap so adjacent regions stay separate under later CCL.
            piece = morph_erode(piece, 1);
            int a2 = 0;
            for (uint8_t v : piece.data) {
                a2 += v > 0 ? 1 : 0;
            }
            if (a2 < min_area / 2) {
                // restore if erode killed it
                for (size_t i = 0; i < labels.size(); ++i) {
                    if (labels[i] == next) {
                        out.data[i] = 255;
                    }
                }
            } else {
                out = or_mask(out, piece);
            }
        } else {
            for (size_t i = 0; i < labels.size(); ++i) {
                if (labels[i] == next) {
                    labels[i] = 0;
                }
            }
        }
        ++next;
    };

    for (int y = step / 2; y < H; y += step) {
        for (int x = step / 2; x < W; x += step) {
            flood(x, y);
        }
    }
    return out;
}

// Legacy closed-interior helper (kept for thin crack sealing experiments).
inline GrayImage regions_between_edges(const GrayImage& edges, int seal_r, int min_area,
                                       int max_area) {
    const int W = edges.width;
    const int H = edges.height;
    GrayImage barrier = morph_dilate(edges, 1);
    GrayImage interior = make_gray(W, H, 0);
    for (size_t i = 0; i < interior.data.size(); ++i) {
        interior.data[i] = barrier.data[i] ? 0 : 255;
    }
    // Seal hairline gaps (door cracks, drawer slits) without welding stove↔fridge.
    if (seal_r > 0) {
        interior = morph_close(interior, seal_r);
    }
    GrayImage out = make_gray(W, H, 0);
    auto ccl = ConnectedComponentLabeler::label(interior);
    for (const auto& c : ccl.components) {
        if (c.area < min_area || c.area > max_area) {
            continue;
        }
        const float aspect = (c.bbox.h > 1.0f) ? (c.bbox.w / c.bbox.h) : 99.0f;
        const float long_aspect =
            std::max(aspect, (c.bbox.w > 1.0f) ? (c.bbox.h / c.bbox.w) : 99.0f);
        const float short_side = std::min(c.bbox.w, c.bbox.h);
        // Drop crack/sliver components (door seams, grout, hand tips).
        if (short_side <= 4.0f && long_aspect >= 5.0f) {
            continue;
        }
        if (short_side <= 6.0f && c.area < min_area * 2) {
            continue;
        }
        paint_label(out, ccl.labels, W, H, c.label);
    }
    return out;
}

inline ObjectProposal propose_objects_from_photo(const GrayImage& rgb_or_luma) {
    ObjectProposal p;
    const int W = rgb_or_luma.width;
    const int H = rgb_or_luma.height;
    const int img_area = std::max(1, W * H);

    GrayImage luma = make_gray(W, H, 0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            luma.at(x, y) = static_cast<uint8_t>(rgb_or_luma.gray(x, y));
        }
    }

    p.smooth = smooth_luma(luma);
    contour::Canny canny;
    canny.low = 0.04f;
    canny.high = 0.11f;
    // Edges from smoothed luma → fewer fur/leaf speckles.
    p.edges = morph_close(canny.detect(p.smooth), 1);
    p.saliency = chrominance_saliency(rgb_or_luma);
    auto sky = morph_dilate(sky_mask(rgb_or_luma), 1);

    std::vector<uint8_t> vals(p.smooth.data.begin(), p.smooth.data.end());
    const uint8_t med = percentile_u8(vals, 0.50);
    const uint8_t p12 = percentile_u8(vals, 0.12);
    const uint8_t p75 = percentile_u8(vals, 0.75);
    const uint8_t p85 = percentile_u8(vals, 0.85);
    const bool bright_scene = med >= 125;

    p.body = make_gray(W, H, 0);
    p.thin = make_gray(W, H, 0);
    p.ground = make_gray(W, H, 0);
    p.binary = make_gray(W, H, 0);
    p.tophat = black_tophat(p.smooth, bright_scene ? 2 : 3);

    // Primary: edge-aware intensity floods (separable areas between edges).
    const int min_region = std::max(200, img_area * 2 / 100);
    const int max_region = img_area * 40 / 100;
    {
        GrayImage best;
        int best_count = 0;
        for (float tol : {10.0f, 14.0f, 18.0f, 22.0f}) {
            auto reg = edge_aware_regions(p.smooth, p.edges, tol, 20, min_region, max_region);
            auto ccl = ConnectedComponentLabeler::label(reg);
            int kept = 0;
            for (const auto& c : ccl.components) {
                if (c.area >= min_region) {
                    ++kept;
                }
            }
            if (kept >= 3 && kept <= 12) {
                best = std::move(reg);
                best_count = kept;
                break;
            }
            if (kept > best_count && kept <= 18) {
                best_count = kept;
                best = std::move(reg);
            }
        }
        if (best_count > 0) {
            p.body = or_mask(p.body, best);
        }
    }

    // Chroma accents (stop sign, colored jerseys).
    GrayImage chroma = make_gray(W, H, 0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t i = static_cast<size_t>(y * W + x);
            if (sky.data[i]) {
                continue;
            }
            if (p.saliency.data[i] >= 55) {
                chroma.data[i] = 255;
            }
            if (rgb_or_luma.channels >= 3) {
                const auto lab = contour::LabColor::at(rgb_or_luma, x, y);
                if (lab.a > 14.0f && lab.L > 12.0f && lab.L < 88.0f) {
                    chroma.data[i] = 255;
                }
            }
        }
    }
    chroma = morph_close(chroma, 2);
    {
        auto ccl_c = ConnectedComponentLabeler::label(chroma);
        for (const auto& c : ccl_c.components) {
            if (c.area < min_region || c.area > img_area * 25 / 100) {
                continue;
            }
            paint_label(p.body, ccl_c.labels, W, H, c.label);
        }
    }

    if (bright_scene) {
        // Dark silhouettes (skier, poles, skis) on snow.
        const int dark_thr = std::min<int>(static_cast<int>(p12) + 20, static_cast<int>(med) - 28);
        GrayImage dark = make_gray(W, H, 0);
        for (size_t i = 0; i < dark.data.size(); ++i) {
            if (!sky.data[i] && static_cast<int>(p.smooth.data[i]) <= dark_thr) {
                dark.data[i] = 255;
            }
        }
        dark = morph_close(dark, 2);
        const GrayImage thick = morph_open(dark, 3);  // peel poles
        {
            auto ccl_b = ConnectedComponentLabeler::label(thick);
            for (const auto& c : ccl_b.components) {
                if (c.area < min_region || c.area > img_area * 35 / 100) {
                    continue;
                }
                paint_label(p.body, ccl_b.labels, W, H, c.label);
            }
        }
        // Thin rods: residual only, light dilate so two poles stay separate.
        GrayImage residual = make_gray(W, H, 0);
        for (size_t i = 0; i < residual.data.size(); ++i) {
            if (dark.data[i] && !thick.data[i]) {
                residual.data[i] = 255;
            }
        }
        std::vector<uint8_t> hv(p.tophat.data.begin(), p.tophat.data.end());
        const uint8_t hat_thr = std::max<uint8_t>(6, percentile_u8(hv, 0.88));
        for (size_t i = 0; i < residual.data.size(); ++i) {
            if (p.tophat.data[i] >= hat_thr && p.smooth.data[i] < med && !sky.data[i]) {
                residual.data[i] = 255;
            }
        }
        residual = morph_close(residual, 1);
        // Dilate 1 only — dilate 2 was welding both ski poles into one box.
        residual = morph_dilate(residual, 1);
        p.thin = keep_elongated(residual, std::max(40, img_area / 2500), 3.2f);
        // Drop short tip fragments (ski tip / glove tip).
        {
            GrayImage thin_ok = make_gray(W, H, 0);
            auto ccl_t = ConnectedComponentLabeler::label(p.thin);
            for (const auto& c : ccl_t.components) {
                const float long_side = std::max(c.bbox.w, c.bbox.h);
                if (long_side < 28.0f || c.area < 40) {
                    continue;
                }
                paint_label(thin_ok, ccl_t.labels, W, H, c.label);
            }
            p.thin = thin_ok;
        }

        // One big slope field (optional context), not flecks.
        GrayImage slope = make_gray(W, H, 0);
        for (size_t i = 0; i < slope.data.size(); ++i) {
            if (!sky.data[i] && !dark.data[i] && p.smooth.data[i] >= med - 5) {
                slope.data[i] = 255;
            }
        }
        slope = morph_close(slope, 3);
        for (size_t i = 0; i < slope.data.size(); ++i) {
            if (p.body.data[i]) {
                slope.data[i] = 0;
            }
        }
        {
            auto ccl_g = ConnectedComponentLabeler::label(slope);
            for (const auto& c : ccl_g.components) {
                if (c.area < img_area * 12 / 100) {
                    continue;
                }
                paint_label(p.ground, ccl_g.labels, W, H, c.label);
            }
        }
    } else {
        // Only add polarity blobs if edge floods did not already separate the scene.
        int edge_n = 0;
        {
            auto ccl0 = ConnectedComponentLabeler::label(p.body);
            for (const auto& c : ccl0.components) {
                if (c.area >= min_region) {
                    ++edge_n;
                }
            }
        }
        if (edge_n < 3) {
            // Intensity floods without edge barriers (close-up bears / soft people).
            GrayImage flood_src = p.smooth;
            {
                contour::BilateralFilter soft;
                soft.radius = 5;
                soft.sigma_s = 3.5f;
                soft.sigma_r = 38.0f;
                flood_src = soft.apply(p.smooth);
            }
            std::vector<GrayImage> parts;
            for (float tol : {14.0f, 20.0f, 26.0f}) {
                parts = intensity_grid_segments(flood_src, 28, tol, min_region, max_region);
                if (parts.size() >= 2 && parts.size() <= 8) {
                    break;
                }
            }
            std::sort(parts.begin(), parts.end(), [](const GrayImage& a, const GrayImage& b) {
                int aa = 0, bb = 0;
                for (uint8_t v : a.data) {
                    aa += v > 0 ? 1 : 0;
                }
                for (uint8_t v : b.data) {
                    bb += v > 0 ? 1 : 0;
                }
                return aa > bb;
            });
            if (parts.size() > 5) {
                parts.resize(5);
            }
            for (auto& part : parts) {
                part = morph_close(part, 2);
                p.body = or_mask(p.body, morph_erode(part, 1));
            }

            GrayImage bright = make_gray(W, H, 0);
            const int bright_thr = std::max<int>(static_cast<int>(percentile_u8(
                std::vector<uint8_t>(p.smooth.data.begin(), p.smooth.data.end()), 0.68)),
                                                 static_cast<int>(med) + 12);
            for (size_t i = 0; i < bright.data.size(); ++i) {
                if (!sky.data[i] && static_cast<int>(p.smooth.data[i]) >= bright_thr) {
                    bright.data[i] = 255;
                }
            }
            bright = morph_close(bright, 2);
            {
                auto ccl_b = ConnectedComponentLabeler::label(bright);
                for (const auto& c : ccl_b.components) {
                    if (c.area < min_region || c.area > img_area * 25 / 100) {
                        continue;
                    }
                    GrayImage piece = make_gray(W, H, 0);
                    paint_label(piece, ccl_b.labels, W, H, c.label);
                    p.body = or_mask(p.body, morph_erode(piece, 1));
                }
            }
            GrayImage dark = make_gray(W, H, 0);
            const int dark_thr = static_cast<int>(p12) + 24;
            for (size_t i = 0; i < dark.data.size(); ++i) {
                if (!sky.data[i] && static_cast<int>(p.smooth.data[i]) <= dark_thr) {
                    dark.data[i] = 255;
                }
            }
            dark = morph_open(morph_close(dark, 2), 1);
            {
                auto ccl_d = ConnectedComponentLabeler::label(dark);
                for (const auto& c : ccl_d.components) {
                    if (c.area < min_region || c.area > img_area * 30 / 100) {
                        continue;
                    }
                    GrayImage piece = make_gray(W, H, 0);
                    paint_label(piece, ccl_d.labels, W, H, c.label);
                    p.body = or_mask(p.body, morph_erode(piece, 1));
                }
            }
        }
    }

    // Final body cleanup: drop leftover flecks; keep sizable CCs only.
    {
        auto ccl_b = ConnectedComponentLabeler::label(p.body);
        GrayImage cleaned = make_gray(W, H, 0);
        for (const auto& c : ccl_b.components) {
            if (c.area < min_region) {
                continue;
            }
            const float short_side = std::min(c.bbox.w, c.bbox.h);
            const float aspect = (c.bbox.h > 1.0f) ? (c.bbox.w / c.bbox.h) : 99.0f;
            const float long_aspect =
                std::max(aspect, (c.bbox.w > 1.0f) ? (c.bbox.h / c.bbox.w) : 99.0f);
            if (short_side <= 5.0f && long_aspect >= 4.0f) {
                continue;  // crack leftovers
            }
            paint_label(cleaned, ccl_b.labels, W, H, c.label);
        }
        p.body = cleaned;
    }

    for (size_t i = 0; i < p.binary.data.size(); ++i) {
        if (sky.data[i]) {
            p.body.data[i] = 0;
            p.thin.data[i] = 0;
        }
        if (p.body.data[i] || p.thin.data[i] || p.ground.data[i]) {
            p.binary.data[i] = 255;
        }
    }
    (void)p85;
    return p;
}

inline GrayImage thick_parts(const GrayImage& binary) {
    return morph_open(binary, 2);
}

inline GrayImage thin_parts(const GrayImage& binary) {
    const auto thick = thick_parts(binary);
    GrayImage thin = make_gray(binary.width, binary.height, 0);
    for (size_t i = 0; i < thin.data.size(); ++i) {
        if (binary.data[i] && !thick.data[i]) {
            thin.data[i] = 255;
        }
    }
    return morph_dilate(thin, 1);
}

// Split a large body mask using intensity watershed when morphology can't.
inline std::vector<GrayImage> split_instances_watershed(const GrayImage& binary,
                                                        const GrayImage& smooth) {
    std::vector<GrayImage> out;
    if (binary.empty()) {
        return out;
    }
    const int W = binary.width;
    const int H = binary.height;
    const int canvas = std::max(1, W * H);
    int fg = 0;
    for (uint8_t v : binary.data) {
        fg += v > 0 ? 1 : 0;
    }

    // Intensity basins restricted to this blob.
    if (fg >= canvas * 8 / 100 && !smooth.empty()) {
        auto ws = contour::Watershed::from_gradient(smooth);
        for (int lab = 1; lab <= ws.n_basins; ++lab) {
            GrayImage piece = make_gray(W, H, 0);
            int area = 0;
            for (size_t i = 0; i < ws.labels.size(); ++i) {
                if (ws.labels[i] == lab && binary.data[i]) {
                    piece.data[i] = 255;
                    ++area;
                }
            }
            if (area >= std::max(80, canvas * 3 / 100) && area <= canvas * 50 / 100) {
                out.push_back(std::move(piece));
            }
        }
        if (out.size() >= 2) {
            return out;
        }
        out.clear();
    }

    // Morphological erode→seed→geodesic grow.
    for (int r = 2; r <= 10; ++r) {
        auto seeds = morph_erode(binary, r);
        auto ccl = ConnectedComponentLabeler::label(seeds);
        int kept = 0;
        for (const auto& c : ccl.components) {
            if (c.area >= 20) {
                ++kept;
            }
        }
        if (kept < 2) {
            continue;
        }
        for (const auto& c : ccl.components) {
            if (c.area < 20) {
                continue;
            }
            GrayImage piece = make_gray(W, H, 0);
            paint_label(piece, ccl.labels, W, H, c.label);
            for (int iter = 0; iter < r * 5 + 8; ++iter) {
                auto grown = morph_dilate(piece, 1);
                for (size_t i = 0; i < piece.data.size(); ++i) {
                    piece.data[i] = (grown.data[i] && binary.data[i]) ? 255 : 0;
                }
            }
            int area = 0;
            for (uint8_t v : piece.data) {
                area += v > 0 ? 1 : 0;
            }
            if (area >= std::max(60, canvas / 250)) {
                out.push_back(std::move(piece));
            }
        }
        if (out.size() >= 2) {
            return out;
        }
        out.clear();
    }

    out.push_back(binary);
    return out;
}

inline std::vector<GrayImage> split_instances_watershed(const GrayImage& binary) {
    GrayImage empty;
    return split_instances_watershed(binary, empty);
}

}  // namespace vision
