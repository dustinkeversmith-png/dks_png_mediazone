#pragma once

// In-test object proposals: edges + chrominance saliency (not bright-luma FG).

#include "math/contour_compat.hpp"
#include "filters/edge/canny/canny.hpp"
#include "filters/lab_color/lab_color_space.hpp"
#include "filters/sobel/sobel.hpp"
#include "segmentation/ccl/connected_components.hpp"
#include "segmentation/watershed/watershed.hpp"
#include "sdf/edt/edt.hpp"

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

// Flood-fill background from image border; remaining enclosed regions = objects.
inline GrayImage fill_holes_from_border(const GrayImage& edges) {
    GrayImage mark = edges;
    std::queue<std::pair<int, int>> q;
    auto push = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= mark.width || y >= mark.height) {
            return;
        }
        if (mark.at(x, y) != 0) {
            return;
        }
        mark.at(x, y) = 128;  // visited background
        q.push({x, y});
    };
    for (int x = 0; x < mark.width; ++x) {
        push(x, 0);
        push(x, mark.height - 1);
    }
    for (int y = 0; y < mark.height; ++y) {
        push(0, y);
        push(mark.width - 1, y);
    }
    static const int dx[4] = {1, -1, 0, 0};
    static const int dy[4] = {0, 0, 1, -1};
    while (!q.empty()) {
        const auto [x, y] = q.front();
        q.pop();
        for (int k = 0; k < 4; ++k) {
            push(x + dx[k], y + dy[k]);
        }
    }
    GrayImage out = make_gray(edges.width, edges.height, 0);
    for (size_t i = 0; i < out.data.size(); ++i) {
        // Enclosed voids + edge pixels become FG objects.
        out.data[i] = (mark.data[i] == 0 || edges.data[i] > 0) ? 255 : 0;
    }
    return out;
}

// Chrominance saliency: |a|+|b| in Lab, with extra boost for red-ish hues.
inline GrayImage chrominance_saliency(const GrayImage& rgb_or_gray) {
    GrayImage out = make_gray(rgb_or_gray.width, rgb_or_gray.height, 0);
    float maxv = 1e-6f;
    std::vector<float> tmp(out.data.size(), 0.0f);
    for (int y = 0; y < rgb_or_gray.height; ++y) {
        for (int x = 0; x < rgb_or_gray.width; ++x) {
            const auto lab = contour::LabColor::at(rgb_or_gray, x, y);
            float s = std::fabs(lab.a) + std::fabs(lab.b);
            // Red stop-sign boost (positive a*, modest b*).
            if (lab.a > 20.0f && lab.L > 20.0f && lab.L < 80.0f) {
                s += lab.a;
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

// Sky / bright-specular suppressor: high L, low chroma.
inline GrayImage sky_mask(const GrayImage& rgb_or_gray, float l_thr = 75.0f, float c_thr = 12.0f) {
    GrayImage out = make_gray(rgb_or_gray.width, rgb_or_gray.height, 0);
    for (int y = 0; y < rgb_or_gray.height; ++y) {
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
    GrayImage edges;
    GrayImage saliency;
};

// Invert polarity when bright pixels dominate FG (sky/gap selected).
inline GrayImage ensure_dark_object_polarity(const GrayImage& luma, const GrayImage& binary) {
    double fg_mean = 0, bg_mean = 0;
    int fg_n = 0, bg_n = 0;
    for (size_t i = 0; i < binary.data.size() && i < luma.data.size(); ++i) {
        if (binary.data[i] > 127) {
            fg_mean += luma.data[i];
            ++fg_n;
        } else {
            bg_mean += luma.data[i];
            ++bg_n;
        }
    }
    if (fg_n < 8 || bg_n < 8) {
        return binary;
    }
    fg_mean /= fg_n;
    bg_mean /= bg_n;
    if (fg_mean <= bg_mean + 8.0) {
        return binary;  // already darker-or-equal FG
    }
    GrayImage inv = binary;
    for (uint8_t& p : inv.data) {
        p = p ? 0 : 255;
    }
    return inv;
}

// Build FG proposals from photo RGB/luma without treating bright sky as objects.
inline ObjectProposal propose_objects_from_photo(const GrayImage& rgb_or_luma) {
    ObjectProposal p;
    contour::Canny canny;
    canny.low = 0.06f;
    canny.high = 0.18f;
    p.edges = canny.detect(rgb_or_luma);
    p.edges = morph_close(p.edges, 1);

    p.saliency = chrominance_saliency(rgb_or_luma);
    const auto sky = sky_mask(rgb_or_luma);

    // Collect non-sky luma stats → prefer darker class as FG (trees, signs, asphalt objects).
    std::vector<uint8_t> nonsky;
    nonsky.reserve(static_cast<size_t>(rgb_or_luma.width * rgb_or_luma.height));
    for (int y = 0; y < rgb_or_luma.height; ++y) {
        for (int x = 0; x < rgb_or_luma.width; ++x) {
            const size_t i = static_cast<size_t>(y * rgb_or_luma.width + x);
            if (sky.data[i] == 0) {
                nonsky.push_back(static_cast<uint8_t>(rgb_or_luma.gray(x, y)));
            }
        }
    }
    uint8_t thr = 127;
    if (!nonsky.empty()) {
        auto sorted = nonsky;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        thr = sorted[sorted.size() / 2];
    }

    GrayImage dark = make_gray(rgb_or_luma.width, rgb_or_luma.height, 0);
    GrayImage sal_bin = make_gray(rgb_or_luma.width, rgb_or_luma.height, 0);
    for (int y = 0; y < rgb_or_luma.height; ++y) {
        for (int x = 0; x < rgb_or_luma.width; ++x) {
            const size_t i = static_cast<size_t>(y * rgb_or_luma.width + x);
            if (sky.data[i]) {
                continue;
            }
            const float g = rgb_or_luma.gray(x, y);
            if (g <= static_cast<float>(thr) + 5.0f) {
                dark.data[i] = 255;
            }
            if (p.saliency.data[i] > 90) {
                sal_bin.data[i] = 255;
            }
        }
    }
    dark = morph_close(dark, 1);
    sal_bin = morph_close(sal_bin, 1);

    // Edge rings help close thin structures (sign borders, limbs).
    auto edge_band = morph_dilate(p.edges, 1);

    GrayImage luma = make_gray(rgb_or_luma.width, rgb_or_luma.height, 0);
    for (int y = 0; y < luma.height; ++y) {
        for (int x = 0; x < luma.width; ++x) {
            luma.at(x, y) = static_cast<uint8_t>(rgb_or_luma.gray(x, y));
        }
    }

    p.binary = make_gray(rgb_or_luma.width, rgb_or_luma.height, 0);
    for (size_t i = 0; i < p.binary.data.size(); ++i) {
        if (sky.data[i]) {
            continue;
        }
        if (dark.data[i] || sal_bin.data[i] || edge_band.data[i]) {
            p.binary.data[i] = 255;
        }
    }
    p.binary = ensure_dark_object_polarity(luma, p.binary);

    auto ccl = ConnectedComponentLabeler::label(p.binary);
    p.binary.data.assign(p.binary.data.size(), 0);
    const int min_area = std::max(64, (rgb_or_luma.width * rgb_or_luma.height) / 500);
    for (const auto& c : ccl.components) {
        if (c.area < min_area) {
            continue;
        }
        // Reject components that are mostly bright (residual sky gaps).
        double mean = 0.0;
        int n = 0;
        for (int y = 0; y < ccl.height; ++y) {
            for (int x = 0; x < ccl.width; ++x) {
                if (ccl.labels[static_cast<size_t>(y * ccl.width + x)] == c.label) {
                    mean += rgb_or_luma.gray(x, y);
                    ++n;
                }
            }
        }
        if (n > 0) {
            mean /= n;
        }
        if (mean > 200.0) {
            continue;
        }
        for (int y = 0; y < ccl.height; ++y) {
            for (int x = 0; x < ccl.width; ++x) {
                if (ccl.labels[static_cast<size_t>(y * ccl.width + x)] == c.label) {
                    p.binary.at(x, y) = 255;
                }
            }
        }
    }
    return p;
}

// Split touching instances via marker-controlled watershed on -EDT.
inline std::vector<GrayImage> split_instances_watershed(const GrayImage& binary) {
    std::vector<GrayImage> out;
    if (binary.empty()) {
        return out;
    }
    auto ws = contour::Watershed::from_mask(binary);
    if (ws.n_basins <= 1) {
        out.push_back(binary);
        return out;
    }
    for (int lab = 1; lab <= ws.n_basins; ++lab) {
        GrayImage piece = make_gray(ws.width, ws.height, 0);
        int area = 0;
        for (size_t i = 0; i < ws.labels.size(); ++i) {
            if (ws.labels[i] == lab) {
                piece.data[i] = 255;
                ++area;
            }
        }
        if (area >= 32) {
            out.push_back(std::move(piece));
        }
    }
    if (out.empty()) {
        out.push_back(binary);
    }
    return out;
}

}  // namespace vision
