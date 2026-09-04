#pragma once

#include "math/contour_compat.hpp"
#include "segmentation/helpers/ccl/connected_components.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace contour {

struct CropWindow {
    ImageBuffer image;
    int origin_x = 0;
    int origin_y = 0;
    int parent_w = 0;
    int parent_h = 0;
    Rect local_bbox;
};

// Axis-aligned bbox helper built on CCL (largest component) + extrema scan.
// Pipeline (recipe): binary mask → morph open → CCL → largest CC → [min,max] bounds.
class BBoxAuto {
public:
    // Scan non-zero pixels for extrema AABB.
    static Rect from_mask(const ImageBuffer& mask, uint8_t thr = 127) {
        int x0 = mask.width, y0 = mask.height, x1 = 0, y1 = 0;
        for (int y = 0; y < mask.height; ++y) {
            for (int x = 0; x < mask.width; ++x) {
                if (mask.at(x, y) > thr) {
                    x0 = std::min(x0, x);
                    y0 = std::min(y0, y);
                    x1 = std::max(x1, x);
                    y1 = std::max(y1, y);
                }
            }
        }
        if (x1 < x0) {
            return {0, 0, static_cast<float>(mask.width), static_cast<float>(mask.height)};
        }
        return {static_cast<float>(x0), static_cast<float>(y0),
                static_cast<float>(x1 - x0 + 1), static_cast<float>(y1 - y0 + 1)};
    }

    // Morph-open → CCL → keep largest component → AABB.
    // Optional watershed split of touching blobs before taking the largest piece.
    static ImageBuffer largest_component_mask(const ImageBuffer& binary, uint8_t thr = 127,
                                             int open_r = 1) {
        ImageBuffer mask = binary;
        if (mask.channels != 1) {
            mask = make_gray(binary.width, binary.height, 0);
            for (int y = 0; y < binary.height; ++y) {
                for (int x = 0; x < binary.width; ++x) {
                    mask.at(x, y) = static_cast<uint8_t>(binary.gray(x, y) > thr ? 255 : 0);
                }
            }
        } else {
            for (uint8_t& p : mask.data) {
                p = p > thr ? 255 : 0;
            }
        }
        if (open_r > 0) {
            mask = morph_open_u8(mask, open_r);
        }
        auto ccl = vision::ConnectedComponentLabeler::label(mask);
        int best = -1;
        int best_area = 0;
        for (const auto& c : ccl.components) {
            if (c.area > best_area) {
                best_area = c.area;
                best = c.label;
            }
        }
        ImageBuffer out = make_gray(mask.width, mask.height, 0);
        if (best < 0) {
            return out;
        }
        for (size_t i = 0; i < ccl.labels.size(); ++i) {
            if (ccl.labels[i] == best) {
                out.data[i] = 255;
            }
        }
        return out;
    }

    static Rect from_binary_via_ccl(const ImageBuffer& binary, uint8_t thr = 127, int open_r = 1) {
        return from_mask(largest_component_mask(binary, thr, open_r));
    }

    static Rect pad(const Rect& b, float margin, int w, int h) {
        const float px = std::max(16.0f, b.w * margin);
        const float py = std::max(16.0f, b.h * margin);
        Rect r{b.x - px, b.y - py, b.w + 2 * px, b.h + 2 * py};
        r.x = std::clamp(r.x, 0.0f, static_cast<float>(std::max(0, w - 1)));
        r.y = std::clamp(r.y, 0.0f, static_cast<float>(std::max(0, h - 1)));
        r.w = std::clamp(r.w, 1.0f, static_cast<float>(w) - r.x);
        r.h = std::clamp(r.h, 1.0f, static_cast<float>(h) - r.y);
        return r;
    }

    static CropWindow crop(const ImageBuffer& src, const Rect& bbox, float margin = 0.12f) {
        const Rect r = pad(bbox, margin, src.width, src.height);
        CropWindow c;
        c.origin_x = static_cast<int>(std::floor(r.x));
        c.origin_y = static_cast<int>(std::floor(r.y));
        c.parent_w = src.width;
        c.parent_h = src.height;
        const int cw = std::min(src.width - c.origin_x, std::max(1, static_cast<int>(std::ceil(r.w))));
        const int ch = std::min(src.height - c.origin_y, std::max(1, static_cast<int>(std::ceil(r.h))));
        c.image.width = cw;
        c.image.height = ch;
        c.image.channels = src.channels;
        c.image.data.assign(static_cast<size_t>(cw * ch * src.channels), 0);
        for (int y = 0; y < ch; ++y) {
            for (int x = 0; x < cw; ++x) {
                const int sx = std::clamp(c.origin_x + x, 0, src.width - 1);
                const int sy = std::clamp(c.origin_y + y, 0, src.height - 1);
                for (int k = 0; k < src.channels; ++k) {
                    c.image.at(x, y, k) = src.at(sx, sy, k);
                }
            }
        }
        c.local_bbox = {bbox.x - static_cast<float>(c.origin_x), bbox.y - static_cast<float>(c.origin_y), bbox.w,
                        bbox.h};
        c.local_bbox.x = std::clamp(c.local_bbox.x, 0.0f, static_cast<float>(cw));
        c.local_bbox.y = std::clamp(c.local_bbox.y, 0.0f, static_cast<float>(ch));
        c.local_bbox.w = std::clamp(c.local_bbox.w, 1.0f, static_cast<float>(cw) - c.local_bbox.x);
        c.local_bbox.h = std::clamp(c.local_bbox.h, 1.0f, static_cast<float>(ch) - c.local_bbox.y);
        return c;
    }

    static ImageBuffer uncrop(const ImageBuffer& cropped, const CropWindow& win) {
        ImageBuffer full = make_gray(win.parent_w, win.parent_h, 0);
        for (int y = 0; y < cropped.height; ++y) {
            for (int x = 0; x < cropped.width; ++x) {
                const int dx = win.origin_x + x;
                const int dy = win.origin_y + y;
                if (dx >= 0 && dy >= 0 && dx < full.width && dy < full.height) {
                    full.at(dx, dy) = cropped.at(x, y);
                }
            }
        }
        return full;
    }

    static Rect dilated_ellipse_box(const Rect& b, float grow = 0.06f) {
        return {b.x - b.w * grow, b.y - b.h * grow, b.w * (1.0f + 2.0f * grow),
                b.h * (1.0f + 2.0f * grow)};
    }

private:
    static ImageBuffer morph_open_u8(const ImageBuffer& src, int r) {
        // Prefer morph_clean helpers when available; fall back to simple square open.
        ImageBuffer eroded = make_gray(src.width, src.height, 0);
        ImageBuffer opened = make_gray(src.width, src.height, 0);
        auto erode_at = [&](int x, int y) {
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= src.width || yy >= src.height) {
                        return static_cast<uint8_t>(0);
                    }
                    if (src.at(xx, yy) == 0) {
                        return static_cast<uint8_t>(0);
                    }
                }
            }
            return static_cast<uint8_t>(255);
        };
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                eroded.at(x, y) = erode_at(x, y);
            }
        }
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                uint8_t v = 0;
                for (int dy = -r; dy <= r && !v; ++dy) {
                    for (int dx = -r; dx <= r; ++dx) {
                        const int xx = x + dx;
                        const int yy = y + dy;
                        if (xx >= 0 && yy >= 0 && xx < src.width && yy < src.height &&
                            eroded.at(xx, yy)) {
                            v = 255;
                            break;
                        }
                    }
                }
                opened.at(x, y) = v;
            }
        }
        return opened;
    }
};

}  // namespace contour
