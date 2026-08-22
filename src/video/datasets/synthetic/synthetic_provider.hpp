#pragma once

#include "../dataset_provider.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace datasets {

enum class SyntheticKind {
    Grids,
    Glyphs,
    Silhouettes,
    Leaf,
    Boxes,
};

inline SyntheticKind synthetic_kind_from_name(const std::string& name) {
    if (name == "synthetic_grids" || name == "synthetic") {
        return SyntheticKind::Grids;
    }
    if (name == "synthetic_glyphs" || name == "char74k") {
        return SyntheticKind::Glyphs;
    }
    if (name == "synthetic_silhouettes" || name == "mpeg7" || name == "kimia99") {
        return SyntheticKind::Silhouettes;
    }
    if (name == "synthetic_leaf" || name == "swedish_leaf") {
        return SyntheticKind::Leaf;
    }
    if (name == "synthetic_boxes") {
        return SyntheticKind::Boxes;
    }
    return SyntheticKind::Grids;
}

class SyntheticProvider : public DatasetProvider {
public:
    SyntheticProvider(std::filesystem::path root, SyntheticKind kind) : DatasetProvider(std::move(root)), kind_(kind) {}

    bool initialize() override {
        entries_.clear();
        switch (kind_) {
        case SyntheticKind::Glyphs:
            entries_ = {"glyph_B", "glyph_O", "glyph_8", "glyph_1"};
            break;
        case SyntheticKind::Leaf:
            entries_ = {"leaf_0", "leaf_1", "leaf_2"};
            break;
        case SyntheticKind::Silhouettes:
            entries_ = {"silhouette_star", "silhouette_blob", "silhouette_crescent"};
            break;
        case SyntheticKind::Boxes:
            entries_ = {"boxes_32"};
            break;
        case SyntheticKind::Grids:
        default:
            entries_ = {"donut", "nested_donut", "concave_star", "sine_band"};
            break;
        }
        return !entries_.empty();
    }

    size_t size() const override { return entries_.size(); }

    VisionSample load_sample(size_t idx) const override {
        VisionSample s;
        if (idx >= entries_.size()) {
            return s;
        }
        s.id = entries_[idx];
        s.label = entries_[idx];
        s.image_path = "synthetic://" + s.id;

        switch (kind_) {
        case SyntheticKind::Glyphs:
            s.mask = render_glyph(s.id, 64, 64);
            break;
        case SyntheticKind::Leaf:
            s.mask = render_leaf(static_cast<int>(idx), 128, 128);
            break;
        case SyntheticKind::Silhouettes:
            s.mask = render_silhouette(s.id, 128, 128);
            break;
        case SyntheticKind::Boxes:
            s.mask = math::make_gray(512, 512, 0);
            s.label = "box_list";
            break;
        case SyntheticKind::Grids:
        default:
            s.mask = render_grid(s.id, 128, 128);
            break;
        }
        s.luma = s.mask;
        s.rgb = s.mask;
        s.rgb.channels = 1;
        return s;
    }

    std::vector<math::Rect> load_boxes(size_t idx) const {
        std::vector<math::Rect> boxes;
        if (kind_ != SyntheticKind::Boxes || idx >= entries_.size()) {
            return boxes;
        }
        const int n = 32;
        for (int i = 0; i < n; ++i) {
            const float x = static_cast<float>((i * 37) % 400);
            const float y = static_cast<float>((i * 53) % 400);
            const float w = 30.0f + static_cast<float>((i * 11) % 40);
            const float h = 20.0f + static_cast<float>((i * 17) % 50);
            boxes.push_back({x, y, w, h});
        }
        return boxes;
    }

private:
    SyntheticKind kind_;

    static math::ImageBuffer render_grid(const std::string& id, int w, int h) {
        if (id == "donut") {
            return make_synthetic_donut(w, h);
        }
        if (id == "nested_donut") {
            auto outer = make_synthetic_disk(w, h, w / 2, h / 2, w / 3);
            auto inner = make_synthetic_disk(w, h, w / 2, h / 2, w / 8);
            for (size_t i = 0; i < outer.data.size(); ++i) {
                if (inner.data[i] > 0) {
                    outer.data[i] = 128;
                }
            }
            return outer;
        }
        if (id == "concave_star") {
            auto im = math::make_gray(w, h, 0);
            const math::Vec2 c{static_cast<float>(w / 2), static_cast<float>(h / 2)};
            for (int i = 0; i < 5; ++i) {
                const float ang = static_cast<float>(i) * 2.0f * 3.14159265f / 5.0f;
                const math::Vec2 p{c.x + 45.0f * std::cos(ang), c.y + 45.0f * std::sin(ang)};
                auto blob = make_synthetic_disk(w, h, static_cast<int>(p.x), static_cast<int>(p.y), 14);
                for (size_t j = 0; j < im.data.size(); ++j) {
                    if (blob.data[j] > 0) {
                        im.data[j] = 255;
                    }
                }
            }
            return im;
        }
        auto im = math::make_gray(w, h, 0);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const float t = static_cast<float>(x) / static_cast<float>(w) * 6.2831853f;
                const float yy = static_cast<float>(h) * 0.5f + 20.0f * std::sin(t * 3.0f);
                if (std::fabs(static_cast<float>(y) - yy) < 2.5f) {
                    im.at(x, y) = 255;
                }
            }
        }
        return im;
    }

    static void stroke_disk(math::ImageBuffer& im, int cx, int cy, int r, uint8_t v) {
        for (int y = 0; y < im.height; ++y) {
            for (int x = 0; x < im.width; ++x) {
                const int dx = x - cx;
                const int dy = y - cy;
                if (dx * dx + dy * dy <= r * r) {
                    im.at(x, y) = v;
                }
            }
        }
    }

    static math::ImageBuffer render_glyph(const std::string& id, int w, int h) {
        auto im = math::make_gray(w, h, 0);
        const int cx = w / 2;
        const int cy = h / 2;
        if (id == "glyph_O") {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int dx = x - cx;
                    const int dy = y - cy;
                    const int d2 = dx * dx + dy * dy;
                    if (d2 <= 20 * 20 && d2 >= 10 * 10) {
                        im.at(x, y) = 255;
                    }
                }
            }
        } else if (id == "glyph_B") {
            for (int y = cy - 22; y <= cy + 22; ++y) {
                if (y >= 0 && y < h) {
                    im.at(cx - 14, y) = 255;
                }
            }
            stroke_disk(im, cx + 2, cy - 10, 10, 255);
            stroke_disk(im, cx + 2, cy + 10, 10, 255);
        } else if (id == "glyph_8") {
            stroke_disk(im, cx, cy - 10, 11, 255);
            stroke_disk(im, cx, cy + 10, 11, 255);
            for (int y = cy - 10; y <= cy + 10; ++y) {
                if (y >= 0 && y < h) {
                    im.at(cx, y) = 255;
                }
            }
        } else if (id == "glyph_1") {
            for (int y = cy - 24; y <= cy + 24; ++y) {
                if (y >= 0 && y < h) {
                    im.at(cx, y) = 255;
                }
            }
        }
        return im;
    }

    static math::ImageBuffer render_leaf(int variant, int w, int h) {
        auto im = math::make_gray(w, h, 0);
        const float cx = static_cast<float>(w) * 0.5f;
        const float cy = static_cast<float>(h) * 0.55f;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const float nx = (static_cast<float>(x) - cx) / (w * 0.35f);
                const float ny = (static_cast<float>(y) - cy) / (h * 0.45f);
                const float angle = std::atan2(ny, nx);
                const float r = std::sqrt(nx * nx + ny * ny);
                const float lobes = 1.0f + 0.35f * std::cos(angle * (3.0f + static_cast<float>(variant)));
                if (r < lobes && ny < 0.35f) {
                    im.at(x, y) = 255;
                }
            }
        }
        return im;
    }

    static math::ImageBuffer render_silhouette(const std::string& id, int w, int h) {
        if (id == "silhouette_star") {
            return render_grid("concave_star", w, h);
        }
        if (id == "silhouette_crescent") {
            auto a = make_synthetic_disk(w, h, w / 2 + 8, h / 2, w / 3);
            auto b = make_synthetic_disk(w, h, w / 2 - 8, h / 2, w / 3);
            for (size_t i = 0; i < a.data.size(); ++i) {
                a.data[i] = static_cast<uint8_t>((a.data[i] > 0 && b.data[i] == 0) ? 255 : 0);
            }
            return a;
        }
        return make_synthetic_disk(w, h, w / 2, h / 2, w / 4);
    }
};

}  // namespace datasets
