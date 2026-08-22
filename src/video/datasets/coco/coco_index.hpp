#pragma once

#include "../io/vision_io.hpp"
#include "../../modules/math/polygon.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace datasets {

struct CocoImageMeta {
    int id = 0;
    int width = 0;
    int height = 0;
    std::string file_name;
};

struct CocoAnnIndex {
    std::unordered_map<std::string, CocoImageMeta> by_stem;
    std::unordered_map<int, std::vector<std::vector<math::Vec2>>> polygons_by_image;

    static std::string stem_of(const std::string& file_name) {
        const auto slash = file_name.find_last_of("/\\");
        std::string base = slash == std::string::npos ? file_name : file_name.substr(slash + 1);
        const auto dot = base.find_last_of('.');
        if (dot != std::string::npos) {
            base = base.substr(0, dot);
        }
        return base;
    }

    static bool parse_polygon_flat(const std::string& text, size_t& i, std::vector<math::Vec2>& poly) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
        if (i >= text.size() || text[i] != '[') {
            return false;
        }
        ++i;
        std::vector<float> nums;
        while (i < text.size()) {
            while (i < text.size() && (std::isspace(static_cast<unsigned char>(text[i])) || text[i] == ',')) {
                ++i;
            }
            if (i < text.size() && text[i] == ']') {
                ++i;
                break;
            }
            if (i < text.size() && text[i] == '[') {
                std::vector<math::Vec2> inner;
                if (!parse_polygon_flat(text, i, inner)) {
                    return false;
                }
                poly = std::move(inner);
                while (i < text.size() && text[i] != ']') {
                    ++i;
                }
                if (i < text.size() && text[i] == ']') {
                    ++i;
                }
                return !poly.empty();
            }
            char* end = nullptr;
            const float v = std::strtof(text.c_str() + i, &end);
            if (end == text.c_str() + i) {
                ++i;
                continue;
            }
            nums.push_back(v);
            i = static_cast<size_t>(end - text.c_str());
        }
        poly.clear();
        for (size_t k = 0; k + 1 < nums.size(); k += 2) {
            poly.push_back({nums[k], nums[k + 1]});
        }
        return !poly.empty();
    }

    static CocoAnnIndex load(const std::string& json_path) {
        CocoAnnIndex idx;
        const std::string text = vision::read_text_file(json_path);
        if (text.empty()) {
            return idx;
        }

        size_t pos = 0;
        while ((pos = text.find("\"images\"", pos)) != std::string::npos) {
            pos = text.find('[', pos);
            if (pos == std::string::npos) {
                break;
            }
            ++pos;
            while (pos < text.size()) {
                pos = text.find("\"id\"", pos);
                if (pos == std::string::npos || pos > text.find("\"annotations\"", pos)) {
                    break;
                }
                const size_t id_colon = text.find(':', pos);
                if (id_colon == std::string::npos) {
                    break;
                }
                CocoImageMeta meta;
                meta.id = std::atoi(text.c_str() + id_colon + 1);

                const size_t fn_key = text.find("\"file_name\"", pos);
                if (fn_key != std::string::npos && fn_key < pos + 512) {
                    const size_t q0 = text.find('"', fn_key + 12);
                    const size_t q1 = text.find('"', q0 + 1);
                    if (q0 != std::string::npos && q1 != std::string::npos) {
                        meta.file_name = text.substr(q0 + 1, q1 - q0 - 1);
                    }
                }
                const size_t w_key = text.find("\"width\"", pos);
                if (w_key != std::string::npos && w_key < pos + 512) {
                    meta.width = std::atoi(text.c_str() + text.find(':', w_key) + 1);
                }
                const size_t h_key = text.find("\"height\"", pos);
                if (h_key != std::string::npos && h_key < pos + 512) {
                    meta.height = std::atoi(text.c_str() + text.find(':', h_key) + 1);
                }
                if (!meta.file_name.empty()) {
                    idx.by_stem[stem_of(meta.file_name)] = meta;
                }
                pos = text.find('}', pos) + 1;
                if (text[pos] == ']') {
                    break;
                }
            }
            break;
        }

        pos = 0;
        while ((pos = text.find("\"segmentation\"", pos)) != std::string::npos) {
            const size_t img_key = text.rfind("\"image_id\"", pos);
            if (img_key == std::string::npos || pos - img_key > 800) {
                pos += 14;
                continue;
            }
            const int image_id = std::atoi(text.c_str() + text.find(':', img_key) + 1);
            size_t seg_i = pos + 14;
            std::vector<math::Vec2> poly;
            if (parse_polygon_flat(text, seg_i, poly)) {
                idx.polygons_by_image[image_id].push_back(std::move(poly));
            }
            pos = seg_i;
        }
        return idx;
    }

    math::ImageBuffer mask_for_stem(const std::string& stem, int fallback_w, int fallback_h) const {
        math::ImageBuffer mask;
        auto it = by_stem.find(stem);
        if (it == by_stem.end()) {
            return mask;
        }
        const int w = it->second.width > 0 ? it->second.width : fallback_w;
        const int h = it->second.height > 0 ? it->second.height : fallback_h;
        auto pit = polygons_by_image.find(it->second.id);
        if (pit == polygons_by_image.end()) {
            return mask;
        }
        mask = math::make_gray(w, h, 0);
        for (const auto& poly : pit->second) {
            if (poly.size() < 3) {
                continue;
            }
            const auto piece = math::rasterize_polygon(poly, w, h);
            for (size_t i = 0; i < mask.data.size() && i < piece.data.size(); ++i) {
                if (piece.data[i] > 0) {
                    mask.data[i] = 255;
                }
            }
        }
        return mask;
    }
};

}  // namespace datasets
