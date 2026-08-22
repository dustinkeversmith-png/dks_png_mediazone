#pragma once

#include "types.hpp"
#include <cctype>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <cstdlib>

namespace contour {

struct CocoInstance {
    int category_id = 0;
    float area = 0.0f;
    Rect bbox;
    std::vector<Vec2> polygon;
};

struct CocoSample {
    std::string file;
    int width = 0;
    int height = 0;
    std::vector<CocoInstance> instances;
};

inline std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return s;
}

inline void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
}

inline bool parse_number_list(const std::string& s, size_t& i, std::vector<float>& out) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[') {
        return false;
    }
    ++i;
    out.clear();
    while (i < s.size()) {
        skip_ws(s, i);
        if (i < s.size() && s[i] == ']') {
            ++i;
            return true;
        }
        if (i < s.size() && s[i] == '[') {
            std::vector<float> inner;
            if (!parse_number_list(s, i, inner)) {
                return false;
            }
            out.insert(out.end(), inner.begin(), inner.end());
        } else {
            char* end = nullptr;
            const float v = std::strtof(s.c_str() + i, &end);
            if (end == s.c_str() + i) {
                ++i;
                continue;
            }
            out.push_back(v);
            i = static_cast<size_t>(end - s.c_str());
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
        }
    }
    return false;
}

inline CocoSample parse_coco_json(const std::string& path) {
    CocoSample sample;
    const std::string s = read_file(path);
    auto find_key = [&](const char* key, size_t from = 0) -> size_t {
        const std::string k = std::string("\"") + key + "\"";
        const size_t p = s.find(k, from);
        if (p == std::string::npos) {
            return std::string::npos;
        }
        size_t c = s.find(':', p + k.size());
        return c == std::string::npos ? std::string::npos : c + 1;
    };
    size_t p = find_key("width");
    if (p != std::string::npos) {
        sample.width = std::atoi(s.c_str() + p);
    }
    p = find_key("height");
    if (p != std::string::npos) {
        sample.height = std::atoi(s.c_str() + p);
    }

    size_t inst = s.find("\"instances\"");
    size_t cursor = inst == std::string::npos ? 0 : inst;
    while (true) {
        const size_t poly_k = find_key("polygon", cursor);
        const size_t bbox_k = find_key("bbox", cursor);
        if (poly_k == std::string::npos && bbox_k == std::string::npos) {
            break;
        }
        CocoInstance inst_obj;
        size_t cat = find_key("category_id", cursor);
        if (cat != std::string::npos && cat < (poly_k == std::string::npos ? s.size() : poly_k) + 64) {
            inst_obj.category_id = std::atoi(s.c_str() + cat);
        }
        if (bbox_k != std::string::npos) {
            size_t i = bbox_k;
            std::vector<float> b;
            if (parse_number_list(s, i, b) && b.size() >= 4) {
                inst_obj.bbox = {b[0], b[1], b[2], b[3]};
            }
        }
        if (poly_k != std::string::npos) {
            size_t i = poly_k;
            std::vector<float> xy;
            if (parse_number_list(s, i, xy)) {
                for (size_t k = 0; k + 1 < xy.size(); k += 2) {
                    inst_obj.polygon.push_back({xy[k], xy[k + 1]});
                }
            }
            cursor = i;
        } else {
            cursor = bbox_k + 1;
        }
        if (!inst_obj.polygon.empty() || inst_obj.bbox.w > 0) {
            sample.instances.push_back(std::move(inst_obj));
        }
        if (cursor >= s.size()) {
            break;
        }
    }
    return sample;
}

}  // namespace contour
