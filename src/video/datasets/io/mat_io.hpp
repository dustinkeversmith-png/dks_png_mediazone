#pragma once

// Minimal MATLAB Level-5 (.mat) reader for uint8/uint16/int32 2D arrays
// (BSDS/SBD groundTruth Segmentation & Boundaries).

#include "../../modules/math/vision_types.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

namespace datasets {

struct MatArray {
    std::string name;
    int rows = 0;
    int cols = 0;
    std::vector<double> data;  // row-major
};

inline uint32_t read_u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
inline uint16_t read_u16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
inline int32_t read_i32(const uint8_t* p) {
    int32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline bool parse_mat_v5(const std::vector<uint8_t>& bytes, std::vector<MatArray>& out) {
    if (bytes.size() < 128) {
        return false;
    }
    // Skip 128-byte header.
    size_t pos = 128;
    auto align8 = [](size_t n) { return (n + 7u) & ~size_t(7); };

    while (pos + 8 <= bytes.size()) {
        uint32_t data_type = 0;
        uint32_t data_size = 0;
        size_t data_pos = 0;
        const uint32_t first = read_u32(bytes.data() + pos);
        if ((first & 0xffff0000u) != 0) {
            // Small data element.
            data_type = first & 0xffffu;
            data_size = (first >> 16) & 0xffffu;
            data_pos = pos + 4;
            pos += 8;
        } else {
            data_type = first;
            data_size = read_u32(bytes.data() + pos + 4);
            data_pos = pos + 8;
            pos = data_pos + align8(data_size);
        }
        if (data_type != 14 /* miMATRIX */) {
            if ((first & 0xffff0000u) == 0) {
                // already advanced
            } else {
                // small element already advanced
            }
            continue;
        }
        // Parse matrix sub-elements inside [data_pos, data_pos+data_size).
        size_t p = data_pos;
        const size_t end = data_pos + data_size;
        if (end > bytes.size()) {
            break;
        }
        auto read_tag = [&](size_t& cur, uint32_t& t, uint32_t& s, size_t& dpos) -> bool {
            if (cur + 4 > end) {
                return false;
            }
            const uint32_t f = read_u32(bytes.data() + cur);
            if ((f & 0xffff0000u) != 0) {
                t = f & 0xffffu;
                s = (f >> 16) & 0xffffu;
                dpos = cur + 4;
                cur += 8;
            } else {
                if (cur + 8 > end) {
                    return false;
                }
                t = f;
                s = read_u32(bytes.data() + cur + 4);
                dpos = cur + 8;
                cur = dpos + align8(s);
            }
            return dpos + s <= end || s == 0;
        };

        uint32_t t = 0, s = 0;
        size_t dpos = 0;
        // Array flags
        if (!read_tag(p, t, s, dpos) || t != 6 /* miUINT32 */ || s < 8) {
            continue;
        }
        const uint32_t flags = read_u32(bytes.data() + dpos);
        const int class_id = static_cast<int>(flags & 0xff);
        // Dimensions
        if (!read_tag(p, t, s, dpos) || t != 5 /* miINT32 */ || s < 8) {
            continue;
        }
        const int rows = read_i32(bytes.data() + dpos);
        const int cols = read_i32(bytes.data() + dpos + 4);
        if (rows <= 0 || cols <= 0 || rows > 10000 || cols > 10000) {
            continue;
        }
        // Name
        std::string name;
        if (!read_tag(p, t, s, dpos) || t != 1 /* miINT8 */) {
            continue;
        }
        name.assign(reinterpret_cast<const char*>(bytes.data() + dpos), s);

        // Real part
        if (!read_tag(p, t, s, dpos)) {
            continue;
        }
        MatArray arr;
        arr.name = name;
        arr.rows = rows;
        arr.cols = cols;
        arr.data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0);
        const size_t n = arr.data.size();
        auto store = [&](size_t i, double v) {
            // MATLAB is column-major.
            const size_t r = i % static_cast<size_t>(rows);
            const size_t c = i / static_cast<size_t>(rows);
            arr.data[r * static_cast<size_t>(cols) + c] = v;
        };
        if (t == 7 /* miDOUBLE */ && s >= n * 8) {
            for (size_t i = 0; i < n; ++i) {
                double v;
                std::memcpy(&v, bytes.data() + dpos + i * 8, 8);
                store(i, v);
            }
        } else if (t == 9 /* miSINGLE */ && s >= n * 4) {
            for (size_t i = 0; i < n; ++i) {
                float v;
                std::memcpy(&v, bytes.data() + dpos + i * 4, 4);
                store(i, v);
            }
        } else if (t == 8 /* miINT32 */ && s >= n * 4) {
            for (size_t i = 0; i < n; ++i) {
                store(i, read_i32(bytes.data() + dpos + i * 4));
            }
        } else if (t == 10 /* miUINT16 */ && s >= n * 2) {
            for (size_t i = 0; i < n; ++i) {
                store(i, read_u16(bytes.data() + dpos + i * 2));
            }
        } else if (t == 11 /* miINT16 */ && s >= n * 2) {
            for (size_t i = 0; i < n; ++i) {
                int16_t v;
                std::memcpy(&v, bytes.data() + dpos + i * 2, 2);
                store(i, v);
            }
        } else if (t == 2 /* miUINT8 */ && s >= n) {
            for (size_t i = 0; i < n; ++i) {
                store(i, bytes[dpos + i]);
            }
        } else if (t == 1 /* miINT8 */ && s >= n) {
            for (size_t i = 0; i < n; ++i) {
                store(i, static_cast<int8_t>(bytes[dpos + i]));
            }
        } else {
            (void)class_id;
            continue;
        }
        out.push_back(std::move(arr));
    }
    return !out.empty();
}

inline bool load_mat_file(const std::string& path, std::vector<MatArray>& arrays) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const auto sz = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(sz);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(sz));
    return parse_mat_v5(bytes, arrays);
}

inline math::ImageBuffer mat_array_to_gray(const MatArray& a, double nonzero_as = 255.0) {
    math::ImageBuffer im = math::make_gray(a.cols, a.rows, 0);
    for (int y = 0; y < a.rows; ++y) {
        for (int x = 0; x < a.cols; ++x) {
            const double v = a.data[static_cast<size_t>(y * a.cols + x)];
            if (v != 0.0) {
                im.at(x, y) = static_cast<uint8_t>(
                    std::clamp(v > 1.0 ? v : nonzero_as, 0.0, 255.0));
            }
        }
    }
    return im;
}

// Prefer named array; else largest 2D numeric array.
inline const MatArray* find_mat_array(const std::vector<MatArray>& arrays, const char* name) {
    for (const auto& a : arrays) {
        if (a.name == name) {
            return &a;
        }
    }
    const MatArray* best = nullptr;
    size_t best_n = 0;
    for (const auto& a : arrays) {
        const size_t n = static_cast<size_t>(a.rows) * static_cast<size_t>(a.cols);
        if (n > best_n) {
            best_n = n;
            best = &a;
        }
    }
    return best;
}

inline math::ImageBuffer load_mat_segmentation(const std::string& path) {
    std::vector<MatArray> arrays;
    if (!load_mat_file(path, arrays)) {
        return {};
    }
    // Nested structs often flatten names; accept Segmentation or largest.
    const MatArray* a = find_mat_array(arrays, "Segmentation");
    if (!a) {
        a = find_mat_array(arrays, "");
    }
    if (!a) {
        return {};
    }
    math::ImageBuffer im = math::make_gray(a->cols, a->rows, 0);
    for (int y = 0; y < a->rows; ++y) {
        for (int x = 0; x < a->cols; ++x) {
            const double v = a->data[static_cast<size_t>(y * a->cols + x)];
            im.at(x, y) = static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
        }
    }
    return im;
}

inline math::ImageBuffer load_mat_boundaries(const std::string& path) {
    std::vector<MatArray> arrays;
    if (!load_mat_file(path, arrays)) {
        return {};
    }
    const MatArray* a = find_mat_array(arrays, "Boundaries");
    if (!a) {
        return {};
    }
    return mat_array_to_gray(*a, 255.0);
}

inline std::vector<math::Rect> boxes_from_label_map(const math::ImageBuffer& labels) {
    struct Acc {
        int x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1, area = 0;
    };
    std::vector<Acc> acc(256);
    for (int y = 0; y < labels.height; ++y) {
        for (int x = 0; x < labels.width; ++x) {
            const uint8_t id = labels.at(x, y);
            if (id == 0) {
                continue;
            }
            auto& a = acc[id];
            a.x0 = std::min(a.x0, x);
            a.y0 = std::min(a.y0, y);
            a.x1 = std::max(a.x1, x);
            a.y1 = std::max(a.y1, y);
            ++a.area;
        }
    }
    std::vector<math::Rect> boxes;
    for (int id = 1; id < 256; ++id) {
        if (acc[id].area <= 0) {
            continue;
        }
        boxes.push_back({static_cast<float>(acc[id].x0), static_cast<float>(acc[id].y0),
                         static_cast<float>(acc[id].x1 - acc[id].x0 + 1),
                         static_cast<float>(acc[id].y1 - acc[id].y0 + 1)});
    }
    return boxes;
}

}  // namespace datasets
