#pragma once

#include "types.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace contour {

inline bool skip_ws_and_comments(std::istream& in) {
    for (;;) {
        const int c = in.peek();
        if (c == EOF) {
            return false;
        }
        if (c == '#') {
            std::string dummy;
            std::getline(in, dummy);
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            in.get();
            continue;
        }
        return true;
    }
}

inline ImageBuffer load_netpbm(const std::string& path) {
    ImageBuffer im;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return im;
    }
    std::string magic;
    in >> magic;
    skip_ws_and_comments(in);
    int w = 0, h = 0, maxv = 0;
    in >> w >> h;
    skip_ws_and_comments(in);
    in >> maxv;
    in.get();  // single whitespace after maxval
    if (w <= 0 || h <= 0 || maxv <= 0) {
        return im;
    }
    im.width = w;
    im.height = h;
    if (magic == "P5" || magic == "P2") {
        im.channels = 1;
        im.data.resize(static_cast<size_t>(w * h));
        if (magic == "P5") {
            in.read(reinterpret_cast<char*>(im.data.data()), static_cast<std::streamsize>(im.data.size()));
        } else {
            for (size_t i = 0; i < im.data.size(); ++i) {
                int v = 0;
                in >> v;
                im.data[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
            }
        }
    } else if (magic == "P6" || magic == "P3") {
        im.channels = 3;
        im.data.resize(static_cast<size_t>(w * h * 3));
        if (magic == "P6") {
            in.read(reinterpret_cast<char*>(im.data.data()), static_cast<std::streamsize>(im.data.size()));
        } else {
            for (size_t i = 0; i < im.data.size(); ++i) {
                int v = 0;
                in >> v;
                im.data[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
            }
        }
    }
    if (maxv != 255 && !im.data.empty()) {
        for (uint8_t& p : im.data) {
            p = static_cast<uint8_t>(std::lround(p * 255.0 / maxv));
        }
    }
    return im;
}

inline ImageBuffer load_bmp(const std::string& path) {
    ImageBuffer im;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return im;
    }
    char hdr[54]{};
    in.read(hdr, 54);
    if (in.gcount() < 54 || hdr[0] != 'B' || hdr[1] != 'M') {
        return im;
    }
    const int offset = *reinterpret_cast<const int32_t*>(hdr + 10);
    const int w = *reinterpret_cast<const int32_t*>(hdr + 18);
    const int h = *reinterpret_cast<const int32_t*>(hdr + 22);
    const int16_t bits = *reinterpret_cast<const int16_t*>(hdr + 28);
    const int32_t comp = *reinterpret_cast<const int32_t*>(hdr + 30);
    if (w <= 0 || h == 0 || comp != 0 || (bits != 24 && bits != 8)) {
        return im;
    }
    const int height = std::abs(h);
    const bool bottom_up = h > 0;
    im.width = w;
    im.height = height;
    im.channels = 1;
    im.data.assign(static_cast<size_t>(w * height), 0);
    in.seekg(offset, std::ios::beg);
    const int row_bytes = bits == 24 ? ((w * 3 + 3) & ~3) : ((w + 3) & ~3);
    std::vector<uint8_t> row(static_cast<size_t>(row_bytes));
    for (int y = 0; y < height; ++y) {
        in.read(reinterpret_cast<char*>(row.data()), row_bytes);
        const int yy = bottom_up ? (height - 1 - y) : y;
        for (int x = 0; x < w; ++x) {
            uint8_t g = 0;
            if (bits == 24) {
                const uint8_t b = row[static_cast<size_t>(x * 3 + 0)];
                const uint8_t gg = row[static_cast<size_t>(x * 3 + 1)];
                const uint8_t r = row[static_cast<size_t>(x * 3 + 2)];
                g = static_cast<uint8_t>(0.299f * r + 0.587f * gg + 0.114f * b);
            } else {
                g = row[static_cast<size_t>(x)];
            }
            im.at(x, yy) = g;
        }
    }
    return im;
}

inline ImageBuffer load_image(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (ext == ".bmp") {
        return load_bmp(path);
    }
    return load_netpbm(path);
}

inline bool save_pgm(const std::string& path, const ImageBuffer& im) {
    if (im.empty()) {
        return false;
    }
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << "P5\n" << im.width << " " << im.height << "\n255\n";
    if (im.channels == 1) {
        out.write(reinterpret_cast<const char*>(im.data.data()),
                  static_cast<std::streamsize>(im.data.size()));
        return static_cast<bool>(out);
    }
    std::vector<uint8_t> gray(static_cast<size_t>(im.width * im.height));
    for (int y = 0; y < im.height; ++y) {
        for (int x = 0; x < im.width; ++x) {
            gray[static_cast<size_t>(y * im.width + x)] =
                static_cast<uint8_t>(std::clamp(im.gray(x, y), 0.0f, 255.0f));
        }
    }
    out.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(out);
}

inline bool save_pgm_field(const std::string& path, const Field& f) {
    ImageBuffer im = make_gray(f.width, f.height);
    float lo = 1e9f, hi = -1e9f;
    for (float v : f.data) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const float span = (hi - lo) > 1e-6f ? (hi - lo) : 1.0f;
    for (size_t i = 0; i < f.data.size(); ++i) {
        im.data[i] = static_cast<uint8_t>(std::clamp((f.data[i] - lo) / span * 255.0f, 0.0f, 255.0f));
    }
    return save_pgm(path, im);
}

}  // namespace contour
