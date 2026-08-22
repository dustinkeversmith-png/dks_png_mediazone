#pragma once

#include "math/vision_types.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>

namespace vision {

GrayImage load_gray_png(const std::string& path);
ImageBuffer load_rgb_png(const std::string& path);
bool save_pgm(const std::string& path, const GrayImage& image);
bool save_pgm_float(const std::string& path, const std::vector<float>& values, int width, int height);

std::vector<DatasetRow> load_tsv(const std::string& path);
std::vector<Vec2> load_xy_json(const std::string& path);
std::string find_data_root(const char* argv0 = nullptr);
std::string find_vision_root(const char* argv0 = nullptr);
std::string join_path(const std::string& a, const std::string& b);
std::vector<std::string> list_png_files(const std::string& dir, bool recursive = true);
std::vector<std::string> list_image_files(const std::string& dir, bool recursive = true);
std::string read_text_file(const std::string& path);
bool write_text_file(const std::string& path, const std::string& text);
std::string parent_dir(const std::string& path);

inline std::string dataset_dir(const std::string& data_root, const std::string& name) {
    return join_path(data_root, name);
}

inline ImageBuffer rgb_to_luma(const ImageBuffer& rgb) {
    ImageBuffer gray = make_gray(rgb.width, rgb.height, 0);
    if (rgb.channels == 1) {
        gray.data = rgb.data;
        return gray;
    }
    for (int y = 0; y < rgb.height; ++y) {
        for (int x = 0; x < rgb.width; ++x) {
            gray.at(x, y) = static_cast<uint8_t>(std::clamp(rgb.gray(x, y), 0.0f, 255.0f));
        }
    }
    return gray;
}

}  // namespace vision
