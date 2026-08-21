#pragma once

#include "vision_types.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>

namespace vision {

GrayImage load_gray_png(const std::string& path);
bool save_pgm(const std::string& path, const GrayImage& image);
bool save_pgm_float(const std::string& path, const std::vector<float>& values, int width, int height);

std::vector<DatasetRow> load_tsv(const std::string& path);
std::vector<Vec2> load_xy_json(const std::string& path);
std::string find_data_root(const char* argv0 = nullptr);
std::string join_path(const std::string& a, const std::string& b);
std::vector<std::string> list_png_files(const std::string& dir, bool recursive = true);
std::string read_text_file(const std::string& path);
bool write_text_file(const std::string& path, const std::string& text);
std::string parent_dir(const std::string& path);

inline std::string dataset_dir(const std::string& data_root, const std::string& name) {
    return join_path(data_root, name);
}

}  // namespace vision
