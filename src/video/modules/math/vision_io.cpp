#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#include "vision_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace vision {
namespace fs = std::filesystem;

std::string join_path(const std::string& a, const std::string& b) {
    return (fs::path(a) / b).string();
}

std::string parent_dir(const std::string& path) {
    return fs::path(path).parent_path().string();
}

GrayImage load_gray_png(const std::string& path) {
    GrayImage image;
    int w = 0, h = 0, n = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 1);
    if (data == nullptr) {
        std::cerr << "Failed to load image: " << path << " (" << stbi_failure_reason() << ")\n";
        return image;
    }
    image.width = w;
    image.height = h;
    image.pixels.assign(data, data + static_cast<size_t>(w) * static_cast<size_t>(h));
    stbi_image_free(data);
    return image;
}

bool save_pgm(const std::string& path, const GrayImage& image) {
    if (image.empty()) {
        return false;
    }
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << "P5\n" << image.width << " " << image.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.pixels.data()),
              static_cast<std::streamsize>(image.pixels.size()));
    return static_cast<bool>(out);
}

bool save_pgm_float(const std::string& path, const std::vector<float>& values, int width, int height) {
    if (values.empty() || width <= 0 || height <= 0) {
        return false;
    }
    float lo = values[0];
    float hi = values[0];
    for (float v : values) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const float span = (hi - lo) > 1e-8f ? (hi - lo) : 1.0f;
    GrayImage image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width * height));
    for (size_t i = 0; i < values.size() && i < image.pixels.size(); ++i) {
        const float n = (values[i] - lo) / span;
        image.pixels[i] = static_cast<uint8_t>(std::clamp(n * 255.0f, 0.0f, 255.0f));
    }
    return save_pgm(path, image);
}

static std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : line) {
        if (c == '\t') {
            parts.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
    }
    parts.push_back(cur);
    return parts;
}

std::vector<DatasetRow> load_tsv(const std::string& path) {
    std::vector<DatasetRow> rows;
    std::ifstream in(path);
    if (!in) {
        return rows;
    }
    std::string line;
    bool header = true;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        auto parts = split_tabs(line);
        if (header) {
            header = false;
            continue;
        }
        DatasetRow row;
        row.fields = parts;
        if (!parts.empty()) {
            row.file = parts[0];
        }
        if (parts.size() > 1) {
            row.label = parts[1];
        }
        if (parts.size() > 2) {
            row.extra = parts[2];
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

std::vector<Vec2> load_xy_json(const std::string& path) {
    std::vector<Vec2> points;
    const std::string text = read_text_file(path);
    if (text.empty()) {
        return points;
    }
    for (size_t i = 0; i + 3 < text.size(); ++i) {
        if (text[i] != '"' || (text[i + 1] != 'x' && text[i + 1] != 'X')) {
            continue;
        }
        if (text[i + 2] != '"') {
            continue;
        }
        size_t colon = text.find(':', i + 3);
        if (colon == std::string::npos) {
            break;
        }
        const float x = std::strtof(text.c_str() + colon + 1, nullptr);
        size_t ykey = text.find("\"y\"", colon);
        if (ykey == std::string::npos) {
            ykey = text.find("\"Y\"", colon);
        }
        if (ykey == std::string::npos) {
            continue;
        }
        size_t ycolon = text.find(':', ykey);
        if (ycolon == std::string::npos) {
            continue;
        }
        const float y = std::strtof(text.c_str() + ycolon + 1, nullptr);
        points.push_back({x, y});
        i = ycolon;
    }
    return points;
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool write_text_file(const std::string& path, const std::string& text) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << text;
    return true;
}

std::vector<std::string> list_png_files(const std::string& dir, bool recursive) {
    std::vector<std::string> files;
    if (!fs::exists(dir)) {
        return files;
    }
    if (recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".png" || ext == ".pgm" || ext == ".bmp" || ext == ".gif") {
                files.push_back(entry.path().string());
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".png" || ext == ".pgm" || ext == ".bmp" || ext == ".gif") {
                files.push_back(entry.path().string());
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

static bool looks_like_data_root(const fs::path& p) {
    return fs::exists(p / "unit_contour") || fs::exists(p / "unit_synthetic") ||
           fs::exists(p / "integration_1_shapes");
}

std::string find_data_root(const char* argv0) {
#ifdef VISION_DATA_DIR
    {
        fs::path configured(VISION_DATA_DIR);
        if (looks_like_data_root(configured)) {
            return configured.string();
        }
    }
#endif
    const char* env = std::getenv("VISION_DATA_ROOT");
    if (env && looks_like_data_root(env)) {
        return std::string(env);
    }

    std::vector<fs::path> seeds;
    seeds.push_back(fs::current_path());
    if (argv0 != nullptr) {
        seeds.push_back(fs::absolute(argv0).parent_path());
    }
    for (fs::path cur : seeds) {
        for (int i = 0; i < 8; ++i) {
            const fs::path candidate = cur / "data" / "vision" / "data_vision";
            if (looks_like_data_root(candidate)) {
                return candidate.string();
            }
            if (!cur.has_parent_path() || cur == cur.parent_path()) {
                break;
            }
            cur = cur.parent_path();
        }
    }
    return (fs::current_path() / "data" / "vision" / "data_vision").string();
}

}  // namespace vision
