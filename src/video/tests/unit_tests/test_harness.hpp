#pragma once

#include "datasets/io/vision_io.hpp"
#include "math/contour_compat.hpp"
#include "atom_config.hpp"
#include "datasets/provider_factory.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Kept for benchmarks / integration suites that still score against GT.
struct AccuracyReport {
    std::string name;
    int total = 0;
    int passed = 0;
    double accuracy = 0.0;
    std::vector<std::string> notes;
    double elapsed_ms = 0.0;

    void finalize() {
        accuracy = total > 0 ? static_cast<double>(passed) / static_cast<double>(total) : 0.0;
    }

    void print() const {
        std::cout << "\n========== " << name << " ==========\n";
        std::cout << "cases     : " << passed << " / " << total << "\n";
        std::cout << "accuracy  : " << std::fixed << std::setprecision(4) << (accuracy * 100.0) << " %\n";
        std::cout << "elapsed   : " << std::setprecision(3) << elapsed_ms << " ms\n";
        for (const auto& note : notes) {
            std::cout << "note      : " << note << "\n";
        }
        std::cout << "====================================\n";
    }
};

// Atoms are input→output demos: no ground-truth pass/fail.
struct AtomDemoReport {
    std::string name;
    int n_inputs = 0;
    int n_outputs = 0;
    double elapsed_ms = 0.0;
    std::vector<std::string> notes;

    void print() const {
        std::cout << "\n========== " << name << " (atom demo) ==========\n";
        std::cout << "inputs    : " << n_inputs << "\n";
        std::cout << "outputs   : " << n_outputs << "\n";
        std::cout << "elapsed   : " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
        for (const auto& note : notes) {
            std::cout << "note      : " << note << "\n";
        }
        std::cout << "==============================================\n";
    }
};

class ScopedTimer {
public:
    explicit ScopedTimer(double* dest_ms) : dest_(dest_ms), start_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        const auto end = std::chrono::steady_clock::now();
        *dest_ = std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    double* dest_;
    std::chrono::steady_clock::time_point start_;
};

struct AtomCli {
    std::string data_root;
    std::string dataset;
    std::string sample_filter;
    std::string artifact_dir;
    bool help = false;
    bool list_only = false;
};

inline void print_atom_cli_help(const char* binary, const char* default_dataset) {
    std::cout
        << "Usage: " << binary << " [options] [data_root]\n"
        << "  Atom demo: load inputs, run the algorithm, write values/images to artifacts/.\n"
        << "  No ground-truth scoring (that belongs in benchmarks).\n\n"
        << "  --data <dir>       Atom data root (default: VISION_DATA_DIR / find_data_root)\n"
        << "  --dataset <name>   Suite folder under data root (default: " << default_dataset << ")\n"
        << "  --sample <substr>  Only process samples whose filename contains this substring\n"
        << "  --artifacts <dir>  Override artifact output directory\n"
        << "  --list             List matching samples and exit\n"
        << "  --help             Show this help\n"
        << "\nArtifacts: atoms/artifacts/<suite>/  (values TSV, PGM images, manifest)\n"
        << "Prepare packs: python data/vision/prepare_atom_datasets.py\n";
}

inline AtomCli parse_atom_cli(int argc, char** argv, const char* default_dataset) {
    AtomCli cli;
    cli.dataset = default_dataset;
#ifdef VISION_TEST_ARTIFACT_ROOT
    cli.artifact_dir = VISION_TEST_ARTIFACT_ROOT;
#else
    cli.artifact_dir = (fs::current_path() / "artifacts").string();
#endif
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                cli.help = true;
                return {};
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            cli.help = true;
        } else if (a == "--list") {
            cli.list_only = true;
        } else if (a == "--data" || a == "--data-root") {
            cli.data_root = need("--data");
        } else if (a == "--dataset") {
            cli.dataset = need("--dataset");
        } else if (a == "--sample") {
            cli.sample_filter = need("--sample");
        } else if (a == "--artifacts") {
            cli.artifact_dir = need("--artifacts");
        } else if (!a.empty() && a[0] != '-') {
            cli.data_root = a;
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            cli.help = true;
        }
    }
    if (cli.data_root.empty()) {
        cli.data_root = vision::find_data_root(argc > 0 ? argv[0] : nullptr);
    }
    return cli;
}

inline std::string make_artifact_dir(const std::string& suite_or_path) {
    fs::path dir;
#ifdef VISION_TEST_ARTIFACT_ROOT
    const fs::path configured(VISION_TEST_ARTIFACT_ROOT);
    if (fs::path(suite_or_path).is_absolute() || suite_or_path.find('/') != std::string::npos ||
        suite_or_path.find('\\') != std::string::npos) {
        dir = suite_or_path;
    } else {
        dir = configured / suite_or_path;
    }
#else
    dir = fs::path(suite_or_path);
    if (!dir.is_absolute()) {
        dir = fs::current_path() / "artifacts" / suite_or_path;
    }
#endif
    fs::create_directories(dir);
    return dir.string();
}

inline void print_banner(const std::string& title) { std::cout << "\n>>> " << title << "\n"; }

struct LoadedSample {
    vision::DatasetRow row;
    vision::GrayImage image;
    std::string path;
};

inline bool sample_matches_filter(const LoadedSample& sample, const std::string& filter) {
    if (filter.empty()) {
        return true;
    }
    return sample.row.file.find(filter) != std::string::npos ||
           sample.path.find(filter) != std::string::npos ||
           sample.row.label.find(filter) != std::string::npos;
}

inline std::vector<LoadedSample> load_png_dataset(const std::string& folder,
                                                  const std::string& sample_filter = {}) {
    std::vector<LoadedSample> samples;
    const auto tsv_path = vision::join_path(folder, "index.tsv");
    auto rows = vision::load_tsv(tsv_path);
    if (!rows.empty()) {
        for (auto& row : rows) {
            if (row.file.size() >= 5 && row.file.substr(row.file.size() - 5) == ".json") {
                continue;
            }
            LoadedSample sample;
            sample.row = row;
            sample.path = vision::join_path(folder, row.file);
            sample.image = vision::load_gray_png(sample.path);
            if (sample.image.empty()) {
                continue;
            }
            if (!sample_matches_filter(sample, sample_filter)) {
                continue;
            }
            samples.push_back(std::move(sample));
        }
        return samples;
    }
    for (const auto& path : vision::list_png_files(folder, true)) {
        LoadedSample sample;
        sample.path = path;
        sample.image = vision::load_gray_png(path);
        sample.row.file = fs::path(path).filename().string();
        sample.row.label = fs::path(path).stem().string();
        const auto stem = sample.row.label;
        const auto under = stem.find('_');
        if (under != std::string::npos) {
            sample.row.label = stem.substr(0, under);
        }
        if (sample.image.empty()) {
            continue;
        }
        if (!sample_matches_filter(sample, sample_filter)) {
            continue;
        }
        samples.push_back(std::move(sample));
    }
    return samples;
}

inline void write_summary_tsv(const std::string& dir, const AccuracyReport& report,
                              const std::string& extra = {}) {
    std::ostringstream ss;
    ss << "suite\ttotal\tpassed\taccuracy\telapsed_ms\n";
    ss << report.name << '\t' << report.total << '\t' << report.passed << '\t' << std::fixed
       << std::setprecision(6) << report.accuracy << '\t' << report.elapsed_ms << '\n';
    if (!extra.empty()) {
        ss << "\n" << extra;
    }
    vision::write_text_file(vision::join_path(dir, "summary.tsv"), ss.str());
}

inline void write_atom_manifest(const std::string& dir, const AtomDemoReport& report,
                                const std::vector<std::string>& files = {}) {
    std::ostringstream ss;
    ss << "atom\t" << report.name << "\n";
    ss << "role\tinput_output_demo\n";
    ss << "n_inputs\t" << report.n_inputs << "\n";
    ss << "n_outputs\t" << report.n_outputs << "\n";
    ss << "elapsed_ms\t" << std::fixed << std::setprecision(3) << report.elapsed_ms << "\n";
    for (const auto& n : report.notes) {
        ss << "note\t" << n << "\n";
    }
    for (const auto& f : files) {
        ss << "file\t" << f << "\n";
    }
    vision::write_text_file(vision::join_path(dir, "manifest.tsv"), ss.str());
}

inline std::string stem_of(const std::string& file) {
    return fs::path(file).stem().string();
}

inline int run_atom_main(int argc, char** argv, const char* default_dataset,
                         const std::function<int(const AtomCli&)>& body) {
    const AtomCli cli = parse_atom_cli(argc, argv, default_dataset);
    if (cli.help) {
        print_atom_cli_help(argc > 0 ? argv[0] : "atom_test", default_dataset);
        return 0;
    }
    std::cout << "data root : " << cli.data_root << "\n";
    std::cout << "dataset   : " << cli.dataset << "\n";
    if (!cli.sample_filter.empty()) {
        std::cout << "sample    : " << cli.sample_filter << "\n";
    }
    std::cout << "artifacts : " << cli.artifact_dir << "\n";
    std::cout.flush();
    return body(cli);
}

// Contour-kit modules use ImageBuffer/Field; atom loaders produce GrayImage.
inline contour::ImageBuffer to_contour(const vision::GrayImage& g) {
    contour::ImageBuffer im;
    im.width = g.width;
    im.height = g.height;
    im.channels = 1;
    im.data = g.data;
    return im;
}

inline vision::GrayImage to_gray(const contour::ImageBuffer& im) {
    vision::GrayImage g;
    g.width = im.width;
    g.height = im.height;
    g.data.resize(static_cast<size_t>(std::max(0, im.width * im.height)));
    if (im.channels == 1 && im.data.size() == g.data.size()) {
        g.data = im.data;
        return g;
    }
    for (int y = 0; y < im.height; ++y) {
        for (int x = 0; x < im.width; ++x) {
            g.at(x, y) = static_cast<uint8_t>(std::clamp(im.gray(x, y), 0.0f, 255.0f));
        }
    }
    return g;
}

inline vision::GrayImage field_to_gray(const contour::Field& f) {
    vision::GrayImage g;
    g.width = f.width;
    g.height = f.height;
    g.data.assign(static_cast<size_t>(std::max(0, f.width * f.height)), 0);
    if (f.data.empty()) {
        return g;
    }
    float lo = f.data.front();
    float hi = f.data.front();
    for (float v : f.data) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const float span = (hi - lo) > 1e-6f ? (hi - lo) : 1.0f;
    for (size_t i = 0; i < f.data.size() && i < g.data.size(); ++i) {
        g.data[i] = static_cast<uint8_t>(std::clamp((f.data[i] - lo) / span * 255.0f, 0.0f, 255.0f));
    }
    return g;
}

inline vision::GrayImage downscale_max_side(const vision::GrayImage& src, int max_side) {
    const int m = std::max(src.width, src.height);
    if (max_side <= 0 || m <= max_side || src.empty()) {
        return src;
    }
    vision::GrayImage out;
    out.width = std::max(1, src.width * max_side / m);
    out.height = std::max(1, src.height * max_side / m);
    out.data.resize(static_cast<size_t>(out.width * out.height));
    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < out.width; ++x) {
            const int sx = std::min(src.width - 1, x * src.width / out.width);
            const int sy = std::min(src.height - 1, y * src.height / out.height);
            out.at(x, y) = src.at(sx, sy);
        }
    }
    return out;
}

inline void plot_line(vision::GrayImage& im, int x0, int y0, int x1, int y1, uint8_t v) {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        if (x0 >= 0 && y0 >= 0 && x0 < im.width && y0 < im.height) {
            im.at(x0, y0) = v;
        }
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

template <typename Point>
inline vision::GrayImage overlay_polyline(const vision::GrayImage& src, const std::vector<Point>& pts,
                                          bool closed, uint8_t ink = 255) {
    vision::GrayImage out = src;
    for (uint8_t& p : out.data) {
        p = static_cast<uint8_t>(p / 2);
    }
    if (pts.size() < 2) {
        return out;
    }
    auto at_i = [&](size_t i) {
        return std::pair<int, int>{static_cast<int>(std::lround(pts[i].x)),
                                   static_cast<int>(std::lround(pts[i].y))};
    };
    for (size_t i = 1; i < pts.size(); ++i) {
        const auto a = at_i(i - 1);
        const auto b = at_i(i);
        plot_line(out, a.first, a.second, b.first, b.second, ink);
    }
    if (closed) {
        const auto a = at_i(pts.size() - 1);
        const auto b = at_i(0);
        plot_line(out, a.first, a.second, b.first, b.second, ink);
    }
    return out;
}

inline vision::GrayImage overlay_rect(const vision::GrayImage& src, float x, float y, float w, float h,
                                      uint8_t ink = 128) {
    vision::GrayImage out = src;
    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, std::max(0, out.width - 1));
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, std::max(0, out.height - 1));
    const int x1 = std::clamp(static_cast<int>(std::ceil(x + w)), 0, out.width);
    const int y1 = std::clamp(static_cast<int>(std::ceil(y + h)), 0, out.height);
    for (int xx = x0; xx < x1; ++xx) {
        if (y0 < out.height) {
            out.at(xx, y0) = ink;
        }
        if (y1 - 1 >= 0 && y1 - 1 < out.height) {
            out.at(xx, y1 - 1) = ink;
        }
    }
    for (int yy = y0; yy < y1; ++yy) {
        if (x0 < out.width) {
            out.at(x0, yy) = ink;
        }
        if (x1 - 1 >= 0 && x1 - 1 < out.width) {
            out.at(x1 - 1, yy) = ink;
        }
    }
    return out;
}

inline std::vector<LoadedSample> load_atom_png_dataset(const std::string& root, const std::string& dataset,
                                                      const std::string& sample_filter,
                                                      std::initializer_list<const char*> fallbacks = {
                                                          "unit_contour", "unit_medial_axis", "unit_euler",
                                                          "unit_hu_moments"}) {
    auto samples = load_png_dataset(vision::dataset_dir(root, dataset), sample_filter);
    if (!samples.empty()) {
        return samples;
    }
    for (const char* fb : fallbacks) {
        if (dataset == fb) {
            continue;
        }
        samples = load_png_dataset(vision::dataset_dir(root, fb), sample_filter);
        if (!samples.empty()) {
            std::cout << "note: using fallback pack " << fb << " (no images in " << dataset << ")\n";
            return samples;
        }
    }
    return samples;
}

inline vision::GrayImage colorize_labels(const std::vector<int>& labels, int w, int h) {
    vision::GrayImage g;
    g.width = w;
    g.height = h;
    g.data.assign(static_cast<size_t>(std::max(0, w * h)), 0);
    for (size_t i = 0; i < g.data.size() && i < labels.size(); ++i) {
        const int L = labels[i];
        g.data[i] = L <= 0 ? 0 : static_cast<uint8_t>(40 + (static_cast<unsigned>(L) * 47u) % 200u);
    }
    return g;
}

inline vision::GrayImage label_boundaries(const std::vector<int>& labels, int w, int h) {
    vision::GrayImage g;
    g.width = w;
    g.height = h;
    g.data.assign(static_cast<size_t>(std::max(0, w * h)), 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int L = labels[static_cast<size_t>(y * w + x)];
            bool edge = false;
            if (x + 1 < w && labels[static_cast<size_t>(y * w + x + 1)] != L) {
                edge = true;
            }
            if (y + 1 < h && labels[static_cast<size_t>((y + 1) * w + x)] != L) {
                edge = true;
            }
            g.at(x, y) = edge ? 255 : 0;
        }
    }
    return g;
}

inline vision::GrayImage overlay_mask(const vision::GrayImage& src, const vision::GrayImage& ink) {
    vision::GrayImage out = src;
    for (uint8_t& p : out.data) {
        p = static_cast<uint8_t>(p / 2);
    }
    const int n = std::min(static_cast<int>(out.data.size()), static_cast<int>(ink.data.size()));
    for (int i = 0; i < n; ++i) {
        if (ink.data[static_cast<size_t>(i)] > 0) {
            out.data[static_cast<size_t>(i)] = 255;
        }
    }
    return out;
}

struct ProviderLoadedSample {
    datasets::VisionSample sample;
    vision::GrayImage image;
    vision::GrayImage ground_truth;
    std::string path;
    vision::DatasetRow row;
};

inline bool provider_sample_matches(const ProviderLoadedSample& s, const std::string& filter) {
    if (filter.empty()) {
        return true;
    }
    return s.sample.id.find(filter) != std::string::npos ||
           s.path.find(filter) != std::string::npos ||
           s.row.label.find(filter) != std::string::npos;
}

inline std::vector<ProviderLoadedSample> load_provider_samples(const std::string& provider_name,
                                                                 const std::string& sample_filter,
                                                                 size_t max_samples = 8) {
    std::vector<ProviderLoadedSample> out;
    try {
        const auto vision_root = fs::path(vision::find_vision_root(nullptr));
        auto provider = datasets::make_provider(provider_name, vision_root);
        const size_t n = std::min(provider->size(), max_samples);
        for (size_t i = 0; i < n; ++i) {
            ProviderLoadedSample loaded;
            loaded.sample = provider->load_sample(i);
            loaded.path = loaded.sample.image_path;
            loaded.row.file = loaded.sample.id.empty() ? loaded.path : loaded.sample.id;
            loaded.row.label = loaded.sample.label.empty() ? loaded.row.file : loaded.sample.label;
            if (!loaded.sample.luma.empty()) {
                loaded.image = loaded.sample.luma;
            } else if (!loaded.sample.rgb.empty()) {
                loaded.image = vision::rgb_to_luma(loaded.sample.rgb);
            } else if (!loaded.sample.mask.empty()) {
                loaded.image = loaded.sample.mask;
            }
            loaded.ground_truth = loaded.sample.mask;
            if (loaded.ground_truth.empty()) {
                loaded.ground_truth = loaded.sample.boundary;
            }
            if (!provider_sample_matches(loaded, sample_filter)) {
                continue;
            }
            out.push_back(std::move(loaded));
        }
    } catch (const std::exception& e) {
        std::cerr << "provider load failed (" << provider_name << "): " << e.what() << "\n";
    }
    return out;
}

inline std::vector<LoadedSample> provider_to_legacy_samples(const std::vector<ProviderLoadedSample>& in) {
    std::vector<LoadedSample> out;
    out.reserve(in.size());
    for (const auto& s : in) {
        LoadedSample legacy;
        legacy.row = s.row;
        legacy.path = s.path;
        legacy.image = s.image;
        out.push_back(std::move(legacy));
    }
    return out;
}

struct MissionLoadResult {
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    std::string provider_name;
};

inline MissionLoadResult load_mission_samples(const AtomCli& cli, const char* argv0, size_t max_samples = 8,
                                              int max_side = 128) {
    MissionLoadResult out;
    const vision::AtomConfig config = vision::load_atom_config_near(argv0);
    out.provider_name = !config.preferred_dataset.empty() ? config.preferred_dataset : cli.dataset;
    out.provider_samples = load_provider_samples(out.provider_name, cli.sample_filter, max_samples);
    if (!out.provider_samples.empty()) {
        out.samples = provider_to_legacy_samples(out.provider_samples);
        const bool use_mask =
            config.input_type.find("BINARY") != std::string::npos ||
            config.input_type.find("MASK") != std::string::npos ||
            config.input_type.find("GRID") != std::string::npos ||
            config.input_type.find("SILHOUETTE") != std::string::npos ||
            out.provider_name.rfind("synthetic", 0) == 0;
        if (max_side > 0) {
            for (size_t i = 0; i < out.samples.size(); ++i) {
                if (i < out.provider_samples.size() && !out.provider_samples[i].ground_truth.empty()) {
                    out.provider_samples[i].ground_truth =
                        downscale_max_side(out.provider_samples[i].ground_truth, max_side);
                }
                if (use_mask && i < out.provider_samples.size() &&
                    !out.provider_samples[i].ground_truth.empty()) {
                    out.samples[i].image = out.provider_samples[i].ground_truth;
                } else if (!out.samples[i].image.empty()) {
                    out.samples[i].image = downscale_max_side(out.samples[i].image, max_side);
                }
            }
        } else if (use_mask) {
            for (size_t i = 0; i < out.samples.size() && i < out.provider_samples.size(); ++i) {
                if (!out.provider_samples[i].ground_truth.empty()) {
                    out.samples[i].image = out.provider_samples[i].ground_truth;
                }
            }
        }
        return out;
    }
    out.samples = load_atom_png_dataset(cli.data_root, cli.dataset, cli.sample_filter);
    if (max_side > 0) {
        for (auto& s : out.samples) {
            s.image = downscale_max_side(s.image, max_side);
        }
    }
    return out;
}

inline vision::GrayImage mission_mask_image(const ProviderLoadedSample* ps, const vision::GrayImage& fallback) {
    if (ps != nullptr) {
        if (!ps->ground_truth.empty()) {
            return ps->ground_truth;
        }
        if (!ps->sample.mask.empty()) {
            return downscale_max_side(ps->sample.mask, std::max(fallback.width, fallback.height));
        }
    }
    return fallback;
}
