#pragma once

#include "math/vision_io.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

inline std::string make_artifact_dir(const std::string& test_name) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::current_path() / "artifacts" / test_name;
    fs::create_directories(dir);
    return dir.string();
}

inline void print_banner(const std::string& title) {
    std::cout << "\n>>> " << title << "\n";
}

struct LoadedSample {
    vision::DatasetRow row;
    vision::GrayImage image;
    std::string path;
};

inline std::vector<LoadedSample> load_png_dataset(const std::string& folder) {
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
            if (!sample.image.empty()) {
                samples.push_back(std::move(sample));
            }
        }
        return samples;
    }
    for (const auto& path : vision::list_png_files(folder, true)) {
        LoadedSample sample;
        sample.path = path;
        sample.image = vision::load_gray_png(path);
        sample.row.file = std::filesystem::path(path).filename().string();
        sample.row.label = std::filesystem::path(path).stem().string();
        const auto stem = sample.row.label;
        const auto under = stem.find('_');
        if (under != std::string::npos) {
            sample.row.label = stem.substr(0, under);
        }
        if (!sample.image.empty()) {
            samples.push_back(std::move(sample));
        }
    }
    return samples;
}
