#include "../contour_kit/contour_kit.hpp"
#include "../boundary_tracing/marching_squares.hpp"
#include "../boundary_tracing/dual_contouring.hpp"
#include "../boundary_tracing/snakes.hpp"
#include "../boundary_tracing/level_set.hpp"
#include "../boundary_tracing/live_wire.hpp"
#include "../boundary_tracing/graph_cut.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace contour;

#ifndef VISION_BENCH_DIR
#define VISION_BENCH_DIR "data/vision/benchmarks"
#endif

struct Row {
    std::string algo;
    std::string sample;
    double iou = 0;
    double f1 = 0;
    double ms = 0;
};

static fs::path bench_root() {
    fs::path p(VISION_BENCH_DIR);
    if (fs::exists(p / "contour_ready")) {
        return p / "contour_ready";
    }
    if (fs::exists("data/vision/benchmarks/contour_ready")) {
        return "data/vision/benchmarks/contour_ready";
    }
    return p / "contour_ready";
}

static void print_table(const std::vector<Row>& rows) {
    std::cout << "\n";
    std::cout << std::left << std::setw(22) << "Algorithm" << " | "
              << std::setw(28) << "Dataset Sample" << " | "
              << std::setw(8) << "IoU" << " | "
              << std::setw(12) << "Boundary F1" << " | "
              << std::setw(12) << "Latency ms" << "\n";
    std::cout << std::string(90, '-') << "\n";
    std::cout << std::fixed << std::setprecision(3);
    for (const auto& r : rows) {
        std::cout << std::left << std::setw(22) << r.algo << " | "
                  << std::setw(28) << r.sample << " | "
                  << std::setw(8) << r.iou << " | "
                  << std::setw(12) << r.f1 << " | "
                  << std::setw(12) << r.ms << "\n";
    }
}

static std::vector<std::string> list_pgms(const fs::path& dir, const std::string& skip_suffix) {
    std::vector<std::string> out;
    if (!fs::exists(dir)) {
        return out;
    }
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) {
            continue;
        }
        const auto name = e.path().filename().string();
        if (e.path().extension() != ".pgm") {
            continue;
        }
        if (name.find(skip_suffix) != std::string::npos) {
            continue;
        }
        out.push_back(e.path().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

static Row eval_dis(const std::string& img_path, const std::string& algo) {
    Row row;
    row.algo = algo;
    row.sample = fs::path(img_path).stem().string();
    const auto img = load_image(img_path);
    const auto mask = load_image(fs::path(img_path).replace_filename(
        fs::path(img_path).stem().string() + "_mask.pgm").string());
    if (img.empty() || mask.empty()) {
        return row;
    }
    const Field sdf = ChamferSDF::from_mask(mask);
    const auto t0 = std::chrono::high_resolution_clock::now();
    Polyline poly;
    ImageBuffer pred;
    if (algo == "MarchingSquares") {
        poly = MarchingSquares().largest_closed(sdf);
        pred = rasterize_polygon(poly.points, mask.width, mask.height);
    } else {
        DualContouring2D dc;
        poly = dc.extract(sdf);
        pred = rasterize_polygon(poly.points, mask.width, mask.height);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    row.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    row.iou = mask_iou(pred, mask);
    row.f1 = boundary_f1(polyline_to_boundary(poly.points, mask.width, mask.height, true),
                         boundary_pixels(mask), 2);
    save_pgm((fs::path("artifacts") / "contour_bench" / (row.sample + "_" + algo + ".pgm")).string(), pred);
    (void)img;
    return row;
}

static Row eval_coco(const std::string& img_path, const std::string& algo) {
    Row row;
    row.algo = algo;
    row.sample = fs::path(img_path).stem().string();
    const auto img = load_image(img_path);
    const auto coco = parse_coco_json(fs::path(img_path).replace_extension(".json").string());
    if (img.empty() || coco.instances.empty()) {
        return row;
    }
    const auto& inst = coco.instances.front();
    ImageBuffer gt = rasterize_polygon(inst.polygon, img.width, img.height);
    const auto t0 = std::chrono::high_resolution_clock::now();
    ImageBuffer pred;
    Polyline poly;
    if (algo == "GraphCut") {
        pred = GraphCutSegmenter().segment(img, inst.bbox);
        poly.points.clear();
    } else {
        poly = SnakeActiveContour().evolve(img, inst.bbox);
        pred = rasterize_polygon(poly.points, img.width, img.height);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    row.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    row.iou = mask_iou(pred, gt);
    row.f1 = poly.points.empty() ? boundary_f1(boundary_pixels(pred), boundary_pixels(gt), 2)
                                 : boundary_f1(polyline_to_boundary(poly.points, img.width, img.height, true),
                                               boundary_pixels(gt), 2);
    return row;
}

static Row eval_bsds(const std::string& img_path) {
    Row row;
    row.algo = "Livewire";
    row.sample = fs::path(img_path).stem().string();
    const auto img = load_image(img_path);
    const auto gt = load_image(fs::path(img_path).replace_filename(
        fs::path(img_path).stem().string() + "_gt.pgm").string());
    if (img.empty() || gt.empty()) {
        return row;
    }
    auto seeds = Livewire::seeds_from_boundary(gt, 40);
    if (seeds.size() < 2) {
        return row;
    }
    if (seeds.size() > 8) {
        seeds.resize(8);
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    const Polyline poly = Livewire().trace_waypoints(img, seeds);
    const auto t1 = std::chrono::high_resolution_clock::now();
    row.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const auto pb = polyline_to_boundary(poly.points, img.width, img.height, true);
    row.f1 = boundary_f1(pb, gt, 3);
    row.iou = mask_iou(dilate_binary(pb, 1), dilate_binary(gt, 1));
    return row;
}

int main() {
    fs::create_directories(fs::path("artifacts") / "contour_bench");
    const fs::path root = bench_root();
    std::cout << "=== Contour benchmark harness ===\n";
    std::cout << "data: " << root.string() << "\n";
    if (!fs::exists(root)) {
        std::cout << "contour_ready/ missing. Run: python data/vision/prepare_contour_pgms.py\n";
        return 1;
    }

    std::vector<Row> rows;
    auto dis = list_pgms(root / "DIS5K", "_mask");
    const int n_dis = std::min(4, static_cast<int>(dis.size()));
    for (int i = 0; i < n_dis; ++i) {
        rows.push_back(eval_dis(dis[static_cast<size_t>(i)], "MarchingSquares"));
        rows.push_back(eval_dis(dis[static_cast<size_t>(i)], "DualContouring"));
    }
    auto coco = list_pgms(root / "COCO", "_none");
    const int n_coco = std::min(4, static_cast<int>(coco.size()));
    for (int i = 0; i < n_coco; ++i) {
        rows.push_back(eval_coco(coco[static_cast<size_t>(i)], "GraphCut"));
        rows.push_back(eval_coco(coco[static_cast<size_t>(i)], "Snakes"));
    }
    auto bsds = list_pgms(root / "BSDS500", "_gt");
    const int n_bsds = std::min(4, static_cast<int>(bsds.size()));
    for (int i = 0; i < n_bsds; ++i) {
        rows.push_back(eval_bsds(bsds[static_cast<size_t>(i)]));
    }

    print_table(rows);
    if (rows.empty()) {
        std::cout << "No samples evaluated.\n";
        return 1;
    }
    int ok = 0;
    for (const auto& r : rows) {
        if (r.f1 > 0.05 || r.iou > 0.05) {
            ++ok;
        }
    }
    std::cout << "\n" << ok << "/" << rows.size() << " rows produced a non-trivial score.\n";
    return ok > 0 ? 0 : 1;
}
