#include "../contour_kit/contour_kit.hpp"
#include "../boundary_tracing/marching_squares.hpp"
#include "../boundary_tracing/dual_contouring.hpp"
#include "../boundary_tracing/snakes.hpp"
#include "../boundary_tracing/live_wire.hpp"
#include "../boundary_tracing/graph_cut.hpp"
#include "../color/lab_color_space.hpp"
#include "../filters/morph_clean/morph_clean.hpp"
#include "../featurizations/sdf/8SSEDT.hpp"
#include "../bbox/bbox_auto.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace contour;

#ifndef VISION_BENCH_DIR
#define VISION_BENCH_DIR "data/vision/benchmarks"
#endif

struct Row {
    std::string algo;
    std::string dataset;
    std::string sample;
    double iou = 0;
    double f1 = 0;
    double mean_dist = 0;
    double hausdorff = 0;
    double ms = 0;
};

struct BenchmarkSummary {
    double mean_iou = 0;
    double mean_boundary_f1 = 0;
    double mean_hausdorff_px = 0;
    double mean_dist_px = 0;
    double mean_latency_ms = 0;
    double p95_latency_ms = 0;
    int n = 0;
};

static fs::path bench_root() {
    fs::path p(VISION_BENCH_DIR);
    if (fs::exists(p / "contour_ready")) {
        return p / "contour_ready";
    }
    return fs::path("data/vision/benchmarks/contour_ready");
}

static ImageBuffer load_color_or_gray(const std::string& pgm_path) {
    fs::path ppm = fs::path(pgm_path).replace_extension(".ppm");
    if (fs::exists(ppm)) {
        auto im = load_image(ppm.string());
        if (!im.empty()) {
            return im;
        }
    }
    return load_image(pgm_path);
}

static std::vector<std::string> list_pgms(const fs::path& dir, const std::string& skip_suffix) {
    std::vector<std::string> out;
    if (!fs::exists(dir)) {
        return out;
    }
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().extension() != ".pgm") {
            continue;
        }
        if (e.path().filename().string().find(skip_suffix) != std::string::npos) {
            continue;
        }
        out.push_back(e.path().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

static Row fill_mask_metrics(Row row, const ImageBuffer& pred, const ImageBuffer& gt, const ImageBuffer& pred_b,
                             double ms, int tol) {
    row.ms = ms;
    row.iou = mask_iou(pred, gt);
    const auto gt_b = boundary_pixels(gt);
    row.f1 = boundary_f1(pred_b, gt_b, tol);
    const auto pa = collect_on(pred_b);
    const auto ga = collect_on(gt_b);
    row.mean_dist = mean_min_distance(pa, ga);
    row.hausdorff = hausdorff(pa, ga);
    return row;
}

static Row eval_surface(const std::string& img_path, const std::string& algo) {
    Row row{algo, "DIS5K", fs::path(img_path).stem().string()};
    auto mask = load_image(fs::path(img_path)
                               .replace_filename(fs::path(img_path).stem().string() + "_mask.pgm")
                               .string());
    if (mask.empty()) {
        return row;
    }
    mask = MorphClean::clean(mask);
    const Field sdf = ExactSDF::from_mask(mask);
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<Polyline> loops;
    if (algo == "MarchingSquares") {
        loops = MarchingSquares().extract(sdf);
    } else {
        DualContouring2D dc;
        dc.extract(sdf);
        loops = dc.last_loops;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const ImageBuffer pred = xor_fill(loops, mask.width, mask.height);
    ImageBuffer pb = make_gray(mask.width, mask.height, 0);
    for (const auto& p : loops) {
        const auto b = polyline_to_boundary(p.points, mask.width, mask.height, true);
        for (size_t i = 0; i < pb.data.size(); ++i) {
            if (b.data[i]) {
                pb.data[i] = 255;
            }
        }
    }
    return fill_mask_metrics(row, pred, mask, pb,
                             std::chrono::duration<double, std::milli>(t1 - t0).count(), 1);
}

static Row eval_snakes_dis(const std::string& img_path) {
    Row row{"SnakesGVF", "DIS5K", fs::path(img_path).stem().string()};
    const auto img = load_color_or_gray(img_path);
    auto mask = load_image(fs::path(img_path)
                               .replace_filename(fs::path(img_path).stem().string() + "_mask.pgm")
                               .string());
    if (img.empty() || mask.empty()) {
        return row;
    }
    mask = MorphClean::clean(mask);
    const Rect bbox = BBoxAuto::from_mask(mask);
    const auto t0 = std::chrono::high_resolution_clock::now();
    const Polyline poly = SnakeActiveContour().evolve(img, bbox);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const ImageBuffer pred0 = rasterize_polygon(poly.points, mask.width, mask.height);
    ImageBuffer pred = pred0;
    Lab mu_bg{}, mu_fg{};
    int nbg = 0, nfg = 0;
    const Rect inner = {bbox.x + bbox.w * 0.22f, bbox.y + bbox.h * 0.22f, bbox.w * 0.56f, bbox.h * 0.56f};
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            const Lab p = LabColor::at(img, x, y);
            const bool in_box = x >= bbox.x && x < bbox.x1() && y >= bbox.y && y < bbox.y1();
            const bool in_inner = x >= inner.x && x < inner.x1() && y >= inner.y && y < inner.y1();
            if (!in_box) {
                mu_bg.L += p.L;
                mu_bg.a += p.a;
                mu_bg.b += p.b;
                ++nbg;
            } else if (in_inner) {
                mu_fg.L += p.L;
                mu_fg.a += p.a;
                mu_fg.b += p.b;
                ++nfg;
            }
        }
    }
    if (nbg > 0) {
        mu_bg.L /= nbg;
        mu_bg.a /= nbg;
        mu_bg.b /= nbg;
    }
    if (nfg > 0) {
        mu_fg.L /= nfg;
        mu_fg.a /= nfg;
        mu_fg.b /= nfg;
    }
    if (nbg > 0 && nfg > 0 && LabColor::delta2(mu_fg, mu_bg) > 300.0f) {
        for (int y = 0; y < pred.height; ++y) {
            for (int x = 0; x < pred.width; ++x) {
                if (!pred.at(x, y)) {
                    continue;
                }
                const Lab p = LabColor::at(img, x, y);
                if (LabColor::delta2(p, mu_bg) + 80.0f < LabColor::delta2(p, mu_fg)) {
                    pred.at(x, y) = 0;
                }
            }
        }
    }
    const auto pb = polyline_to_boundary(poly.points, mask.width, mask.height, true);
    return fill_mask_metrics(row, pred, mask, pb,
                             std::chrono::duration<double, std::milli>(t1 - t0).count(), 2);
}

static Row eval_graphcut(const std::string& img_path, const ImageBuffer& gt, const Rect& bbox,
                         const std::string& dataset, const std::string& sample) {
    Row row{"GraphCut", dataset, sample};
    const auto img = load_color_or_gray(img_path);
    if (img.empty() || gt.empty()) {
        return row;
    }
    const auto crop = BBoxAuto::crop(img, bbox, 0.15f);
    const auto t0 = std::chrono::high_resolution_clock::now();
    GraphCutSegmenter gc;
    if (dataset == "DIS5K") {
        gc.ellipse_fg = false;
        gc.refine_iters = 2;
    }
    const ImageBuffer pred_c = gc.segment(crop.image, crop.local_bbox);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const ImageBuffer pred = BBoxAuto::uncrop(pred_c, crop);
    return fill_mask_metrics(row, pred, gt, boundary_pixels(pred),
                             std::chrono::duration<double, std::milli>(t1 - t0).count(), 2);
}

static Row eval_graphcut_dis(const std::string& img_path) {
    auto mask = load_image(fs::path(img_path)
                               .replace_filename(fs::path(img_path).stem().string() + "_mask.pgm")
                               .string());
    mask = MorphClean::clean(mask);
    return eval_graphcut(img_path, mask, BBoxAuto::from_mask(mask), "DIS5K",
                         fs::path(img_path).stem().string());
}

static Row eval_graphcut_coco(const std::string& img_path) {
    const auto coco = parse_coco_json(fs::path(img_path).replace_extension(".json").string());
    if (coco.instances.empty()) {
        return {"GraphCut", "COCO", fs::path(img_path).stem().string()};
    }
    const auto& inst = coco.instances.front();
    const auto img = load_color_or_gray(img_path);
    const ImageBuffer gt = rasterize_polygon(inst.polygon, img.width, img.height);
    return eval_graphcut(img_path, gt, inst.bbox, "COCO", fs::path(img_path).stem().string());
}

static Row eval_livewire(const std::string& img_path) {
    Row row{"Livewire", "BSDS500", fs::path(img_path).stem().string()};
    const auto img = load_image(img_path);
    const auto gt = load_image(fs::path(img_path)
                                   .replace_filename(fs::path(img_path).stem().string() + "_gt.pgm")
                                   .string());
    if (img.empty() || gt.empty()) {
        return row;
    }
    const auto chain = Livewire::longest_boundary_chain(gt);
    auto seeds = Livewire::seeds_every_n_px(chain, 30.0f);
    if (seeds.size() < 2) {
        return row;
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    const Polyline poly = Livewire().trace_waypoints(img, seeds, false);
    const auto t1 = std::chrono::high_resolution_clock::now();
    row.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const auto pb = polyline_to_boundary(poly.points, img.width, img.height, false);
    ImageBuffer chain_b = make_gray(gt.width, gt.height, 0);
    for (const auto& p : chain) {
        const int x = std::clamp(static_cast<int>(p.x), 0, gt.width - 1);
        const int y = std::clamp(static_cast<int>(p.y), 0, gt.height - 1);
        chain_b.at(x, y) = 255;
    }
    row.iou = 0;  // open path: not a region metric
    row.f1 = boundary_f1(pb, chain_b, 3);
    row.mean_dist = mean_boundary_distance(pb, chain_b);
    row.hausdorff = hausdorff(collect_on(pb), collect_on(chain_b));
    return row;
}

static BenchmarkSummary summarize(const std::vector<Row>& rows, const std::string& algo,
                                 const std::string& dataset = "") {
    BenchmarkSummary s;
    std::vector<double> lat;
    for (const auto& r : rows) {
        if (r.algo != algo) {
            continue;
        }
        if (!dataset.empty() && r.dataset != dataset) {
            continue;
        }
        s.mean_iou += r.iou;
        s.mean_boundary_f1 += r.f1;
        s.mean_hausdorff_px += r.hausdorff;
        s.mean_dist_px += r.mean_dist;
        s.mean_latency_ms += r.ms;
        lat.push_back(r.ms);
        ++s.n;
    }
    if (s.n == 0) {
        return s;
    }
    s.mean_iou /= s.n;
    s.mean_boundary_f1 /= s.n;
    s.mean_hausdorff_px /= s.n;
    s.mean_dist_px /= s.n;
    s.mean_latency_ms /= s.n;
    std::sort(lat.begin(), lat.end());
    s.p95_latency_ms = lat[static_cast<size_t>(std::min(lat.size() - 1, (lat.size() * 95) / 100))];
    return s;
}

static void write_tsv(const fs::path& path, const std::vector<Row>& rows) {
    std::ofstream out(path);
    out << "algorithm\tdataset\tsample\tiou\tboundary_f1\tmean_dist_px\thausdorff_px\tlatency_ms\n";
    out << std::fixed << std::setprecision(5);
    for (const auto& r : rows) {
        out << r.algo << '\t' << r.dataset << '\t' << r.sample << '\t' << r.iou << '\t' << r.f1 << '\t'
            << r.mean_dist << '\t' << r.hausdorff << '\t' << r.ms << '\n';
    }
}

int main(int argc, char** argv) {
    bool dis_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dis-only") {
            dis_only = true;
        }
    }
    const fs::path art = fs::path("artifacts") / "contour_bench";
    fs::create_directories(art);
    const fs::path root = bench_root();
    std::cout << "=== Contour benchmark harness ===\n";
    std::cout << "data: " << root.string() << "\n";
    if (!fs::exists(root)) {
        std::cout << "contour_ready/ missing. Run: python data/vision/prepare_contour_pgms.py\n";
        return 1;
    }

    std::vector<Row> rows;
    auto dis = list_pgms(root / "DIS5K", "_mask");
    auto coco = list_pgms(root / "COCO", "_none");
    auto bsds = list_pgms(root / "BSDS500", "_gt");
    std::cout << "DIS5K=" << dis.size() << " COCO=" << coco.size() << " BSDS500=" << bsds.size() << "\n";

    for (size_t i = 0; i < dis.size(); ++i) {
        if (!dis_only) {
            rows.push_back(eval_surface(dis[i], "MarchingSquares"));
            rows.push_back(eval_surface(dis[i], "DualContouring"));
        }
        rows.push_back(eval_snakes_dis(dis[i]));
        rows.push_back(eval_graphcut_dis(dis[i]));
        if ((i + 1) % 10 == 0 || i + 1 == dis.size()) {
            std::cout << "  DIS " << (i + 1) << "/" << dis.size() << "\n";
        }
    }
    if (!dis_only) {
        const size_t n_coco = coco.size();
        for (size_t i = 0; i < n_coco; ++i) {
            rows.push_back(eval_graphcut_coco(coco[i]));
            if ((i + 1) % 20 == 0 || i + 1 == n_coco) {
                std::cout << "  COCO GraphCut " << (i + 1) << "/" << n_coco << "\n";
            }
        }
        for (size_t i = 0; i < bsds.size(); ++i) {
            rows.push_back(eval_livewire(bsds[i]));
            if ((i + 1) % 20 == 0 || i + 1 == bsds.size()) {
                std::cout << "  BSDS Livewire " << (i + 1) << "/" << bsds.size() << "\n";
            }
        }
    }

    const fs::path tsv = art / "scores.tsv";
    write_tsv(tsv, rows);
    std::cout << "\nPer-sample scores: " << tsv.string() << "  (" << rows.size() << " rows)\n\n";

    std::cout << std::left << std::setw(22) << "Algorithm" << std::right << std::setw(6) << "N"
              << std::setw(10) << "IoU" << std::setw(10) << "F1" << std::setw(12) << "mean_px"
              << std::setw(12) << "Hausdorff" << std::setw(10) << "ms" << std::setw(10) << "p95ms"
              << "\n";
    std::cout << std::string(88, '-') << "\n";
    std::cout << std::fixed << std::setprecision(3);
    int nonempty = 0;
    const char* algos[] = {"MarchingSquares", "DualContouring", "SnakesGVF", "Livewire"};
    auto print_sum = [&](const std::string& label, const BenchmarkSummary& s) {
        if (s.n == 0) {
            return;
        }
        ++nonempty;
        std::cout << std::left << std::setw(22) << label << std::right << std::setw(6) << s.n
                  << std::setw(10) << s.mean_iou << std::setw(10) << s.mean_boundary_f1
                  << std::setw(12) << s.mean_dist_px << std::setw(12) << s.mean_hausdorff_px
                  << std::setw(10) << s.mean_latency_ms << std::setw(10) << s.p95_latency_ms << "\n";
    };
    for (const char* a : algos) {
        print_sum(a, summarize(rows, a));
    }
    print_sum("GraphCut/DIS5K", summarize(rows, "GraphCut", "DIS5K"));
    print_sum("GraphCut/COCO", summarize(rows, "GraphCut", "COCO"));
    return nonempty > 0 ? 0 : 1;
}
