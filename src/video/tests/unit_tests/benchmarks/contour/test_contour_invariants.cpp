#include "../contour_kit/contour_kit.hpp"
#include "../boundary_tracing/marching_squares.hpp"
#include "../boundary_tracing/dual_contouring.hpp"
#include "../boundary_tracing/snakes.hpp"
#include "../boundary_tracing/level_set.hpp"
#include "../boundary_tracing/live_wire.hpp"
#include "../boundary_tracing/graph_cut.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>

namespace fs = std::filesystem;
using namespace contour;

static fs::path artifacts_dir() {
    return fs::path("artifacts") / "contour_invariants";
}

static int g_fail = 0;

static void check(bool cond, const std::string& name, const std::string& detail) {
    if (cond) {
        std::cout << "  [ok] " << name << "  " << detail << "\n";
    } else {
        std::cout << "  [FAIL] " << name << "  " << detail << "\n";
        ++g_fail;
    }
}

static void test_qef_corner() {
    DualContouring2D::Hermite a{{20.3f, 20.0f}, {1.0f, 0.0f}};
    DualContouring2D::Hermite b{{20.0f, 20.3f}, {0.0f, 1.0f}};
    const Vec2 x = DualContouring2D::solve_qef({a, b}, {20.5f, 20.5f});
    const float err = dist(x, {20.3f, 20.3f});
    check(err < 0.08f, "QEF 90-deg corner", "err=" + std::to_string(err));
}

static void test_dual_contouring_box() {
    const Field sdf = ChamferSDF::analytic_box_sdf(64, 64, 32.0f, 32.0f, 12.3f, 12.3f);
    DualContouring2D dc;
    const auto t0 = std::chrono::high_resolution_clock::now();
    const Polyline poly = dc.extract(sdf);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    float best = 1e9f;
    for (const auto& v : dc.cell_vertices) {
        best = std::min(best, dist(v, {19.7f, 19.7f}));
    }
    check(poly.closed && poly.points.size() >= 4, "DC closed loop",
          "n=" + std::to_string(poly.points.size()) + " ms=" + std::to_string(ms));
    check(best < 0.85f, "DC reconstructs sharp box corner", "corner_err=" + std::to_string(best));
    save_pgm((artifacts_dir() / "dc_box_sdf.pgm").string(), [&] {
        ImageBuffer im = make_gray(64, 64);
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x)
                im.at(x, y) = sdf.at(x, y) <= 0 ? 255 : 0;
        return im;
    }());
}

static void test_marching_squares_circle() {
    const float r = 18.0f;
    const Field sdf = ChamferSDF::analytic_circle_sdf(64, 64, 32.0f, 32.0f, r);
    MarchingSquares ms;
    const auto t0 = std::chrono::high_resolution_clock::now();
    const Polyline loop = ms.largest_closed(sdf);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms_t = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const float area = shoelace(loop.points);
    const float expect = kPi * r * r;
    const float rel = std::fabs(area - expect) / expect;
    check(loop.closed && loop.points.size() >= 16, "MS closed loop",
          "n=" + std::to_string(loop.points.size()));
    check(rel < 0.12f, "MS preserves circular area pi r^2",
          "area=" + std::to_string(area) + " expect=" + std::to_string(expect) +
              " rel=" + std::to_string(rel) + " ms=" + std::to_string(ms_t));
}

static void test_chan_vese_noisy_blob() {
    ImageBuffer im = make_gray(64, 64, 40);
    ImageBuffer gt = make_gray(64, 64, 0);
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0f, 18.0f);
    const float cx = 32.0f, cy = 32.0f, r = 14.0f;
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            const float d = std::hypot(x - cx, y - cy);
            float v = d < r ? 190.0f : 40.0f;
            v += noise(rng);
            im.at(x, y) = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
            if (d < r) {
                gt.at(x, y) = 255;
            }
        }
    }
    ChanVeseLevelSet cv;
    cv.iterations = 40;
    const auto t0 = std::chrono::high_resolution_clock::now();
    const Polyline p = cv.segment(im, {18, 18, 28, 28});
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const ImageBuffer pred = rasterize_polygon(p.points, 64, 64);
    const double iou = mask_iou(pred, gt);
    check(iou > 0.55, "Chan-Vese noisy blob (no hard edges)",
          "IoU=" + std::to_string(iou) + " ms=" + std::to_string(ms));
    save_pgm((artifacts_dir() / "chanvese_pred.pgm").string(), pred);
}

static void test_snakes_and_graphcut() {
    ImageBuffer im = make_gray(80, 80, 20);
    ImageBuffer gt = make_gray(80, 80, 0);
    for (int y = 0; y < 80; ++y) {
        for (int x = 0; x < 80; ++x) {
            if (x >= 22 && x <= 58 && y >= 22 && y <= 58) {
                im.at(x, y) = 210;
                gt.at(x, y) = 255;
            }
        }
    }
    const Rect box{18, 18, 44, 44};
    SnakeActiveContour snake;
    snake.iterations = 60;
    const auto t0 = std::chrono::high_resolution_clock::now();
    const Polyline sp = snake.evolve(im, box);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const ImageBuffer sm = rasterize_polygon(sp.points, 80, 80);
    const double siou = mask_iou(sm, gt);
    check(siou > 0.45, "Snakes parametric contour",
          "IoU=" + std::to_string(siou) + " ms=" +
              std::to_string(std::chrono::duration<double, std::milli>(t1 - t0).count()));

    GraphCutSegmenter gc;
    const auto t2 = std::chrono::high_resolution_clock::now();
    const ImageBuffer gm = gc.segment(im, box);
    const auto t3 = std::chrono::high_resolution_clock::now();
    const double giou = mask_iou(gm, gt);
    check(giou > 0.5, "Graph-cut bbox prior",
          "IoU=" + std::to_string(giou) + " ms=" +
              std::to_string(std::chrono::duration<double, std::milli>(t3 - t2).count()));
}

static void test_livewire() {
    ImageBuffer im = make_gray(64, 64, 30);
    ImageBuffer gt = make_gray(64, 64, 0);
    for (int x = 12; x <= 50; ++x) {
        im.at(x, 12) = 220;
        im.at(x, 50) = 220;
        gt.at(x, 12) = 255;
        gt.at(x, 50) = 255;
    }
    for (int y = 12; y <= 50; ++y) {
        im.at(12, y) = 220;
        im.at(50, y) = 220;
        gt.at(12, y) = 255;
        gt.at(50, y) = 255;
    }
    Livewire lw;
    const std::vector<Vec2> seeds{{12, 12}, {50, 12}, {50, 50}, {12, 50}};
    const auto t0 = std::chrono::high_resolution_clock::now();
    const Polyline p = lw.trace_waypoints(im, seeds);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto pb = polyline_to_boundary(p.points, 64, 64, true);
    const double f1 = boundary_f1(pb, gt, 2);
    check(p.points.size() >= 8 && f1 > 0.4, "Livewire Dijkstra rectangle",
          "F1=" + std::to_string(f1) + " n=" + std::to_string(p.points.size()) + " ms=" +
              std::to_string(std::chrono::duration<double, std::milli>(t1 - t0).count()));
}

int main() {
    fs::create_directories(artifacts_dir());
    std::cout << "=== Contour invariant unit tests ===\n";
    test_qef_corner();
    test_dual_contouring_box();
    test_marching_squares_circle();
    test_chan_vese_noisy_blob();
    test_snakes_and_graphcut();
    test_livewire();
    std::cout << (g_fail ? "FAILED " : "PASSED ") << (6 - g_fail) << "/6 groups, failures=" << g_fail
              << "\n";
    return g_fail ? 1 : 0;
}
