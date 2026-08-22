#pragma once

#include "math/contour_compat.hpp"
#include "math/contour_metrics.hpp"
#include "topology/euler/euler_characteristic.hpp"
#include "segmentation/ccl/connected_components.hpp"

#include <array>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace mission {

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        default:
            out += c;
        }
    }
    return out;
}

inline std::string json_num(double v, int prec = 6) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

inline double bbox_iou(const vision::Rect& a, const vision::Rect& b) {
    const float x0 = std::max(a.x, b.x);
    const float y0 = std::max(a.y, b.y);
    const float x1 = std::min(a.x1(), b.x1());
    const float y1 = std::min(a.y1(), b.y1());
    const float inter = std::max(0.0f, x1 - x0) * std::max(0.0f, y1 - y0);
    const float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0.0f ? static_cast<double>(inter / uni) : 0.0;
}

inline vision::GrayImage rotate_gray_90(const vision::GrayImage& src) {
    vision::GrayImage out;
    out.width = src.height;
    out.height = src.width;
    out.data.resize(static_cast<size_t>(out.width * out.height));
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            out.at(src.height - 1 - y, x) = src.at(x, y);
        }
    }
    return out;
}

inline contour::ImageBuffer to_contour_buf(const vision::GrayImage& g) {
    contour::ImageBuffer im;
    im.width = g.width;
    im.height = g.height;
    im.channels = 1;
    im.data = g.data;
    return im;
}

inline double chamfer_polyline(const contour::Polyline& poly, const vision::GrayImage& gt) {
    if (gt.empty() || poly.points.size() < 2) {
        return 0.0;
    }
    const auto gt_buf = to_contour_buf(gt);
    return contour::mean_boundary_distance(
        contour::polyline_to_boundary(poly.points, gt.width, gt.height, poly.closed),
        contour::boundary_pixels(gt_buf));
}

inline double rmse_polyline(const contour::Polyline& poly, const vision::GrayImage& gt) {
    if (gt.empty() || poly.points.size() < 2) {
        return 0.0;
    }
    const auto gt_pts = contour::collect_on(contour::boundary_pixels(to_contour_buf(gt)));
    if (gt_pts.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto& p : poly.points) {
        float best = 1.0e9f;
        for (const auto& g : gt_pts) {
            best = std::min(best, contour::dist(p, g));
        }
        sum += static_cast<double>(best * best);
    }
    return std::sqrt(sum / static_cast<double>(poly.points.size()));
}

inline std::optional<int> expected_glyph_chi(const std::string& label) {
    if (label.find("glyph_B") != std::string::npos || label.find("glyph_8") != std::string::npos) {
        return -1;
    }
    if (label.find("glyph_O") != std::string::npos || label.find("donut") != std::string::npos) {
        return 0;
    }
    if (label.find("glyph_1") != std::string::npos) {
        return 1;
    }
    return std::nullopt;
}

inline double l2_delta(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 1.0;
    }
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return std::sqrt(s / static_cast<double>(a.size()));
}

inline double l2_delta7(const std::array<double, 7>& a, const std::array<double, 7>& b) {
    double s = 0.0;
    for (size_t i = 0; i < 7; ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s / 7.0);
}

template <typename Point>
inline void write_polyline_svg(const std::string& path, int w, int h, const std::vector<Point>& pts,
                               bool closed) {
    std::ostringstream ss;
    ss << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\"" << h
       << "\" viewBox=\"0 0 " << w << " " << h << "\">\n<polyline fill=\"none\" stroke=\"white\" "
          "stroke-width=\"1.5\" points=\"";
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i) {
            ss << ' ';
        }
        ss << std::fixed << std::setprecision(2) << pts[i].x << ',' << pts[i].y;
    }
    if (closed && pts.size() > 1) {
        ss << ' ' << std::fixed << std::setprecision(2) << pts.front().x << ',' << pts.front().y;
    }
    ss << "\"/>\n</svg>\n";
    vision::write_text_file(path, ss.str());
}

inline void write_mesh_vertices_json(const std::string& path, const std::vector<contour::Vec2>& verts) {
    std::ostringstream ss;
    ss << "{\n  \"vertices\": [\n";
    for (size_t i = 0; i < verts.size(); ++i) {
        ss << "    {\"x\": " << json_num(verts[i].x, 3) << ", \"y\": " << json_num(verts[i].y, 3)
           << "}";
        if (i + 1 < verts.size()) {
            ss << ',';
        }
        ss << '\n';
    }
    ss << "  ]\n}\n";
    vision::write_text_file(path, ss.str());
}

inline void write_bbox_json(const std::string& path, const std::string& file,
                            const std::vector<vision::ConnectedComponentLabeler::Component>& comps) {
    std::ostringstream ss;
    ss << "{\n  \"file\": \"" << json_escape(file) << "\",\n  \"boxes\": [\n";
    for (size_t i = 0; i < comps.size(); ++i) {
        const auto& c = comps[i];
        ss << "    {\"label\": " << c.label << ", \"area\": " << c.area << ", \"x\": "
           << json_num(c.bbox.x, 2) << ", \"y\": " << json_num(c.bbox.y, 2) << ", \"w\": "
           << json_num(c.bbox.w, 2) << ", \"h\": " << json_num(c.bbox.h, 2) << "}";
        if (i + 1 < comps.size()) {
            ss << ',';
        }
        ss << '\n';
    }
    ss << "  ]\n}\n";
    vision::write_text_file(path, ss.str());
}

inline void write_hu_json(const std::string& path, const std::string& file,
                          const std::array<double, 7>& hu, double rot_delta) {
    std::ostringstream ss;
    ss << "{\n  \"file\": \"" << json_escape(file) << "\",\n  \"rot_delta\": " << json_num(rot_delta, 8)
       << ",\n  \"hu\": [";
    for (size_t i = 0; i < hu.size(); ++i) {
        if (i) {
            ss << ", ";
        }
        ss << json_num(hu[i], 8);
    }
    ss << "]\n}\n";
    vision::write_text_file(path, ss.str());
}

inline void write_convergence_json(const std::string& path, const std::string& file, int iterations,
                                   double chamfer, double area) {
    std::ostringstream ss;
    ss << "{\n  \"file\": \"" << json_escape(file) << "\",\n  \"iterations\": " << iterations
       << ",\n  \"chamfer_px\": " << json_num(chamfer, 4) << ",\n  \"final_area\": "
       << json_num(area, 2) << "\n}\n";
    vision::write_text_file(path, ss.str());
}

inline void write_topological_tree_json(const std::string& path, const std::string& file, int components,
                                        int holes, int chi, std::optional<int> expected) {
    std::ostringstream ss;
    ss << "{\n  \"file\": \"" << json_escape(file) << "\",\n  \"components\": " << components
       << ",\n  \"holes\": " << holes << ",\n  \"chi\": " << chi;
    if (expected.has_value()) {
        ss << ",\n  \"expected_chi\": " << *expected << ",\n  \"genus_ok\": "
           << (chi == *expected ? "true" : "false");
    }
    ss << "\n}\n";
    vision::write_text_file(path, ss.str());
}

inline void write_l2_matrix_json(const std::string& path, const std::vector<std::string>& labels,
                                 const std::vector<std::vector<float>>& dists) {
    std::ostringstream ss;
    ss << "{\n  \"labels\": [";
    for (size_t i = 0; i < labels.size(); ++i) {
        if (i) {
            ss << ", ";
        }
        ss << '"' << json_escape(labels[i]) << '"';
    }
    ss << "],\n  \"matrix\": [\n";
    for (size_t i = 0; i < dists.size(); ++i) {
        ss << "    [";
        for (size_t j = 0; j < dists[i].size(); ++j) {
            if (j) {
                ss << ", ";
            }
            ss << json_num(dists[i][j], 6);
        }
        ss << ']';
        if (i + 1 < dists.size()) {
            ss << ',';
        }
        ss << '\n';
    }
    ss << "  ]\n}\n";
    vision::write_text_file(path, ss.str());
}

inline void write_rtree_json(const std::string& path, size_t n_boxes, size_t n_queries,
                             bool queries_match_brute) {
    std::ostringstream ss;
    ss << "{\n  \"n_boxes\": " << n_boxes << ",\n  \"n_queries\": " << n_queries
       << ",\n  \"exact_intersection\": " << (queries_match_brute ? "true" : "false") << "\n}\n";
    vision::write_text_file(path, ss.str());
}

inline void write_vp_partitions_json(const std::string& path, size_t n_samples, size_t nn_exact_matches) {
    std::ostringstream ss;
    ss << "{\n  \"n_samples\": " << n_samples << ",\n  \"nn_exact_matches\": " << nn_exact_matches
       << "\n}\n";
    vision::write_text_file(path, ss.str());
}

inline vision::GrayImage cost_to_gray(const std::vector<float>& cost, int w, int h) {
    vision::GrayImage g;
    g.width = w;
    g.height = h;
    g.data.resize(static_cast<size_t>(w * h));
    float hi = 1e-6f;
    for (float v : cost) {
        hi = std::max(hi, v);
    }
    for (size_t i = 0; i < g.data.size() && i < cost.size(); ++i) {
        g.data[i] = static_cast<uint8_t>(std::clamp(cost[i] / hi * 255.0f, 0.0f, 255.0f));
    }
    return g;
}

inline vision::GrayImage gvf_magnitude(const contour::Field& u, const contour::Field& v) {
    vision::GrayImage g;
    g.width = u.width;
    g.height = u.height;
    g.data.resize(static_cast<size_t>(u.width * u.height));
    float hi = 1e-6f;
    for (size_t i = 0; i < g.data.size(); ++i) {
        const float m = std::sqrt(u.data[i] * u.data[i] + v.data[i] * v.data[i]);
        hi = std::max(hi, m);
    }
    for (size_t i = 0; i < g.data.size(); ++i) {
        const float m = std::sqrt(u.data[i] * u.data[i] + v.data[i] * v.data[i]);
        g.data[i] = static_cast<uint8_t>(std::clamp(m / hi * 255.0f, 0.0f, 255.0f));
    }
    return g;
}

}  // namespace mission
