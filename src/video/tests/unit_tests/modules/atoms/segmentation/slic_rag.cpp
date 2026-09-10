#include "test_harness.hpp"
#include "math/contour_metrics.hpp"
#include "segmentation/helpers/slic_rag/slic_rag.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace slic_rag = segmentation::slic_rag;

namespace {

struct AccuracyChecks {
    int total = 0;
    int passed = 0;
    std::vector<std::string> failures;

    void expect(bool condition, const std::string& message) {
        ++total;
        if (condition) {
            ++passed;
        } else {
            failures.push_back(message);
        }
    }
};

math::ImageBuffer rgb_for_sample(const ProviderLoadedSample* provider,
                                 const vision::GrayImage& luma, int max_side) {
    if (provider != nullptr && !provider->sample.rgb.empty()) {
        math::ImageBuffer rgb = downscale_max_side(provider->sample.rgb, max_side);
        if (rgb.width == luma.width && rgb.height == luma.height && rgb.channels >= 3) {
            return rgb;
        }
    }
    math::ImageBuffer rgb;
    rgb.width = luma.width;
    rgb.height = luma.height;
    rgb.channels = 3;
    rgb.data.resize(static_cast<size_t>(rgb.width * rgb.height * 3));
    for (int y = 0; y < rgb.height; ++y) {
        for (int x = 0; x < rgb.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                rgb.at(x, y, c) = luma.at(x, y);
            }
        }
    }
    return rgb;
}

vision::GrayImage overlay_boundaries(const vision::GrayImage& base,
                                     const std::vector<int>& labels) {
    return overlay_mask(base, label_boundaries(labels, base.width, base.height));
}

void draw_cross(vision::GrayImage& image, int x, int y, uint8_t value) {
    for (int d = -2; d <= 2; ++d) {
        if (x + d >= 0 && x + d < image.width && y >= 0 && y < image.height) {
            image.at(x + d, y) = value;
        }
        if (x >= 0 && x < image.width && y + d >= 0 && y + d < image.height) {
            image.at(x, y + d) = value;
        }
    }
}

vision::GrayImage center_overlay(const vision::GrayImage& base,
                                 const std::vector<slic_rag::Center>& centers) {
    vision::GrayImage output = base;
    for (uint8_t& value : output.data) {
        value = static_cast<uint8_t>(value / 2);
    }
    for (const auto& center : centers) {
        draw_cross(output, static_cast<int>(std::lround(center.x)),
                   static_cast<int>(std::lround(center.y)), 255);
    }
    return output;
}

std::vector<std::pair<float, float>> centroids(const std::vector<int>& labels, int n_labels,
                                               int w, int h) {
    std::vector<double> sx(static_cast<size_t>(n_labels), 0.0);
    std::vector<double> sy(static_cast<size_t>(n_labels), 0.0);
    std::vector<int> count(static_cast<size_t>(n_labels), 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int label = labels[static_cast<size_t>(y * w + x)];
            sx[static_cast<size_t>(label)] += x;
            sy[static_cast<size_t>(label)] += y;
            ++count[static_cast<size_t>(label)];
        }
    }
    std::vector<std::pair<float, float>> result(static_cast<size_t>(n_labels));
    for (int label = 0; label < n_labels; ++label) {
        if (count[static_cast<size_t>(label)] > 0) {
            result[static_cast<size_t>(label)] = {
                static_cast<float>(sx[static_cast<size_t>(label)] /
                                   count[static_cast<size_t>(label)]),
                static_cast<float>(sy[static_cast<size_t>(label)] /
                                   count[static_cast<size_t>(label)])};
        }
    }
    return result;
}

vision::GrayImage rag_overlay(const vision::GrayImage& base,
                              const slic_rag::RagResult& rag,
                              const std::vector<int>& labels) {
    vision::GrayImage output = base;
    for (uint8_t& value : output.data) {
        value = static_cast<uint8_t>(value / 3);
    }
    const auto centers = centroids(labels, rag.initial_regions, base.width, base.height);
    for (const auto& edge : rag.initial_edges_list) {
        const auto& a = centers[static_cast<size_t>(edge.first)];
        const auto& b = centers[static_cast<size_t>(edge.second)];
        plot_line(output, static_cast<int>(std::lround(a.first)),
                  static_cast<int>(std::lround(a.second)),
                  static_cast<int>(std::lround(b.first)),
                  static_cast<int>(std::lround(b.second)), 150);
    }
    for (const auto& center : centers) {
        draw_cross(output, static_cast<int>(std::lround(center.first)),
                   static_cast<int>(std::lround(center.second)), 255);
    }
    return output;
}

bool labels_are_dense(const std::vector<int>& labels, int expected) {
    if (labels.empty() || expected <= 0) {
        return false;
    }
    std::vector<uint8_t> seen(static_cast<size_t>(expected), 0);
    for (int label : labels) {
        if (label < 0 || label >= expected) {
            return false;
        }
        seen[static_cast<size_t>(label)] = 1;
    }
    return std::all_of(seen.begin(), seen.end(), [](uint8_t value) { return value != 0; });
}

bool each_label_is_connected(const std::vector<int>& labels, int n_labels, int w, int h) {
    std::vector<uint8_t> visited(labels.size(), 0);
    std::vector<int> components(static_cast<size_t>(n_labels), 0);
    std::vector<int> stack;
    constexpr int dx[4] = {1, -1, 0, 0};
    constexpr int dy[4] = {0, 0, 1, -1};
    for (int seed = 0; seed < w * h; ++seed) {
        if (visited[static_cast<size_t>(seed)]) {
            continue;
        }
        const int label = labels[static_cast<size_t>(seed)];
        if (++components[static_cast<size_t>(label)] > 1) {
            return false;
        }
        stack.assign(1, seed);
        visited[static_cast<size_t>(seed)] = 1;
        while (!stack.empty()) {
            const int p = stack.back();
            stack.pop_back();
            const int x = p % w;
            const int y = p / w;
            for (int d = 0; d < 4; ++d) {
                const int xx = x + dx[d];
                const int yy = y + dy[d];
                if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                    continue;
                }
                const int q = yy * w + xx;
                if (!visited[static_cast<size_t>(q)] &&
                    labels[static_cast<size_t>(q)] == label) {
                    visited[static_cast<size_t>(q)] = 1;
                    stack.push_back(q);
                }
            }
        }
    }
    return true;
}

vision::GrayImage ground_truth_boundary(const ProviderLoadedSample* provider, int w, int h) {
    if (provider == nullptr) {
        return {};
    }
    vision::GrayImage gt;
    if (!provider->sample.boundary.empty()) {
        gt = provider->sample.boundary;
    } else if (!provider->ground_truth.empty()) {
        gt = to_gray(contour::boundary_pixels(to_contour(binarize_mask(provider->ground_truth))));
    }
    if (!gt.empty() && (gt.width != w || gt.height != h)) {
        gt = resize_nearest(gt, w, h);
    }
    return binarize_mask(gt);
}

class SlicRagAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"slic_rag"};
    AccuracyChecks checks;
    std::ostringstream values;
    std::ostringstream nodes_tsv;
    std::ostringstream edges_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load full-image segmentation samples");
        const auto mission =
            load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, max_side);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        report.n_inputs = static_cast<int>(samples.size());
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name
                  << " (max side " << max_side << ")\n";
        return !samples.empty();
    }

    void run(const std::string& artifact_dir) {
        print_banner("run SLIC -> connectivity -> RAG contraction");
        ScopedTimer timer(&report.elapsed_ms);
        values << "file\tseed_centers\tslic_regions\tconnected_regions\trag_edges\tmerges"
                  "\tcleanup_merges\tfinal_regions\tslic_iters\tchanged_last\tlast_cost"
                  "\tboundary_f1\tms\n";
        nodes_tsv << "file\tnode\tarea\tmean_r\tmean_g\tmean_b\tdegree\n";
        edges_tsv << "file\tu\tv\tinitial_cost\n";
        double f1_sum = 0.0;
        int f1_count = 0;

        for (size_t index = 0; index < samples.size(); ++index) {
            const LoadedSample& sample = samples[index];
            const ProviderLoadedSample* provider =
                index < provider_samples.size() ? &provider_samples[index] : nullptr;
            const vision::GrayImage luma = mission_luma_image(provider, sample.image);
            const math::ImageBuffer rgb = rgb_for_sample(provider, luma, max_side);

            slic_rag::Slic slic;
            slic.iterations = slic_iterations;
            slic.compactness = compactness;
            slic_rag::RagHierarchicalMerger merger;
            merger.target_regions = target_regions;
            merger.max_cost = max_cost;
            merger.min_region_fraction = min_region_fraction;

            double elapsed_ms = 0.0;
            slic_rag::SlicResult superpixels;
            slic_rag::RagResult rag;
            {
                ScopedTimer sample_timer(&elapsed_ms);
                superpixels = slic.segment(rgb, desired_superpixels);
                rag = merger.merge(superpixels, rgb);
            }

            const vision::GrayImage predicted_boundary =
                label_boundaries(rag.labels, luma.width, luma.height);
            const vision::GrayImage gt =
                ground_truth_boundary(provider, luma.width, luma.height);
            double boundary_f1 = 0.0;
            if (!gt.empty()) {
                boundary_f1 =
                    contour::boundary_f1(to_contour(predicted_boundary), to_contour(gt), 2);
                f1_sum += boundary_f1;
                ++f1_count;
            }

            const std::string stem = stem_of(sample.row.file);
            validate(stem, superpixels, rag);
            write_artifacts(artifact_dir, stem, luma, gt, superpixels, rag);
            write_graph_tables(stem, rag);

            values << sample.row.file << '\t' << superpixels.seeds.size() << '\t'
                   << superpixels.raw_regions << '\t' << superpixels.connected_regions << '\t'
                   << rag.initial_edges << '\t' << rag.merges << '\t' << rag.cleanup_merges
                   << '\t' << rag.final_regions
                   << '\t' << superpixels.iterations_run << '\t' << superpixels.changed_last
                   << '\t' << rag.last_merge_cost << '\t' << boundary_f1 << '\t'
                   << elapsed_ms << '\n';
            std::cout << "  " << sample.row.file << "  SLIC="
                      << superpixels.connected_regions << " -> RAG=" << rag.final_regions
                      << "  merges=" << rag.merges << "  boundary_f1=" << std::fixed
                      << std::setprecision(3) << boundary_f1 << "  " << elapsed_ms << " ms\n";
            ++report.n_outputs;
        }

        std::ostringstream note;
        note << "SLIC K=" << desired_superpixels << ", m=" << compactness
             << ", target=" << target_regions << ", max_cost=" << max_cost;
        if (f1_count > 0) {
            note << ", mean boundary F1=" << std::fixed << std::setprecision(3)
                 << f1_sum / f1_count;
        }
        report.notes.push_back(note.str());
        if (f1_count >= 4) {
            checks.expect(f1_sum / f1_count >= 0.30,
                          "dataset: mean boundary F1 remains above 0.30");
        }
    }

    void write(const std::string& artifact_dir) {
        vision::write_text_file(vision::join_path(artifact_dir, "slic_rag.tsv"),
                                values.str());
        vision::write_text_file(vision::join_path(artifact_dir, "rag_nodes.tsv"),
                                nodes_tsv.str());
        vision::write_text_file(vision::join_path(artifact_dir, "rag_edges.tsv"),
                                edges_tsv.str());
        written.insert(written.begin(), "rag_edges.tsv");
        written.insert(written.begin(), "rag_nodes.tsv");
        written.insert(written.begin(), "slic_rag.tsv");
        write_atom_manifest(artifact_dir, report, written);
        report.print();
        std::cout << "invariants: " << checks.passed << " / " << checks.total << " passed\n";
        for (const std::string& failure : checks.failures) {
            std::cout << "  FAIL  " << failure << '\n';
        }
        std::cout << "artifacts -> " << artifact_dir << '\n';
    }

    int status() const { return checks.failures.empty() ? 0 : 1; }

private:
    int max_side = 192;
    int desired_superpixels = 180;
    int target_regions = 20;
    int slic_iterations = 10;
    float compactness = 12.0f;
    float max_cost = 0.50f;
    float min_region_fraction = 0.005f;

    void validate(const std::string& stem, const slic_rag::SlicResult& slic,
                  const slic_rag::RagResult& rag) {
        const int n = slic.width * slic.height;
        checks.expect(static_cast<int>(slic.labels.size()) == n,
                      stem + ": SLIC labels cover every pixel");
        checks.expect(labels_are_dense(slic.labels, slic.connected_regions),
                      stem + ": connected SLIC labels are dense");
        checks.expect(each_label_is_connected(slic.labels, slic.connected_regions,
                                              slic.width, slic.height),
                      stem + ": every SLIC label is 4-connected");
        checks.expect(static_cast<int>(rag.labels.size()) == n,
                      stem + ": final labels cover every pixel");
        checks.expect(labels_are_dense(rag.labels, rag.final_regions),
                      stem + ": final labels are dense");
        checks.expect(each_label_is_connected(rag.labels, rag.final_regions,
                                              slic.width, slic.height),
                      stem + ": every final region is 4-connected");
        checks.expect(rag.final_regions <= rag.initial_regions,
                      stem + ": contraction never increases region count");
        checks.expect(rag.final_regions >= 1,
                      stem + ": at least one final region survives");
        checks.expect(rag.last_merge_cost <= max_cost + 1e-6f || rag.merges == 0,
                      stem + ": threshold-governed merges obey the cost ceiling");
        checks.expect(smallest_region(rag.labels, rag.final_regions) >=
                          std::min(rag.final_regions == 1
                                       ? n
                                       : static_cast<int>(min_region_fraction * n),
                                   n),
                      stem + ": closure pass leaves no sub-minimum region");
    }

    static int smallest_region(const std::vector<int>& labels, int n_labels) {
        if (labels.empty() || n_labels <= 0) {
            return 0;
        }
        std::vector<int> areas(static_cast<size_t>(n_labels), 0);
        for (int label : labels) {
            ++areas[static_cast<size_t>(label)];
        }
        return *std::min_element(areas.begin(), areas.end());
    }

    void save(const std::string& dir, const std::string& name,
              const vision::GrayImage& image) {
        vision::save_pgm(vision::join_path(dir, name), image);
        written.push_back(name);
    }

    void write_graph_tables(const std::string& stem, const slic_rag::RagResult& rag) {
        for (size_t node = 0; node < rag.initial_nodes.size(); ++node) {
            const auto& value = rag.initial_nodes[node];
            nodes_tsv << stem << '\t' << node << '\t' << value.area << '\t'
                      << value.mean[0] << '\t' << value.mean[1] << '\t'
                      << value.mean[2] << '\t' << value.neighbors.size() << '\n';
        }
        for (size_t edge = 0; edge < rag.initial_edges_list.size(); ++edge) {
            edges_tsv << stem << '\t' << rag.initial_edges_list[edge].first << '\t'
                      << rag.initial_edges_list[edge].second << '\t'
                      << rag.initial_edge_costs[edge] << '\n';
        }
    }

    void write_artifacts(const std::string& dir, const std::string& stem,
                         const vision::GrayImage& luma,
                         const vision::GrayImage& gt,
                         const slic_rag::SlicResult& slic,
                         const slic_rag::RagResult& rag) {
        save(dir, stem + "_00_input_base.pgm", luma);
        save(dir, stem + "_01_seed_grid_overlay.pgm",
             center_overlay(luma, slic.seeds));
        save(dir, stem + "_02_local_assignment_labels.pgm",
             colorize_labels(slic.first_labels, slic.width, slic.height));
        save(dir, stem + "_02_local_assignment_overlay.pgm",
             overlay_boundaries(luma, slic.first_labels));
        save(dir, stem + "_03_converged_slic_labels.pgm",
             colorize_labels(slic.raw_labels, slic.width, slic.height));
        save(dir, stem + "_03_centroid_overlay.pgm",
             center_overlay(overlay_boundaries(luma, slic.raw_labels), slic.centers));
        save(dir, stem + "_04_connectivity_labels.pgm",
             colorize_labels(slic.labels, slic.width, slic.height));
        save(dir, stem + "_04_connectivity_overlay.pgm",
             overlay_boundaries(luma, slic.labels));
        save(dir, stem + "_05_rag_overlay.pgm",
             rag_overlay(luma, rag, slic.labels));
        save(dir, stem + "_06_contraction_mid_labels.pgm",
             colorize_labels(rag.middle_labels, slic.width, slic.height));
        save(dir, stem + "_06_contraction_mid_overlay.pgm",
             overlay_boundaries(luma, rag.middle_labels));
        save(dir, stem + "_07_final_labels.pgm",
             colorize_labels(rag.labels, slic.width, slic.height));
        save(dir, stem + "_07_final_overlay.pgm",
             overlay_boundaries(luma, rag.labels));
        if (!gt.empty()) {
            save(dir, stem + "_gt_boundary_overlay.pgm", overlay_mask(luma, gt));
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "bsds500", [&](const AtomCli& cli) -> int {
        SlicRagAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for SLIC-RAG atom\n";
            return 1;
        }
        if (cli.list_only) {
            for (const auto& sample : atom.samples) {
                std::cout << "  " << sample.row.file << '\n';
            }
            return 0;
        }
        bool artifacts_overridden = false;
        for (int i = 1; i < argc; ++i) {
            artifacts_overridden = artifacts_overridden ||
                                   std::string(argv[i]) == "--artifacts";
        }
        const std::string artifact_request =
            artifacts_overridden ? cli.artifact_dir
                                 : vision::join_path(cli.artifact_dir, "slic_rag");
        const std::string artifacts = make_artifact_dir(artifact_request);
        atom.run(artifacts);
        atom.write(artifacts);
        return atom.status();
    });
}
