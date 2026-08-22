#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"
#include "structures/vp_tree/vp_tree.hpp"

#include <limits>
#include <sstream>

// Atom demo: masks in → Chamfer distances / VP-tree NN out.
class VpTreeAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"vp_tree"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 128);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run VP-tree → Chamfer NN table");
        ScopedTimer timer(&report.elapsed_ms);
        std::vector<std::vector<vision::Vec2>> contours(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            auto c = vision::MooreNeighborTracer::trace(samples[i].image);
            contours[i] = vision::MooreNeighborTracer::resample(c.points, 24);
        }
        vision::VPTree<int> tree;
        std::vector<int> ids(samples.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = static_cast<int>(i);
        }
        tree.build(ids, [&](const int& a, const int& b) {
            return vision::VPTree<int>::chamfer(contours[static_cast<size_t>(a)],
                                                contours[static_cast<size_t>(b)]);
        });
        values_tsv << "file\tlabel\tnn_id\tnn_file\tnn_label\tchamfer\ttree_nn_id\ttree_dist\tnn_exact\n";
        size_t nn_exact = 0;
        for (size_t i = 0; i < samples.size(); ++i) {
            float best_d = std::numeric_limits<float>::max();
            int best = -1;
            for (size_t j = 0; j < samples.size(); ++j) {
                if (i == j) {
                    continue;
                }
                const float d =
                    vision::VPTree<int>::chamfer(contours[i], contours[static_cast<size_t>(j)]);
                if (d < best_d) {
                    best_d = d;
                    best = static_cast<int>(j);
                }
            }
            float tree_d = 0;
            const int tree_nn = tree.nearest(static_cast<int>(i), &tree_d);
            const bool exact = tree_nn == best;
            if (exact) {
                ++nn_exact;
            }
            std::string nn_file = "?", nn_label = "?";
            if (best >= 0) {
                nn_file = samples[static_cast<size_t>(best)].row.file;
                nn_label = samples[static_cast<size_t>(best)].row.label;
            }
            std::cout << "  " << samples[i].row.file << "  nn=" << nn_file << "  chamfer=" << best_d
                      << "  tree_nn=" << tree_nn << "\n";
            values_tsv << samples[i].row.file << '\t' << samples[i].row.label << '\t' << best << '\t'
                       << nn_file << '\t' << nn_label << '\t' << best_d << '\t' << tree_nn << '\t'
                       << tree_d << '\t' << (exact ? 1 : 0) << '\n';
            const std::string stem = stem_of(samples[i].row.file);
            const std::string in_name = stem + "_input.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), samples[i].image);
            written.push_back(in_name);
            ++report.n_outputs;
        }
        mission::write_vp_partitions_json(vision::join_path(art_dir, "vp_tree_partitions.json"),
                                          samples.size(), nn_exact);
        written.push_back("vp_tree_partitions.json");
        report.notes.push_back("artifacts: query_bench.tsv, vp_tree_partitions.json");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "query_bench.tsv"), values_tsv.str());
        written.insert(written.begin(), "query_bench.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_vptree";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        VpTreeAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for " << cli.dataset << " under " << cli.data_root << "\n";
            return 1;
        }
        if (cli.list_only) {
            for (const auto& s : atom.samples) {
                std::cout << "  " << s.row.file << "\n";
            }
            return 0;
        }
        const std::string art = make_artifact_dir(cli.artifact_dir);
        atom.run(art);
        atom.write(art);
        return 0;
    });
}
