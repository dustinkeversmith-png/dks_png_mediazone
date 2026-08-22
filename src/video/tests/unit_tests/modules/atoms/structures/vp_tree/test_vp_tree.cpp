#include "test_harness.hpp"
#include "boundary_tracing/moore_neighbor.hpp"
#include "spatial_trees/vp_tree/vp_tree.hpp"

#include <limits>
#include <sstream>

// Atom demo: masks in → Chamfer distances / VP-tree NN out.
class VpTreeAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"vp_tree"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_png_dataset(vision::dataset_dir(root, dataset), sample_filter);
        if (samples.empty()) {
            samples = load_png_dataset(vision::dataset_dir(root, "unit_fourier"), sample_filter);
        }
        std::cout << "loaded " << samples.size() << " silhouettes\n";
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
        values_tsv << "file\tlabel\tnn_id\tnn_file\tnn_label\tchamfer\ttree_nn_id\ttree_dist\n";
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
            std::string nn_file = "?", nn_label = "?";
            if (best >= 0) {
                nn_file = samples[static_cast<size_t>(best)].row.file;
                nn_label = samples[static_cast<size_t>(best)].row.label;
            }
            std::cout << "  " << samples[i].row.file << "  nn=" << nn_file << "  chamfer=" << best_d
                      << "  tree_nn=" << tree_nn << "\n";
            values_tsv << samples[i].row.file << '\t' << samples[i].row.label << '\t' << best << '\t'
                       << nn_file << '\t' << nn_label << '\t' << best_d << '\t' << tree_nn << '\t'
                       << tree_d << '\n';
            const std::string stem = stem_of(samples[i].row.file);
            const std::string in_name = stem + "_input.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), samples[i].image);
            written.push_back(in_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: vptree_nn.tsv, *_input.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "vptree_nn.tsv"), values_tsv.str());
        written.insert(written.begin(), "vptree_nn.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_vptree";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        VpTreeAtom atom;
        if (!atom.load(cli.data_root, cli.dataset, cli.sample_filter)) {
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
