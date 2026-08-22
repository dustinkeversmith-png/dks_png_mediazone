#include "test_harness.hpp"
#include "screen_detection/CCL/connected_components.hpp"

#include <filesystem>
#include <sstream>

// Atom demo: screen mask in → component boxes + overlay images out.
class CclAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"ccl"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_png_dataset(vision::dataset_dir(root, dataset), sample_filter);
        std::cout << "loaded " << samples.size() << " screens\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run CCL → components + overlays");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel_id\tarea\tx\ty\tw\th\n";
        for (const auto& sample : samples) {
            auto ccl = vision::ConnectedComponentLabeler::label(sample.image);
            std::cout << "  " << sample.row.file << "  components=" << ccl.components.size() << "\n";
            vision::GrayImage vis = sample.image;
            for (const auto& c : ccl.components) {
                values_tsv << sample.row.file << '\t' << c.label << '\t' << c.area << '\t' << c.bbox.x
                           << '\t' << c.bbox.y << '\t' << c.bbox.w << '\t' << c.bbox.h << '\n';
                const int x0 = static_cast<int>(c.bbox.x);
                const int y0 = static_cast<int>(c.bbox.y);
                const int x1 = static_cast<int>(c.bbox.x1());
                const int y1 = static_cast<int>(c.bbox.y1());
                for (int x = x0; x < x1 && x < vis.width; ++x) {
                    if (y0 >= 0 && y0 < vis.height) {
                        vis.at(x, y0) = 128;
                    }
                    if (y1 - 1 >= 0 && y1 - 1 < vis.height) {
                        vis.at(x, y1 - 1) = 128;
                    }
                }
                for (int y = y0; y < y1 && y < vis.height; ++y) {
                    if (x0 >= 0 && x0 < vis.width) {
                        vis.at(x0, y) = 128;
                    }
                    if (x1 - 1 >= 0 && x1 - 1 < vis.width) {
                        vis.at(x1 - 1, y) = 128;
                    }
                }
            }
            const std::string stem = stem_of(sample.row.file);
            const std::string overlay = stem + "_boxes.pgm";
            const std::string input = stem + "_input.pgm";
            vision::save_pgm(vision::join_path(art_dir, input), sample.image);
            vision::save_pgm(vision::join_path(art_dir, overlay), vis);
            written.push_back(input);
            written.push_back(overlay);
            report.n_outputs += 1 + static_cast<int>(ccl.components.size());
        }
        report.notes.push_back("outputs: components.tsv, *_input.pgm, *_boxes.pgm overlays");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "components.tsv"), values_tsv.str());
        written.insert(written.begin(), "components.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_ccl";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        CclAtom atom;
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
