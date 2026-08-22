#include "test_harness.hpp"
#include "bbox/bbox_auto.hpp"
#include "boundary_tracing/graph_cut.hpp"

#include <sstream>

// Atom demo: image + bbox prior in → min-cut mask out.
class GraphCutAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"graph_cut"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        for (auto& s : samples) {
            s.image = downscale_max_side(s.image, 80);
        }
        std::cout << "loaded " << samples.size() << " images (max side 80)\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run graph-cut → FG mask");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tfg_pixels\tw\th\n";
        contour::GraphCutSegmenter gc;
        gc.ellipse_fg = false;
        gc.refine_iters = 1;
        gc.k_gmm = 3;
        for (const auto& sample : samples) {
            const auto im = to_contour(sample.image);
            const contour::Rect box = contour::BBoxAuto::from_mask(im);
            const contour::ImageBuffer mask = gc.segment(im, box);
            int fg = 0;
            for (uint8_t p : mask.data) {
                fg += p > 0 ? 1 : 0;
            }
            std::cout << "  " << sample.row.file << "  fg=" << fg << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << fg << '\t'
                       << sample.image.width << '\t' << sample.image.height << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string m_name = stem + "_mask.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, m_name), to_gray(mask));
            written.push_back(in_name);
            written.push_back(m_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: graph_cut.tsv, *_input.pgm, *_mask.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "graph_cut.tsv"), values_tsv.str());
        written.insert(written.begin(), "graph_cut.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_graph_cut";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        GraphCutAtom atom;
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
