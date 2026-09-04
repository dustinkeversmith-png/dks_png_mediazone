#include "test_harness.hpp"
#include "filters/edge/canny/canny.hpp"

#include <sstream>

// Atom demo: gray in → Canny edge map out.
class CannyAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"canny"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        std::cout << "loaded " << samples.size() << " images\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run Canny → edge PGMs");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tedge_pixels\tw\th\n";
        contour::Canny canny;
        for (const auto& sample : samples) {
            // Recipe: luma → Gaussian (5×5, σ≈1.4) → Canny thin edges.
            const auto blurred = gaussian_blur_gray(sample.image, 1.4f);
            const auto edges = canny.detect(to_contour(blurred));
            int n = 0;
            for (uint8_t p : edges.data) {
                n += p > 0 ? 1 : 0;
            }
            std::cout << "  " << sample.row.file << "  edges=" << n << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << n << '\t'
                       << sample.image.width << '\t' << sample.image.height << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_processed_base.pgm"), blurred);
            vision::save_pgm(vision::join_path(art_dir, stem + "_canny_edges.pgm"), to_gray(edges));
            written.push_back(stem + "_processed_base.pgm");
            written.push_back(stem + "_canny_edges.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("recipe: luma→Gaussian→Canny; *_processed_base.pgm, *_canny_edges.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "canny.tsv"), values_tsv.str());
        written.insert(written.begin(), "canny.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_canny";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        CannyAtom atom;
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
