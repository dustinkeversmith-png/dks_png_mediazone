#include "test_harness.hpp"
#include "filters/bilateral.hpp"

#include <sstream>

// Atom demo: gray in → bilateral-smoothed image out.
class BilateralAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"bilateral"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        for (auto& s : samples) {
            s.image = downscale_max_side(s.image, 128);
        }
        std::cout << "loaded " << samples.size() << " images (max side 128)\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run bilateral filter → smoothed PGM");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tmean_abs_delta\n";
        contour::BilateralFilter bf;
        bf.radius = 2;
        for (const auto& sample : samples) {
            const auto src = to_contour(sample.image);
            const auto out = bf.apply(src);
            double acc = 0;
            const int n = src.width * src.height;
            for (int i = 0; i < n; ++i) {
                acc += std::abs(static_cast<int>(out.data[static_cast<size_t>(i)]) -
                                static_cast<int>(src.data[static_cast<size_t>(i)]));
            }
            const double mad = n > 0 ? acc / n : 0.0;
            std::cout << "  " << sample.row.file << "  mean_abs_delta=" << mad << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << mad << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string o_name = stem + "_bilateral.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, o_name), to_gray(out));
            written.push_back(in_name);
            written.push_back(o_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: bilateral.tsv, *_input.pgm, *_bilateral.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "bilateral.tsv"), values_tsv.str());
        written.insert(written.begin(), "bilateral.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_bilateral";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        BilateralAtom atom;
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
