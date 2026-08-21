#include "test_harness.hpp"
#include "featurizations/laplace_gaussian/laplace_gaussian.hpp"

#include <sstream>

// Atom demo: gray in → LoG response + zero-crossing map out.
class LaplaceGaussianAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"laplace_gaussian"};
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
        print_banner("run LoG → response + zero crossings");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tzero_crossings\tmin_resp\tmax_resp\n";
        contour::LaplaceGaussian log;
        for (const auto& sample : samples) {
            const auto im = to_contour(sample.image);
            const contour::Field resp = log.response(im);
            const auto zc = log.zero_crossings(im);
            int nz = 0;
            for (uint8_t p : zc.data) {
                nz += p > 0 ? 1 : 0;
            }
            float lo = 0, hi = 0;
            if (!resp.data.empty()) {
                lo = hi = resp.data.front();
                for (float v : resp.data) {
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                }
            }
            std::cout << "  " << sample.row.file << "  zc=" << nz << "  resp=[" << lo << "," << hi
                      << "]\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << nz << '\t' << lo
                       << '\t' << hi << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string r_name = stem + "_response.pgm";
            const std::string z_name = stem + "_zerocross.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, r_name), field_to_gray(resp));
            vision::save_pgm(vision::join_path(art_dir, z_name), to_gray(zc));
            written.push_back(in_name);
            written.push_back(r_name);
            written.push_back(z_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: laplace_gaussian.tsv, *_input.pgm, *_response.pgm, *_zerocross.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "laplace_gaussian.tsv"), values_tsv.str());
        written.insert(written.begin(), "laplace_gaussian.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_laplace_gaussian";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        LaplaceGaussianAtom atom;
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
