#include "test_harness.hpp"
#include "filters/sobel/sobel.hpp"

#include <sstream>

// Atom demo: gray in → Sobel gx/gy/magnitude out.
class SobelAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"sobel"};
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
        print_banner("run Sobel → magnitude / gx / gy");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tmax_mag\tmean_mag\n";
        for (const auto& sample : samples) {
            contour::SobelFilter sobel;
            sobel.compute(to_contour(sample.image));
            float mx = 0, sum = 0;
            for (float v : sobel.mag.data) {
                mx = std::max(mx, v);
                sum += v;
            }
            const float mean = sobel.mag.data.empty() ? 0.0f : sum / static_cast<float>(sobel.mag.data.size());
            std::cout << "  " << sample.row.file << "  max_mag=" << mx << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << mx << '\t' << mean
                       << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string mag_name = stem + "_mag.pgm";
            const std::string gx_name = stem + "_gx.pgm";
            const std::string gy_name = stem + "_gy.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, mag_name), field_to_gray(sobel.mag));
            vision::save_pgm(vision::join_path(art_dir, gx_name), field_to_gray(sobel.gx));
            vision::save_pgm(vision::join_path(art_dir, gy_name), field_to_gray(sobel.gy));
            written.push_back(in_name);
            written.push_back(mag_name);
            written.push_back(gx_name);
            written.push_back(gy_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: sobel.tsv, *_input.pgm, *_mag.pgm, *_gx.pgm, *_gy.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "sobel.tsv"), values_tsv.str());
        written.insert(written.begin(), "sobel.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_sobel";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        SobelAtom atom;
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
