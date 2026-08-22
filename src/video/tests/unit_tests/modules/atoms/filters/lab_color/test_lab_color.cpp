#include "test_harness.hpp"
#include "color/lab_color_space.hpp"

#include <sstream>

// Atom demo: gray/RGB in → Lab stats + lightness field out.
class LabColorAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"lab_color"};
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
        print_banner("run LabColor → mean Lab + lightness PGM");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tmean_L\tmean_a\tmean_b\tfg_L\tfg_a\tfg_b\n";
        for (const auto& sample : samples) {
            const auto im = to_contour(sample.image);
            const contour::Field L = contour::LabColor::lightness(im);
            double sL = 0, sa = 0, sb = 0, fL = 0, fa = 0, fb = 0;
            int n = 0, nfg = 0;
            for (int y = 0; y < im.height; ++y) {
                for (int x = 0; x < im.width; ++x) {
                    const contour::Lab p = contour::LabColor::at(im, x, y);
                    sL += p.L;
                    sa += p.a;
                    sb += p.b;
                    ++n;
                    if (im.at(x, y) > 127) {
                        fL += p.L;
                        fa += p.a;
                        fb += p.b;
                        ++nfg;
                    }
                }
            }
            const double inv = n > 0 ? 1.0 / n : 0.0;
            const double invf = nfg > 0 ? 1.0 / nfg : 0.0;
            std::cout << "  " << sample.row.file << "  L=" << (sL * inv) << "  a=" << (sa * inv)
                      << "  b=" << (sb * inv) << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << (sL * inv) << '\t'
                       << (sa * inv) << '\t' << (sb * inv) << '\t' << (fL * invf) << '\t' << (fa * invf)
                       << '\t' << (fb * invf) << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string l_name = stem + "_lightness.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, l_name), field_to_gray(L));
            written.push_back(in_name);
            written.push_back(l_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: lab.tsv, *_input.pgm, *_lightness.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "lab.tsv"), values_tsv.str());
        written.insert(written.begin(), "lab.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_lab_color";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        LabColorAtom atom;
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
