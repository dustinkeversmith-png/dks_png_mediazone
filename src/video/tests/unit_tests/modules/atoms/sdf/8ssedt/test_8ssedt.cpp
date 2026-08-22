#include "test_harness.hpp"
#include "featurizations/sdf/8SSEDT.hpp"

#include <sstream>

// Atom demo: mask in → exact Euclidean SDF (Felzenszwalb / 8SSEDT) out.
class ExactSdfAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"exact_sdf"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        std::cout << "loaded " << samples.size() << " masks\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run ExactSDF / 8SSEDT → field PGM + stats");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tmin\tmax\tmean\tn_inside\n";
        for (const auto& sample : samples) {
            const contour::Field sdf = contour::ExactSDF::from_mask(to_contour(sample.image));
            float lo = 0, hi = 0, sum = 0;
            int inside = 0;
            if (!sdf.data.empty()) {
                lo = hi = sdf.data.front();
                for (float v : sdf.data) {
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                    sum += v;
                    inside += v <= 0.0f ? 1 : 0;
                }
            }
            const float mean = sdf.data.empty() ? 0.0f : sum / static_cast<float>(sdf.data.size());
            std::cout << "  " << sample.row.file << "  sdf=[" << lo << "," << hi << "]  inside=" << inside
                      << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << lo << '\t' << hi << '\t'
                       << mean << '\t' << inside << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string sdf_name = stem + "_sdf.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, sdf_name), field_to_gray(sdf));
            written.push_back(in_name);
            written.push_back(sdf_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: exact_sdf.tsv, *_input.pgm, *_sdf.pgm (negative inside)");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "exact_sdf.tsv"), values_tsv.str());
        written.insert(written.begin(), "exact_sdf.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_8ssedt";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        ExactSdfAtom atom;
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
