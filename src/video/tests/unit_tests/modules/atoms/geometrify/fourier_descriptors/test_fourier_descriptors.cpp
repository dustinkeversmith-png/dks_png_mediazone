#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "geometrify/fourier_descriptors/fourier_descriptors.hpp"

#include <iomanip>
#include <sstream>

// Atom demo: silhouette in → Fourier magnitude descriptor out.
class FourierAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"fourier_descriptors"};
    std::ostringstream spectrum_csv;
    std::ostringstream values_tsv;
    std::vector<std::string> written;
    vision::FourierDescriptors engine;

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
        print_banner("run Fourier descriptors → vector TSV + masks");
        ScopedTimer timer(&report.elapsed_ms);
        spectrum_csv << "file,label,rot_delta";
        values_tsv << "file\tlabel\tdim";
        // header filled after first compute
        bool header_dims = false;
        for (const auto& sample : samples) {
            auto desc = engine.compute(sample.image);
            if (!header_dims) {
                for (size_t k = 0; k < desc.size(); ++k) {
                    values_tsv << "\tf" << k;
                    spectrum_csv << ",f" << k;
                }
                values_tsv << "\trot_delta\n";
                spectrum_csv << "\n";
                header_dims = true;
            }
            const auto rot_desc = engine.compute(mission::rotate_gray_90(sample.image));
            const double rot_delta = mission::l2_delta(desc, rot_desc);
            std::cout << "  " << sample.row.file << "  dim=" << desc.size() << "  rot_delta=" << rot_delta
                      << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << desc.size();
            spectrum_csv << sample.row.file << ',' << sample.row.label << ',' << rot_delta;
            values_tsv << std::scientific << std::setprecision(6);
            for (float v : desc) {
                values_tsv << '\t' << v;
                spectrum_csv << ',' << v;
            }
            values_tsv << '\t' << rot_delta << '\n';
            spectrum_csv << '\n';
            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            written.push_back(in_name);
            ++report.n_outputs;
        }
        report.notes.push_back("artifacts: fourier_spectrum.csv, fourier_descriptors.tsv");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "fourier_descriptors.tsv"), values_tsv.str());
        vision::write_text_file(vision::join_path(dir, "fourier_spectrum.csv"), spectrum_csv.str());
        written.insert(written.begin(), "fourier_spectrum.csv");
        written.insert(written.begin(), "fourier_descriptors.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_fourier";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        FourierAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for " << cli.dataset << " under " << cli.data_root << "\n";
            return 1;
        }
        if (cli.list_only) {
            for (const auto& s : atom.samples) {
                std::cout << "  " << s.row.file << "\t" << s.row.label << "\n";
            }
            return 0;
        }
        const std::string art = make_artifact_dir(cli.artifact_dir);
        atom.run(art);
        atom.write(art);
        return 0;
    });
}
