#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "contour/laplace_gaussian/laplace_gaussian.hpp"

#include <sstream>

// Atom: DIS5K luma → LoG response + zero-crossing map.
class LaplaceGaussianAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"laplace_gaussian"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 160);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run LoG → response + zero crossings");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tzero_crossings\tmin_resp\tmax_resp\n";
        contour::LaplaceGaussian log;
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            const auto luma = mission_luma_image(ps, sample.image);
            const auto im = to_contour(luma);
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
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_response.pgm"), field_to_gray(resp));
            vision::save_pgm(vision::join_path(art_dir, stem + "_zerocross.pgm"), to_gray(zc));
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_response.pgm");
            written.push_back(stem + "_zerocross.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("DIS5K luma → LoG response + zero crossings");
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
    return run_atom_main(argc, argv, "dis5k", [&](const AtomCli& cli) -> int {
        LaplaceGaussianAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for laplace_gaussian atom\n";
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
