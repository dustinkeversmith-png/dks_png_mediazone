#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "features/turning_function/turning_function.hpp"

#include <iomanip>
#include <sstream>

// Atom demo: mask in → turning-function samples out.
class TurningAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"turning_function"};
    std::ostringstream plot_csv;
    std::vector<std::vector<float>> features;
    std::vector<std::string> labels;
    std::vector<std::string> written;
    vision::TurningFunction engine;

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
        print_banner("run turning function → Θ(s) vectors");
        ScopedTimer timer(&report.elapsed_ms);
        plot_csv << "file,label,s,theta\n";
        for (const auto& sample : samples) {
            auto feat = engine.compute(sample.image);
            labels.push_back(sample.row.label);
            features.push_back(feat);
            for (size_t k = 0; k < feat.size(); ++k) {
                const float s = static_cast<float>(k) / static_cast<float>(std::max<size_t>(1, feat.size()));
                plot_csv << sample.row.file << ',' << sample.row.label << ',' << s << ',' << feat[k] << '\n';
            }
            std::cout << "  " << sample.row.file << "  dim=" << feat.size();
            if (!feat.empty()) {
                std::cout << "  t0=" << feat[0];
            }
            std::cout << "\n";
            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), sample.image);
            written.push_back(stem + "_input.pgm");
            ++report.n_outputs;
        }
        std::vector<std::vector<float>> dists(features.size());
        for (size_t i = 0; i < features.size(); ++i) {
            dists[i].resize(features.size());
            for (size_t j = 0; j < features.size(); ++j) {
                dists[i][j] = vision::TurningFunction::circular_l2(features[i], features[j]);
            }
        }
        mission::write_l2_matrix_json(vision::join_path(art_dir, "l2_distance_matrix.json"), labels, dists);
        written.push_back("l2_distance_matrix.json");
        report.notes.push_back("artifacts: turning_function_plot.csv, l2_distance_matrix.json");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "turning_function_plot.csv"), plot_csv.str());
        written.insert(written.begin(), "turning_function_plot.csv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_turning";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        TurningAtom atom;
        if (!atom.load(cli, argc, argv)) {
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
