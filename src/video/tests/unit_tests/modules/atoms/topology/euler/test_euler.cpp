#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "topology/euler/euler_characteristic.hpp"

#include <sstream>

// Atom demo: mask in → components / holes / chi out.
class EulerAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"euler"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 64);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run Euler characteristic → topology TSV + masks");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tcomponents\tholes\tchi\texpected_chi\tgenus_ok\tw\th\n";
        for (const auto& sample : samples) {
            auto e = vision::EulerCharacteristic::compute(sample.image);
            const auto expected = mission::expected_glyph_chi(sample.row.label);
            const bool genus_ok = expected.has_value() && e.chi == *expected;
            std::cout << "  " << sample.row.file << "  C=" << e.components << "  H=" << e.holes
                      << "  chi=" << e.chi;
            if (expected.has_value()) {
                std::cout << "  expected=" << *expected << "  ok=" << genus_ok;
            }
            std::cout << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << e.components << '\t'
                       << e.holes << '\t' << e.chi << '\t';
            if (expected.has_value()) {
                values_tsv << *expected << '\t' << (genus_ok ? 1 : 0);
            } else {
                values_tsv << "-\t-";
            }
            values_tsv << '\t' << sample.image.width << '\t' << sample.image.height << '\n';
            const std::string stem = stem_of(sample.row.file);
            mission::write_topological_tree_json(
                vision::join_path(art_dir, stem + "_topological_tree.json"), sample.row.file, e.components,
                e.holes, e.chi, expected);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), sample.image);
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_topological_tree.json");
            ++report.n_outputs;
        }
        report.notes.push_back("artifacts: genus_summary.tsv, topological_tree.json");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "genus_summary.tsv"), values_tsv.str());
        written.insert(written.begin(), "genus_summary.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_euler";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        EulerAtom atom;
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
