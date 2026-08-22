#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "geometrify/hu_moments/hu_moments.hpp"

#include <iomanip>
#include <sstream>

// Atom demo: mask in → Hu / Flusser values + images out. No GT scoring.
class HuMomentsAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"hu_moments"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

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
        print_banner("run hu_moments → values + images");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tw\th\tarea\tcx\tcy\thu1\thu2\thu3\thu4\thu5\thu6\thu7"
                      "\tflusser1\tflusser2\tflusser3\tflusser4\trot_delta\n";
        values_tsv << std::scientific << std::setprecision(8);
        for (const auto& sample : samples) {
            const auto r = vision::HuMoments::compute(sample.image);
            const auto r_rot = vision::HuMoments::compute(mission::rotate_gray_90(sample.image));
            const double rot_delta = mission::l2_delta7(r.hu, r_rot.hu);
            std::cout << "  " << sample.row.file << "  area=" << r.area << "  cx=" << r.cx
                      << "  cy=" << r.cy << "\n";
            std::cout << "      hu=[" << r.hu[0] << ", " << r.hu[1] << ", " << r.hu[2] << ", "
                      << r.hu[3] << ", " << r.hu[4] << ", " << r.hu[5] << ", " << r.hu[6] << "]\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << sample.image.width
                       << '\t' << sample.image.height << '\t' << r.area << '\t' << r.cx << '\t'
                       << r.cy;
            for (double h : r.hu) {
                values_tsv << '\t' << h;
            }
            for (double f : r.flusser) {
                values_tsv << '\t' << f;
            }
            values_tsv << '\t' << rot_delta << '\n';
            const std::string stem = stem_of(sample.row.file);
            mission::write_hu_json(vision::join_path(art_dir, stem + "_hu_moment_invariants.json"),
                                   sample.row.file, r.hu, rot_delta);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), sample.image);
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_hu_moment_invariants.json");
            ++report.n_outputs;
        }
        report.notes.push_back("artifacts: hu_moment_invariants.json, hu_moments.tsv");
        report.notes.push_back("rot_delta measures scale/translation/rotation invariance (target <= 1e-4)");
    }

    void write(const std::string& dir) {
        const std::string tsv = "hu_moments.tsv";
        vision::write_text_file(vision::join_path(dir, tsv), values_tsv.str());
        written.insert(written.begin(), tsv);
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_hu_moments";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        HuMomentsAtom atom;
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
