#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "contour/moore_neighborhood/moore_neighbor.hpp"

#include <sstream>

// Atom: DIS5K mask → Moore outer contour overlay.
class MooreNeighborAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"moore_neighbor"};
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
        print_banner("run Moore neighbor → contour overlay + stats");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_points\tclosed\tarea\tperimeter\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            const auto mask_img = binarize_mask(mission_mask_image(ps, sample.image));
            const auto luma = mission_luma_image(ps, sample.image);
            auto c = vision::MooreNeighborTracer::trace(mask_img);
            std::cout << "  " << sample.row.file << "  n=" << c.points.size()
                      << "  closed=" << (c.closed ? 1 : 0) << "  area=" << c.area << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << c.points.size() << '\t'
                       << (c.closed ? 1 : 0) << '\t' << c.area << '\t' << c.perimeter << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_mask.pgm"), mask_img);
            vision::save_pgm(vision::join_path(art_dir, stem + "_contour.pgm"),
                             overlay_polyline(luma, c.points, c.closed));
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_mask.pgm");
            written.push_back(stem + "_contour.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("DIS5K mask → Moore boundary trace");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "moore.tsv"), values_tsv.str());
        written.insert(written.begin(), "moore.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "dis5k", [&](const AtomCli& cli) -> int {
        MooreNeighborAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for moore_neighbor atom\n";
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
