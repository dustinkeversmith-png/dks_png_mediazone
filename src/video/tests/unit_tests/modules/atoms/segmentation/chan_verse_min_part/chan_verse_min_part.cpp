#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/chan_verse_min_part/chan_verse_min_part.hpp"
#include "math/contour_metrics.hpp"

#include <sstream>

// Atom: DIS5K luma → Chan–Vese minimal partition (no seeds / no edges).
class ChanVeseMinPartAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"chan_verse_min_part"};
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
        print_banner("run Chan–Vese minimal partition → φ / mask");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_regions\tc1\tc2\tenergy\tiou\tboundary_f1\tn_contour\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            // In-test prep: use photo luma only (no GT seed) — algorithm finds partition.
            const auto luma = mission_luma_image(ps, sample.image);
            const auto gt_mask = binarize_mask(mission_mask_image(ps, sample.image));

            contour::ChanVeseMinPartition cv;
            cv.iterations = 60;
            cv.checker_period = 6;
            const auto result = cv.segment(to_contour(luma));

            double iou = 0.0;
            double boundary_f1 = 0.0;
            if (!gt_mask.empty() && gt_mask.width == result.partition.width &&
                gt_mask.height == result.partition.height) {
                const auto score =
                    datasets::evaluate_mask(result.partition, to_contour(gt_mask), report.elapsed_ms);
                iou = score.iou;
                boundary_f1 = score.boundary_f1;
            }

            std::cout << "  " << sample.row.file << "  regions=" << result.n_regions
                      << "  iou=" << iou << "  f1=" << boundary_f1 << "  E=" << result.energy << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << result.n_regions
                       << '\t' << cv.c1 << '\t' << cv.c2 << '\t' << result.energy << '\t' << iou
                       << '\t' << boundary_f1 << '\t' << result.contour.points.size() << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_gt_mask.pgm"), gt_mask);
            vision::save_pgm(vision::join_path(art_dir, stem + "_partition.pgm"),
                             to_gray(result.partition));
            vision::save_pgm(vision::join_path(art_dir, stem + "_phi.pgm"), field_to_gray(cv.phi));
            vision::save_pgm(vision::join_path(art_dir, stem + "_contour_overlay.pgm"),
                             overlay_polyline(luma, result.contour.points, result.contour.closed));
            mission::write_convergence_json(
                vision::join_path(art_dir, stem + "_energy.json"), sample.row.file, cv.iterations,
                boundary_f1, result.energy);

            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_gt_mask.pgm");
            written.push_back(stem + "_partition.pgm");
            written.push_back(stem + "_phi.pgm");
            written.push_back(stem + "_contour_overlay.pgm");
            written.push_back(stem + "_energy.json");
            ++report.n_outputs;
        }
        report.notes.push_back("DIS5K luma: checkerboard-init Chan–Vese (no seeds/edges)");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "chan_vese_min_part.tsv"), values_tsv.str());
        written.insert(written.begin(), "chan_vese_min_part.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "dis5k", [&](const AtomCli& cli) -> int {
        ChanVeseMinPartAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for chan_verse_min_part atom\n";
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
