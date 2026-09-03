#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "contour/marching_squares/marching_squares.hpp"
#include "sdf/8ssedt/8SSEDT.hpp"
#include "math/contour_metrics.hpp"

#include <sstream>

// Atom: DIS5K photo mask → Exact SDF → marching-squares iso loops.
class MarchingSquaresAtom {
public:
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"marching_squares"};
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
        print_banner("run marching squares → iso polylines + SDF");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_loops\tn_points\tclosed\tarea\tchi\tboundary_f1\n";
        contour::MarchingSquares ms;
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            const auto mask_img = prepare_contour_mask(mission_mask_image(ps, sample.image));
            const auto luma = mission_luma_image(ps, sample.image);
            constexpr int kPad = 1;
            const auto padded = pad_mask_border(mask_img, kPad);
            const auto mask = to_contour(padded);
            const contour::Field sdf = contour::ExactSDF::from_mask(mask);
            auto loops = ms.extract(sdf, 0.0f);
            contour::Polyline largest = ms.largest_closed(sdf, 0.0f);
            unpad_polyline(largest, kPad);
            for (auto& loop : loops) {
                unpad_polyline(loop, kPad);
            }
            const float area = std::fabs(contour::shoelace(largest.points));
            const auto e = vision::EulerCharacteristic::compute(mask_img);
            double boundary_f1 = 0.0;
            {
                auto gt_b = contour::boundary_pixels(to_contour(mask_img));
                auto pred_b = contour::make_gray(mask_img.width, mask_img.height, 0);
                for (const auto& loop : loops) {
                    const auto b = contour::polyline_to_boundary(
                        loop.points, mask_img.width, mask_img.height, loop.closed);
                    for (size_t i = 0; i < pred_b.data.size(); ++i) {
                        if (b.data[i]) {
                            pred_b.data[i] = 255;
                        }
                    }
                }
                boundary_f1 = contour::boundary_f1(pred_b, gt_b, 2);
            }
            std::cout << "  " << sample.row.file << "  loops=" << loops.size()
                      << "  n=" << largest.points.size() << "  f1=" << boundary_f1 << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << loops.size() << '\t'
                       << largest.points.size() << '\t' << (largest.closed ? 1 : 0) << '\t' << area
                       << '\t' << e.chi << '\t' << boundary_f1 << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_iso_field.pgm"), field_to_gray(sdf));
            mission::write_polyline_svg(vision::join_path(art_dir, stem + "_contour_edges.svg"),
                                        luma.width, luma.height, largest.points, largest.closed);
            vision::save_pgm(vision::join_path(art_dir, stem + "_input.pgm"), luma);
            vision::save_pgm(vision::join_path(art_dir, stem + "_mask.pgm"), mask_img);
            vision::save_pgm(vision::join_path(art_dir, stem + "_iso.pgm"),
                             overlay_polylines(luma, loops));
            written.push_back(stem + "_input.pgm");
            written.push_back(stem + "_mask.pgm");
            written.push_back(stem + "_iso_field.pgm");
            written.push_back(stem + "_contour_edges.svg");
            written.push_back(stem + "_iso.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("DIS5K photo+mask → ExactSDF → marching squares");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "marching_squares.tsv"), values_tsv.str());
        written.insert(written.begin(), "marching_squares.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "dis5k", [&](const AtomCli& cli) -> int {
        MarchingSquaresAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for marching_squares atom\n";
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
