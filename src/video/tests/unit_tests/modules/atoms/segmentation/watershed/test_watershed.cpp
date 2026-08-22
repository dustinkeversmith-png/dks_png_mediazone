#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "atom_config.hpp"
#include "segmentation/watershed/watershed.hpp"
#include "datasets/evaluation/evaluation.hpp"
#include "filters/sobel/sobel.hpp"
#include "sdf/edt/edt.hpp"

#include <sstream>

// Atom: raw RGB/mask -> grayscale -> gradient/EDT -> Meyer watershed -> scored artifacts.
class WatershedAtom {
public:
    vision::AtomConfig config;
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"watershed"};
    std::ostringstream values_tsv;
    std::ostringstream genus_tsv;
    std::vector<std::string> written;

    explicit WatershedAtom(int argc, char** argv) {
        config = vision::load_atom_config_near(argc > 0 ? argv[0] : nullptr);
    }

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, 128);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    math::ImageBuffer preprocess_mask(const LoadedSample& sample, size_t idx) const {
        math::ImageBuffer mask;
        if (idx < provider_samples.size() && !provider_samples[idx].ground_truth.empty()) {
            mask = provider_samples[idx].ground_truth;
        } else {
            mask = to_contour(sample.image);
        }
        if (config.input_type == "RAW_RGB" && idx < provider_samples.size() &&
            !provider_samples[idx].sample.rgb.empty()) {
            contour::SobelFilter sobel;
            sobel.compute(provider_samples[idx].sample.rgb.channels > 1
                              ? provider_samples[idx].sample.rgb
                              : to_contour(sample.image));
            mask = to_contour(field_to_gray(sobel.mag));
            for (uint8_t& p : mask.data) {
                p = p > 32 ? 255 : 0;
            }
        }
        return mask;
    }

    void run(const std::string& art_dir) {
        print_banner("run watershed pipeline");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_basins\tn_watershed\tboundary_f1\tiou\tchi\n";
        genus_tsv << "file\tlabel\tchi\tboundary_f1\tiou\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const auto mask = preprocess_mask(sample, si);
            const auto ws = contour::Watershed::from_mask(mask);
            double boundary_f1 = 0.0;
            double iou = 0.0;
            int chi = 0;
            if (si < provider_samples.size() && !provider_samples[si].ground_truth.empty()) {
                const auto e = vision::EulerCharacteristic::compute(provider_samples[si].ground_truth);
                chi = e.chi;
                math::ImageBuffer pred = math::make_gray(ws.width, ws.height, 0);
                for (size_t i = 0; i < pred.data.size() && i < ws.labels.size(); ++i) {
                    pred.data[i] = ws.labels[i] > 0 ? 255 : 0;
                }
                const auto score =
                    datasets::evaluate_mask(pred, provider_samples[si].ground_truth, report.elapsed_ms);
                boundary_f1 = score.boundary_f1;
                iou = score.iou;
            }
            std::cout << "  " << sample.row.file << "  basins=" << ws.n_basins << "  lines=" << ws.n_watershed
                      << "  f1=" << boundary_f1 << "  iou=" << iou << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << ws.n_basins << '\t'
                       << ws.n_watershed << '\t' << boundary_f1 << '\t' << iou << '\t' << chi << '\n';
            genus_tsv << sample.row.file << '\t' << sample.row.label << '\t' << chi << '\t' << boundary_f1
                      << '\t' << iou << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string mk_name = stem + "_marker_seeds.pgm";
            const std::string rf_name = stem + "_edt_relief.pgm";
            const std::string lb_name = stem + "_segmented_labels.pgm";
            vision::save_pgm(vision::join_path(art_dir, mk_name),
                             colorize_labels(ws.markers, ws.width, ws.height));
            vision::save_pgm(vision::join_path(art_dir, rf_name), field_to_gray(ws.relief));
            vision::save_pgm(vision::join_path(art_dir, lb_name),
                             colorize_labels(ws.labels, ws.width, ws.height));
            written.push_back(mk_name);
            written.push_back(rf_name);
            written.push_back(lb_name);
            ++report.n_outputs;
        }
        report.notes.push_back("artifacts: marker_seeds, edt_relief, segmented_labels, watershed.tsv, genus_summary.tsv");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "watershed.tsv"), values_tsv.str());
        vision::write_text_file(vision::join_path(dir, "genus_summary.tsv"), genus_tsv.str());
        written.insert(written.begin(), "genus_summary.tsv");
        written.insert(written.begin(), "watershed.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "ade20k", [&](const AtomCli& cli) -> int {
        WatershedAtom atom(argc, argv);
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for watershed atom\n";
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
