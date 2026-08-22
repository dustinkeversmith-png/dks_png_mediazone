#include "test_harness.hpp"
#include "segmentation/bbox_auto/bbox_auto.hpp"

#include <sstream>

// Atom demo: mask in → tight bbox, padded crop, uncrop canvas out.
class BBoxAutoAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"bbox_auto"};
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
        print_banner("run BBoxAuto → crop / uncrop PGMs + box TSV");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tx\ty\tw\th\tcrop_w\tcrop_h\torigin_x\torigin_y\n";
        for (const auto& sample : samples) {
            const auto mask = to_contour(sample.image);
            const contour::Rect box = contour::BBoxAuto::from_mask(mask);
            const auto crop = contour::BBoxAuto::crop(mask, box, 0.12f);
            const auto restored = contour::BBoxAuto::uncrop(crop.image, crop);
            std::cout << "  " << sample.row.file << "  box=" << box.w << "x" << box.h
                      << "  crop=" << crop.image.width << "x" << crop.image.height << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << box.x << '\t' << box.y
                       << '\t' << box.w << '\t' << box.h << '\t' << crop.image.width << '\t'
                       << crop.image.height << '\t' << crop.origin_x << '\t' << crop.origin_y << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string box_name = stem + "_bbox.pgm";
            const std::string crop_name = stem + "_crop.pgm";
            const std::string uncrop_name = stem + "_uncrop.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, box_name),
                             overlay_rect(sample.image, box.x, box.y, box.w, box.h));
            vision::save_pgm(vision::join_path(art_dir, crop_name), to_gray(crop.image));
            vision::save_pgm(vision::join_path(art_dir, uncrop_name), to_gray(restored));
            written.push_back(in_name);
            written.push_back(box_name);
            written.push_back(crop_name);
            written.push_back(uncrop_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: bbox.tsv, *_input.pgm, *_bbox.pgm, *_crop.pgm, *_uncrop.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "bbox.tsv"), values_tsv.str());
        written.insert(written.begin(), "bbox.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_bbox_auto";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        BBoxAutoAtom atom;
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
