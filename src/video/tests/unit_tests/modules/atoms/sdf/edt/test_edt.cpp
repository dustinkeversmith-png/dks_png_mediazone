#include "test_harness.hpp"
#include "sdf/edt/edt.hpp"

#include <algorithm>
#include <sstream>

// Atom demo: mask in → unsigned FG/BG distance + signed EDT out.
class EdtAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"edt"};
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
        print_banner("run EDT → fg / bg / signed fields");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tmax_fg\tmax_bg\tmin_sdf\tmax_sdf\n";
        for (const auto& sample : samples) {
            const auto mask = to_contour(sample.image);
            const auto fg = contour::EuclideanDistanceTransform::to_foreground(mask);
            const auto bg = contour::EuclideanDistanceTransform::to_background(mask);
            const auto sdf = contour::EuclideanDistanceTransform::signed_from_mask(mask);
            float mfg = 0, mbg = 0, lo = 0, hi = 0;
            if (!fg.data.empty()) {
                mfg = *std::max_element(fg.data.begin(), fg.data.end());
                mbg = *std::max_element(bg.data.begin(), bg.data.end());
                lo = *std::min_element(sdf.data.begin(), sdf.data.end());
                hi = *std::max_element(sdf.data.begin(), sdf.data.end());
            }
            std::cout << "  " << sample.row.file << "  max_fg=" << mfg << "  max_bg=" << mbg << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << mfg << '\t' << mbg << '\t'
                       << lo << '\t' << hi << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string fg_name = stem + "_dist_fg.pgm";
            const std::string bg_name = stem + "_dist_bg.pgm";
            const std::string sdf_name = stem + "_signed.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, fg_name), field_to_gray(fg));
            vision::save_pgm(vision::join_path(art_dir, bg_name), field_to_gray(bg));
            vision::save_pgm(vision::join_path(art_dir, sdf_name), field_to_gray(sdf));
            written.push_back(in_name);
            written.push_back(fg_name);
            written.push_back(bg_name);
            written.push_back(sdf_name);
            ++report.n_outputs;
        }
        report.notes.push_back("stages: *_input.pgm, *_dist_fg.pgm, *_dist_bg.pgm, *_signed.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "edt.tsv"), values_tsv.str());
        written.insert(written.begin(), "edt.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_edt";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        EdtAtom atom;
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
