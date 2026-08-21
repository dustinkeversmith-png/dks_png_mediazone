#include "test_harness.hpp"
#include "featurizations/rbf/thin_plate_spline.hpp"
#include "boundary_tracing/moore_neighbor.hpp"

#include <sstream>

// Atom demo: mask in → contour + TPS fit stats out.
class RbfAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"rbf"};
    std::ostringstream values_tsv;
    std::ostringstream contour_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_png_dataset(vision::dataset_dir(root, dataset), sample_filter);
        std::cout << "loaded " << samples.size() << " masks\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run RBF / TPS → fit stats + contour samples");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_contour\tn_centers\tfitted\trmse\n";
        contour_tsv << "file\ti\tx\ty\n";
        for (const auto& sample : samples) {
            auto contour = vision::MooreNeighborTracer::trace(sample.image);
            vision::ThinPlateSplineRBF rbf;
            const bool fitted = rbf.fit_identity(contour.points, 28);
            const float rmse = fitted ? rbf.reconstruction_rmse() : -1.0f;
            std::cout << "  " << sample.row.file << "  contour=" << contour.points.size()
                      << "  centers=" << rbf.centers.size() << "  rmse=" << rmse << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << contour.points.size()
                       << '\t' << rbf.centers.size() << '\t' << (fitted ? 1 : 0) << '\t' << rmse
                       << '\n';
            const size_t step = std::max<size_t>(1, contour.points.size() / 64);
            for (size_t i = 0; i < contour.points.size(); i += step) {
                contour_tsv << sample.row.file << '\t' << i << '\t' << contour.points[i].x << '\t'
                            << contour.points[i].y << '\n';
            }
            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            written.push_back(in_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: rbf_fit.tsv, contours.tsv, *_input.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "rbf_fit.tsv"), values_tsv.str());
        vision::write_text_file(vision::join_path(dir, "contours.tsv"), contour_tsv.str());
        written.insert(written.begin(), "contours.tsv");
        written.insert(written.begin(), "rbf_fit.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_rbf";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        RbfAtom atom;
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
