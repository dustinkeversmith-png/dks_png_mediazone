#include "test_harness.hpp"
#include "filters/gvf/gvh.hpp"
#include "filters/edge/canny/canny.hpp"
#include "filters/bilateral/bilateral.hpp"

#include <sstream>

// Atom demo: edges in → Gradient Vector Flow field out.
class GvfAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"gvf"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        for (auto& s : samples) {
            s.image = downscale_max_side(s.image, 96);
        }
        std::cout << "loaded " << samples.size() << " images (max side 96)\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run GVF → flow magnitude");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tmax_flow\tmean_flow\n";
        contour::Canny canny;
        contour::BilateralFilter bf;
        bf.radius = 2;
        bf.sigma_s = 2.0f;
        bf.sigma_r = 25.0f;
        for (const auto& sample : samples) {
            // Recipe: bilateral → Canny edge force → GVF diffusion.
            const auto smoothed = bf.apply(to_contour(sample.image));
            const auto edges = canny.detect(smoothed);
            contour::GradientVectorFlow gvf;
            gvf.iterations = 24;
            gvf.compute(edges, true);
            contour::Field mag = contour::make_field(gvf.u.width, gvf.u.height);
            float mx = 0, sum = 0;
            for (int y = 0; y < mag.height; ++y) {
                for (int x = 0; x < mag.width; ++x) {
                    const float v = std::hypot(gvf.u.at(x, y), gvf.v.at(x, y));
                    mag.at(x, y) = v;
                    mx = std::max(mx, v);
                    sum += v;
                }
            }
            const float mean = mag.data.empty() ? 0.0f : sum / static_cast<float>(mag.data.size());
            std::cout << "  " << sample.row.file << "  max_flow=" << mx << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << mx << '\t' << mean
                       << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_processed_base.pgm"), to_gray(smoothed));
            vision::save_pgm(vision::join_path(art_dir, stem + "_edge_force.pgm"), to_gray(edges));
            vision::save_pgm(vision::join_path(art_dir, stem + "_gvf_mag.pgm"), field_to_gray(mag));
            written.push_back(stem + "_processed_base.pgm");
            written.push_back(stem + "_edge_force.pgm");
            written.push_back(stem + "_gvf_mag.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("recipe: bilateral→Canny→GVF; *_processed_base, *_gvf_mag");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "gvf.tsv"), values_tsv.str());
        written.insert(written.begin(), "gvf.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_gvf";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        GvfAtom atom;
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
