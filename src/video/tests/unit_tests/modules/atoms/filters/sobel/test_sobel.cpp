#include "test_harness.hpp"
#include "filters/sobel/sobel.hpp"

#include <sstream>

// Atom demo: gray in → Sobel gx/gy/magnitude out.
class SobelAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"sobel"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        std::cout << "loaded " << samples.size() << " images\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run Sobel → magnitude / gx / gy");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tmax_mag\tmean_mag\n";
        for (const auto& sample : samples) {
            // Recipe: luma → Gaussian blur (σ≈1) before discrete derivatives.
            const auto blurred = gaussian_blur_gray(sample.image, 1.0f);
            contour::SobelFilter sobel;
            sobel.compute(to_contour(blurred));
            float mx = 0, sum = 0;
            for (float v : sobel.mag.data) {
                mx = std::max(mx, v);
                sum += v;
            }
            const float mean = sobel.mag.data.empty() ? 0.0f : sum / static_cast<float>(sobel.mag.data.size());
            contour::Field orient = contour::make_field(sobel.gx.width, sobel.gx.height, 0);
            for (int y = 0; y < orient.height; ++y) {
                for (int x = 0; x < orient.width; ++x) {
                    const float th = std::atan2(sobel.gy.at(x, y), sobel.gx.at(x, y));
                    orient.at(x, y) = (th + 3.14159265f) / (2.0f * 3.14159265f);  // [0,1]
                }
            }
            std::cout << "  " << sample.row.file << "  max_mag=" << mx << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << mx << '\t' << mean
                       << '\n';

            const std::string stem = stem_of(sample.row.file);
            vision::save_pgm(vision::join_path(art_dir, stem + "_processed_base.pgm"), blurred);
            vision::save_pgm(vision::join_path(art_dir, stem + "_sobel_mag.pgm"), field_to_gray(sobel.mag));
            vision::save_pgm(vision::join_path(art_dir, stem + "_sobel_orient.pgm"), field_to_gray(orient));
            vision::save_pgm(vision::join_path(art_dir, stem + "_gx.pgm"), field_to_gray(sobel.gx));
            vision::save_pgm(vision::join_path(art_dir, stem + "_gy.pgm"), field_to_gray(sobel.gy));
            written.push_back(stem + "_processed_base.pgm");
            written.push_back(stem + "_sobel_mag.pgm");
            written.push_back(stem + "_sobel_orient.pgm");
            written.push_back(stem + "_gx.pgm");
            written.push_back(stem + "_gy.pgm");
            ++report.n_outputs;
        }
        report.notes.push_back("recipe: luma→Gaussian→Sobel; *_processed_base, *_sobel_mag, *_sobel_orient");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "sobel.tsv"), values_tsv.str());
        written.insert(written.begin(), "sobel.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_sobel";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        SobelAtom atom;
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
