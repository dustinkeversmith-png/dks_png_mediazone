#include "test_harness.hpp"
#include "geometrify/media_axis/medial_axis.hpp"

#include <sstream>

// Atom demo: mask in → skeleton image + radius stats out.
class MedialAxisAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"medial_axis"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_png_dataset(vision::dataset_dir(root, dataset), sample_filter);
        std::cout << "loaded " << samples.size() << " silhouettes\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run medial axis → skeleton PGMs + stats");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tskeleton_pixels\tmean_radius\tw\th\n";
        for (const auto& sample : samples) {
            auto m = vision::MedialAxis::extract(sample.image);
            std::cout << "  " << sample.row.file << "  skel=" << m.skeleton_pixels
                      << "  mean_r=" << m.mean_radius << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << m.skeleton_pixels
                       << '\t' << m.mean_radius << '\t' << sample.image.width << '\t'
                       << sample.image.height << '\n';
            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string sk_name = stem + "_skeleton.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            if (!m.skeleton.empty()) {
                vision::save_pgm(vision::join_path(art_dir, sk_name), m.skeleton);
                written.push_back(sk_name);
            }
            written.push_back(in_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: skeleton.tsv, *_input.pgm, *_skeleton.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "skeleton.tsv"), values_tsv.str());
        written.insert(written.begin(), "skeleton.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_medial_axis";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        MedialAxisAtom atom;
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
