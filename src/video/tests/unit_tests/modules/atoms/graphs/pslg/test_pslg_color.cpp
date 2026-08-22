#include "test_harness.hpp"
#include "graphs/pslg/pslg_color.hpp"
#include "structures/slic/slic.hpp"

#include <sstream>

// Atom demo: PSLG faces in → Welsh–Powell map coloring out.
class PslgColorAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"pslg_color"};
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
        print_banner("run PSLG face coloring");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_faces\tn_colors\tn_adj\n";
        contour::SlicSuperpixels slic;
        slic.k = 24;
        slic.iterations = 5;
        const uint8_t palette[6] = {40, 80, 120, 160, 200, 240};
        for (const auto& sample : samples) {
            const auto im = to_contour(sample.image);
            const auto sl = slic.segment(im);
            const auto rag = contour::RegionAdjacency::from_labels(sl.labels, sl.width, sl.height, im);
            const auto col = contour::PslgColor::color(rag);
            int nfaces = 0;
            for (const auto& f : rag.faces) {
                nfaces += f.area > 0 ? 1 : 0;
            }
            std::cout << "  " << sample.row.file << "  faces=" << nfaces << "  colors=" << col.n_colors
                      << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << nfaces << '\t'
                       << col.n_colors << '\t' << rag.adj.size() << '\n';

            vision::GrayImage painted;
            painted.width = col.width;
            painted.height = col.height;
            painted.data.resize(static_cast<size_t>(col.width * col.height));
            for (size_t i = 0; i < painted.data.size(); ++i) {
                const int c = col.colored_labels[i];
                painted.data[i] = c <= 0 ? 0 : palette[static_cast<size_t>((c - 1) % 6)];
            }
            const auto bounds = label_boundaries(rag.labels, rag.width, rag.height);

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string cl_name = stem + "_colored.pgm";
            const std::string ov_name = stem + "_colored_bounds.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, cl_name), painted);
            vision::save_pgm(vision::join_path(art_dir, ov_name), overlay_mask(painted, bounds));
            written.push_back(in_name);
            written.push_back(cl_name);
            written.push_back(ov_name);
            ++report.n_outputs;
        }
        report.notes.push_back("stages: *_input.pgm, *_colored.pgm, *_colored_bounds.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "pslg_color.tsv"), values_tsv.str());
        written.insert(written.begin(), "pslg_color.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_pslg_color";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        PslgColorAtom atom;
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
