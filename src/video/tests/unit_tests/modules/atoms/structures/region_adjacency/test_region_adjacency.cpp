#include "test_harness.hpp"
#include "structures/region_adjacency/region_adjacency.hpp"
#include "structures/slic/slic.hpp"

#include <sstream>

// Atom demo: labels in → RAG + DCEL half-edges / face Lab profiles out.
class RegionAdjacencyAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"region_adjacency"};
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
        print_banner("run region adjacency / DCEL");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_faces\tn_adj\tn_halfedges\tn_vertices\n";
        contour::SlicSuperpixels slic;
        slic.k = 24;
        slic.iterations = 5;
        for (const auto& sample : samples) {
            const auto im = to_contour(sample.image);
            const auto sl = slic.segment(im);
            const auto rag = contour::RegionAdjacency::from_labels(sl.labels, sl.width, sl.height, im);
            int nfaces = 0;
            for (const auto& f : rag.faces) {
                nfaces += f.area > 0 ? 1 : 0;
            }
            std::cout << "  " << sample.row.file << "  faces=" << nfaces << "  adj=" << rag.adj.size()
                      << "  he=" << rag.half_edges.size() << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << nfaces << '\t'
                       << rag.adj.size() << '\t' << rag.half_edges.size() << '\t' << rag.vertices.size()
                       << '\n';

            std::ostringstream faces;
            faces << "id\tarea\tcx\tcy\tL\ta\tb\n";
            for (const auto& f : rag.faces) {
                if (f.area <= 0) {
                    continue;
                }
                faces << f.id << '\t' << f.area << '\t' << f.centroid.x << '\t' << f.centroid.y << '\t'
                      << f.mean_lab.L << '\t' << f.mean_lab.a << '\t' << f.mean_lab.b << '\n';
            }
            const std::string stem = stem_of(sample.row.file);
            const std::string face_name = stem + "_faces.tsv";
            vision::write_text_file(vision::join_path(art_dir, face_name), faces.str());
            written.push_back(face_name);

            const auto bounds = label_boundaries(rag.labels, rag.width, rag.height);
            const std::string in_name = stem + "_input.pgm";
            const std::string lb_name = stem + "_labels.pgm";
            const std::string bd_name = stem + "_bounds.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, lb_name),
                             colorize_labels(rag.labels, rag.width, rag.height));
            vision::save_pgm(vision::join_path(art_dir, bd_name), overlay_mask(sample.image, bounds));
            written.push_back(in_name);
            written.push_back(lb_name);
            written.push_back(bd_name);
            ++report.n_outputs;
        }
        report.notes.push_back(
            "stages: *_input.pgm, *_labels.pgm, *_bounds.pgm, *_faces.tsv (DCEL + RAG)");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "region_adjacency.tsv"), values_tsv.str());
        written.insert(written.begin(), "region_adjacency.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_region_adjacency";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        RegionAdjacencyAtom atom;
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
