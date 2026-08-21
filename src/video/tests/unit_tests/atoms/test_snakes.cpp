#include "test_harness.hpp"
#include "bbox/bbox_auto.hpp"
#include "boundary_tracing/snakes.hpp"
#include "contour_kit/metrics.hpp"

#include <sstream>

// Atom demo: mask in → GVF snake polyline from bbox prior out.
class SnakesAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"snakes"};
    std::ostringstream values_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        samples = load_atom_png_dataset(root, dataset, sample_filter);
        for (auto& s : samples) {
            s.image = downscale_max_side(s.image, 96);
        }
        std::cout << "loaded " << samples.size() << " masks (max side 96)\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run snakes → active-contour overlay");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_points\tclosed\tarea\n";
        contour::SnakeActiveContour snake;
        snake.iterations = 40;
        snake.n_points = 96;
        for (const auto& sample : samples) {
            const auto mask = to_contour(sample.image);
            const contour::Rect box = contour::BBoxAuto::from_mask(mask);
            const contour::Polyline poly = snake.evolve(mask, box);
            const float area = contour::shoelace(poly.points);
            std::cout << "  " << sample.row.file << "  n=" << poly.points.size() << "  area=" << area
                      << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << poly.points.size()
                       << '\t' << (poly.closed ? 1 : 0) << '\t' << area << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string ov_name = stem + "_snake.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, ov_name),
                             overlay_polyline(sample.image, poly.points, poly.closed));
            written.push_back(in_name);
            written.push_back(ov_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: snakes.tsv, *_input.pgm, *_snake.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "snakes.tsv"), values_tsv.str());
        written.insert(written.begin(), "snakes.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_snakes";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        SnakesAtom atom;
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
