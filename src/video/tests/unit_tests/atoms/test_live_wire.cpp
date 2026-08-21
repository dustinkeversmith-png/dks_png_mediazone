#include "test_harness.hpp"
#include "bbox/bbox_auto.hpp"
#include "boundary_tracing/live_wire.hpp"
#include "boundary_tracing/moore_neighbor.hpp"

#include <sstream>

// Atom demo: image in → livewire path through Moore/bbox seeds out.
class LiveWireAtom {
public:
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"live_wire"};
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

    static std::vector<contour::Vec2> make_seeds(const vision::GrayImage& image) {
        auto moore = vision::MooreNeighborTracer::trace(image);
        auto resampled = vision::MooreNeighborTracer::resample(moore.points, 12);
        std::vector<contour::Vec2> seeds;
        seeds.reserve(resampled.size());
        for (const auto& p : resampled) {
            seeds.push_back({p.x, p.y});
        }
        if (seeds.size() >= 3) {
            return seeds;
        }
        const auto box = contour::BBoxAuto::from_mask(to_contour(image));
        const float x0 = box.x + box.w * 0.15f;
        const float y0 = box.y + box.h * 0.15f;
        const float x1 = box.x + box.w * 0.85f;
        const float y1 = box.y + box.h * 0.85f;
        return {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
    }

    void run(const std::string& art_dir) {
        print_banner("run livewire → shortest-path contour");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_seeds\tn_points\tclosed\n";
        for (const auto& sample : samples) {
            const auto im = to_contour(sample.image);
            const auto seeds = make_seeds(sample.image);
            contour::Livewire lw;
            const contour::Polyline poly = lw.trace_waypoints(im, seeds, true);
            std::cout << "  " << sample.row.file << "  seeds=" << seeds.size()
                      << "  n=" << poly.points.size() << "\n";
            values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << seeds.size() << '\t'
                       << poly.points.size() << '\t' << (poly.closed ? 1 : 0) << '\n';

            const std::string stem = stem_of(sample.row.file);
            const std::string in_name = stem + "_input.pgm";
            const std::string ov_name = stem + "_livewire.pgm";
            vision::save_pgm(vision::join_path(art_dir, in_name), sample.image);
            vision::save_pgm(vision::join_path(art_dir, ov_name),
                             overlay_polyline(sample.image, poly.points, poly.closed));
            written.push_back(in_name);
            written.push_back(ov_name);
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: live_wire.tsv, *_input.pgm, *_livewire.pgm");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "live_wire.tsv"), values_tsv.str());
        written.insert(written.begin(), "live_wire.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_live_wire";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        LiveWireAtom atom;
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
