#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "segmentation/helpers/chan_vese/chan_vese.hpp"
#include "math/contour_metrics.hpp"

#include <cstring>
#include <sstream>

// Atom: DIS5K photo → Chan–Vese minimal partition (no seeds / no edges).
// The multiphase level sets carve the frame into appearance classes and every
// connected piece of a class becomes its own object instance, so the atom is
// scored on how well it isolates each object, not just on a foreground cut.

// Sweep the module's own parameters without a rebuild. `params` *is* the
// module's default-constructed configuration, so the atom can never drift away
// from the defaults it is supposed to be demonstrating.
struct CvTuning {
    contour::ChanVeseMinPartition params;
    int max_side = 192;
    bool color = true;

    // Strips the flags it understands so the shared atom CLI still validates the rest.
    std::vector<char*> parse(int argc, char** argv) {
        std::vector<char*> rest;
        for (int i = 0; i < argc; ++i) {
            const std::string a = argv[i];
            auto value = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
            if (a == "--size") {
                max_side = std::stoi(value());
            } else if (a == "--iters") {
                params.iterations = std::stoi(value());
            } else if (a == "--scales") {
                params.scales = std::stoi(value());
            } else if (a == "--levelsets") {
                params.n_levelsets = std::stoi(value());
            } else if (a == "--mu") {
                params.mu = std::stof(value());
            } else if (a == "--dt") {
                params.dt = std::stof(value());
            } else if (a == "--merge-tol") {
                params.merge_tol = std::stof(value());
            } else if (a == "--rag-tol") {
                params.rag_tol = std::stof(value());
            } else if (a == "--bnd-con") {
                params.bnd_con_bg = std::stof(value());
            } else if (a == "--min-area-frac") {
                params.min_area_frac = std::stof(value());
            } else if (a == "--edge-beta") {
                params.edge_beta = std::stof(value());
            } else if (a == "--border-band") {
                params.border_band_frac = std::stof(value());
            } else if (a == "--bg-color-tol") {
                params.bg_color_tol = std::stof(value());
            } else if (a == "--denoise") {
                params.denoise_passes = std::stoi(value());
            } else if (a == "--gray") {
                color = false;
            } else {
                rest.push_back(argv[i]);
            }
        }
        return rest;
    }
};

// Atom demos have no pass/fail, but the algorithm has invariants that a silent
// regression (a level set that never moves, say) would otherwise hide.
struct Invariants {
    int checked = 0;
    std::vector<std::string> failures;

    void expect(bool ok, const std::string& what) {
        ++checked;
        if (!ok) {
            failures.push_back(what);
        }
    }
    void print() const {
        std::cout << "\ninvariants: " << (checked - static_cast<int>(failures.size())) << " / "
                  << checked << " passed\n";
        for (const auto& f : failures) {
            std::cout << "  FAIL  " << f << "\n";
        }
    }
};

class ChanVeseMinPartAtom {
public:
    CvTuning tune;
    std::vector<ProviderLoadedSample> provider_samples;
    std::vector<LoadedSample> samples;
    AtomDemoReport report{"chan_vese"};
    Invariants checks;
    std::ostringstream values_tsv;
    std::ostringstream objects_tsv;
    std::vector<std::string> written;
    double sum_fg_iou = 0.0;
    double sum_best_iou = 0.0;
    double sum_f1 = 0.0;
    double sum_oracle = 0.0;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission =
            load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 8, tune.max_side);
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run Chan–Vese multiphase partition → objects");
        ScopedTimer timer(&report.elapsed_ms);
        values_tsv << "file\tlabel\tn_phases\tn_objects\titers\tc1\tc2\tenergy\tfg_iou\tboundary_f1"
                      "\tbest_object_iou\toracle_iou\tobjects_on_gt\tms\n";
        objects_tsv << "file\tid\tphase\tis_bg\tarea\tx\ty\tw\th\tmean\tbnd_con\tiou_vs_gt\n";
        for (size_t si = 0; si < samples.size(); ++si) {
            const auto& sample = samples[si];
            const ProviderLoadedSample* ps =
                si < provider_samples.size() ? &provider_samples[si] : nullptr;
            // In-test prep: photo pixels only (no GT seed) — the algorithm finds the partition.
            const auto luma = mission_luma_image(ps, sample.image);
            const auto gt_mask = binarize_mask(mission_mask_image(ps, sample.image));
            const auto input = tune.color ? mission_color_image(ps, luma) : luma;

            contour::ChanVeseMinPartition cv = tune.params;

            double ms = 0.0;
            contour::ChanVeseMinPartition::Result result;
            {
                ScopedTimer t(&ms);
                result = cv.segment(input);
            }
            score_and_write(art_dir, sample, luma, gt_mask, cv, result, ms);
        }
        const double n = std::max<size_t>(1, samples.size());
        std::ostringstream note;
        note << "DIS5K " << tune.max_side << "px, " << (1 << tune.params.n_levelsets)
             << "-phase vector Chan-Vese: mean fg IoU " << std::fixed << std::setprecision(3)
             << (sum_fg_iou / n) << ", mean best-object IoU " << (sum_best_iou / n)
             << ", mean oracle IoU " << (sum_oracle / n) << ", mean boundary F1 " << (sum_f1 / n);
        report.notes.push_back(note.str());
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "chan_vese_min_part.tsv"), values_tsv.str());
        vision::write_text_file(vision::join_path(dir, "objects.tsv"), objects_tsv.str());
        written.insert(written.begin(), "objects.tsv");
        written.insert(written.begin(), "chan_vese_min_part.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        checks.print();
        std::cout << "artifacts -> " << dir << "\n";
    }

private:
    // Colour separates objects that share a luma level; fall back to luma when
    // the provider has no RGB. Downscaling to the luma extent keeps every buffer
    // (and therefore every metric) on the same grid as the ground truth.
    static vision::GrayImage mission_color_image(const ProviderLoadedSample* ps,
                                                 const vision::GrayImage& luma) {
        if (ps == nullptr || ps->sample.rgb.empty()) {
            return luma;
        }
        const auto rgb = downscale_max_side(ps->sample.rgb, std::max(luma.width, luma.height));
        if (rgb.width != luma.width || rgb.height != luma.height) {
            return luma;
        }
        return rgb;
    }

    // Upper bound on what any foreground rule could score with these regions:
    // greedily union whichever region raises IoU the most. A high oracle with a
    // low fg_iou means the partition is fine and the background rule is not; a
    // low oracle means the level sets never isolated the object at all.
    static double oracle_iou(const std::vector<int>& labels, int n_regions,
                             const vision::GrayImage& gt) {
        std::vector<int> inter(static_cast<size_t>(n_regions + 1), 0);
        std::vector<int> area(static_cast<size_t>(n_regions + 1), 0);
        int gt_area = 0;
        for (size_t k = 0; k < labels.size() && k < gt.data.size(); ++k) {
            const int id = labels[k];
            const bool on = gt.data[k] > 127;
            gt_area += on ? 1 : 0;
            if (id > 0) {
                ++area[static_cast<size_t>(id)];
                inter[static_cast<size_t>(id)] += on ? 1 : 0;
            }
        }
        if (gt_area == 0) {
            return 0.0;
        }
        std::vector<bool> taken(static_cast<size_t>(n_regions + 1), false);
        int sel_inter = 0;
        int sel_area = 0;
        double best = 0.0;
        for (int step = 0; step < n_regions; ++step) {
            int pick = 0;
            double pick_iou = best;
            for (int id = 1; id <= n_regions; ++id) {
                if (taken[static_cast<size_t>(id)]) {
                    continue;
                }
                const int i2 = sel_inter + inter[static_cast<size_t>(id)];
                const int u2 = sel_area + area[static_cast<size_t>(id)] + gt_area - i2;
                const double iou = u2 > 0 ? static_cast<double>(i2) / u2 : 0.0;
                if (iou > pick_iou) {
                    pick_iou = iou;
                    pick = id;
                }
            }
            if (pick == 0) {
                break;
            }
            taken[static_cast<size_t>(pick)] = true;
            sel_inter += inter[static_cast<size_t>(pick)];
            sel_area += area[static_cast<size_t>(pick)];
            best = pick_iou;
        }
        return best;
    }

    static vision::GrayImage mask_of_label(const std::vector<int>& labels, int w, int h, int id) {
        vision::GrayImage m = vision::make_gray(w, h, 0);
        for (size_t k = 0; k < m.data.size() && k < labels.size(); ++k) {
            m.data[k] = labels[k] == id ? 255 : 0;
        }
        return m;
    }

    void score_and_write(const std::string& art_dir, const LoadedSample& sample,
                         const vision::GrayImage& luma, const vision::GrayImage& gt_mask,
                         const contour::ChanVeseMinPartition& cv,
                         const contour::ChanVeseMinPartition::Result& result, double ms) {
        const int w = result.partition.width;
        const int h = result.partition.height;
        const bool have_gt = !gt_mask.empty() && gt_mask.width == w && gt_mask.height == h;

        double fg_iou = 0.0;
        double boundary_f1 = 0.0;
        if (have_gt) {
            const auto score = datasets::evaluate_mask(result.partition, gt_mask, ms);
            fg_iou = score.iou;
            boundary_f1 = score.boundary_f1;
        }

        const double oracle = have_gt ? oracle_iou(result.labels, result.n_regions, gt_mask) : 0.0;
        double best_object_iou = 0.0;
        int objects_on_gt = 0;
        const std::string stem = stem_of(sample.row.file);
        for (const auto& o : result.objects) {
            double iou = 0.0;
            if (have_gt && !o.background) {
                const auto m = mask_of_label(result.labels, w, h, o.label);
                iou = image::mask_iou(m, gt_mask);
                best_object_iou = std::max(best_object_iou, iou);
                int inside = 0;
                for (size_t k = 0; k < m.data.size(); ++k) {
                    if (m.data[k] && gt_mask.data[k]) {
                        ++inside;
                    }
                }
                if (o.area > 0 && static_cast<double>(inside) / o.area > 0.5) {
                    ++objects_on_gt;
                }
            }
            objects_tsv << stem << '\t' << o.label << '\t' << o.phase << '\t' << (o.background ? 1 : 0)
                        << '\t' << o.area << '\t' << o.bbox.x << '\t' << o.bbox.y << '\t' << o.bbox.w
                        << '\t' << o.bbox.h << '\t' << o.mean[0] << '\t' << o.bnd_con << '\t' << iou
                        << '\n';
        }

        sum_fg_iou += fg_iou;
        sum_best_iou += best_object_iou;
        sum_f1 += boundary_f1;
        sum_oracle += oracle;

        std::cout << "  " << sample.row.file << "  phases=" << result.n_phases
                  << " objects=" << result.n_objects << " iters=" << cv.iterations_run
                  << "  fg_iou=" << std::fixed << std::setprecision(3) << fg_iou
                  << "  best_obj=" << best_object_iou << "  oracle=" << oracle
                  << "  f1=" << boundary_f1 << "  E=" << result.energy << "  "
                  << std::setprecision(1) << ms << " ms\n"
                  << std::setprecision(6);

        values_tsv << sample.row.file << '\t' << sample.row.label << '\t' << result.n_phases << '\t'
                   << result.n_objects << '\t' << cv.iterations_run << '\t' << cv.c1 << '\t' << cv.c2
                   << '\t' << result.energy << '\t' << fg_iou << '\t' << boundary_f1 << '\t'
                   << best_object_iou << '\t' << oracle << '\t' << objects_on_gt << '\t' << ms << '\n';

        check_invariants(stem, result);
        save_artifacts(art_dir, stem, sample, luma, gt_mask, cv, result, boundary_f1);
    }

    void check_invariants(const std::string& stem, const contour::ChanVeseMinPartition::Result& r) {
        const size_t n_px = r.labels.size();
        size_t unlabelled = 0;
        int max_label = 0;
        for (int id : r.labels) {
            if (id <= 0) {
                ++unlabelled;
            }
            max_label = std::max(max_label, id);
        }
        size_t fg = 0;
        for (uint8_t p : r.partition.data) {
            fg += p ? 1 : 0;
        }
        checks.expect(unlabelled == 0, stem + ": every pixel belongs to an instance");
        checks.expect(max_label == r.n_regions, stem + ": instance ids are compact 1..n_regions");
        // A level set that fails to move leaves one phase covering the frame —
        // the exact failure mode this atom regressed into before.
        checks.expect(r.n_phases >= 2, stem + ": partition splits into >= 2 phases");
        checks.expect(r.n_objects >= 1, stem + ": at least one foreground object");
        checks.expect(fg > 0 && fg < n_px, stem + ": foreground is a strict subset of the frame");
        checks.expect(std::isfinite(r.energy), stem + ": energy is finite");
    }

    void save_artifacts(const std::string& art_dir, const std::string& stem,
                        const LoadedSample& sample, const vision::GrayImage& luma,
                        const vision::GrayImage& gt_mask,
                        const contour::ChanVeseMinPartition& cv,
                        const contour::ChanVeseMinPartition::Result& result, double boundary_f1) {
        auto emit = [&](const std::string& suffix, const vision::GrayImage& img) {
            vision::save_pgm(vision::join_path(art_dir, stem + suffix), img);
            written.push_back(stem + suffix);
        };
        // The object preview must not draw boundaries between background
        // phases: those texture partitions look like hallucinated detections.
        std::vector<int> foreground_labels = result.labels;
        for (int& id : foreground_labels) {
            if (id <= 0 || id > static_cast<int>(result.objects.size()) ||
                result.objects[static_cast<size_t>(id - 1)].background) {
                id = 0;
            }
        }
        vision::GrayImage boxes =
            overlay_mask(luma, label_boundaries(foreground_labels, result.partition.width,
                                                result.partition.height));
        for (const auto& o : result.objects) {
            if (!o.background) {
                boxes = overlay_rect(boxes, o.bbox.x, o.bbox.y, o.bbox.w, o.bbox.h, 200);
            }
        }
        emit("_processed_base.pgm", luma);
        emit("_gt_mask.pgm", gt_mask);
        emit("_partition.pgm", result.partition);
        emit("_labels.pgm", result.label_image);
        emit("_phases.pgm", result.phase_image);
        emit("_objects.pgm", boxes);
        emit("_chan_vese_phi.pgm", field_to_gray(cv.phi));
        emit("_contour_overlay.pgm", overlay_polylines(luma, result.contours, 255, true));

        mission::write_convergence_json(vision::join_path(art_dir, stem + "_energy.json"),
                                        sample.row.file, cv.iterations_run, boundary_f1,
                                        result.energy);
        std::vector<vision::ConnectedComponentLabeler::Component> comps;
        for (const auto& o : result.objects) {
            if (!o.background) {
                comps.push_back({o.label, o.area, o.bbox});
            }
        }
        mission::write_bbox_json(vision::join_path(art_dir, stem + "_objects.json"), sample.row.file,
                                 comps);
        written.push_back(stem + "_energy.json");
        written.push_back(stem + "_objects.json");
        ++report.n_outputs;
    }
};

void print_tuning_help() {
    std::cout << "\nchan_vese tuning flags (swept without a rebuild):\n"
              << "  --size <px>          Longest side of the working image (default 192)\n"
              << "  --iters <n>          Iteration budget at the coarsest scale\n"
              << "  --scales <n>         Coarse-to-fine pyramid levels (1 = single scale)\n"
              << "  --levelsets <n>      Level set functions; phases = 2^n\n"
              << "  --mu <f>             Length penalty\n"
              << "  --dt <f>             Max |delta phi| per iteration\n"
              << "  --edge-beta <f>      Edge-indicator strength (0 = plain curvature flow)\n"
              << "  --merge-tol <f>      Merge phases with means this close\n"
              << "  --rag-tol <f>        Merge touching instances with means this close\n"
              << "  --min-area-frac <f>  Absorb instances below this share of the frame\n"
              << "  --bnd-con <f>        Boundary connectivity above which a region is backdrop\n"
              << "  --border-band <f>    Frame margin band, as a share of the short side\n"
              << "  --bg-color-tol <f>   Demote border regions this close to the backdrop mean\n"
              << "  --denoise <n>        Bilateral prefilter passes (erases thin filaments)\n"
              << "  --gray               Feed luma instead of RGB\n";
}

int main(int argc, char** argv) {
    ChanVeseMinPartAtom atom;
    std::vector<char*> filtered = atom.tune.parse(argc, argv);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            print_tuning_help();
            break;
        }
    }
    const int fargc = static_cast<int>(filtered.size());
    char** fargv = filtered.data();
    return run_atom_main(fargc, fargv, "dis5k", [&](const AtomCli& cli) -> int {
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no inputs for chan_vese atom\n";
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
        return atom.checks.failures.empty() ? 0 : 1;
    });
}
