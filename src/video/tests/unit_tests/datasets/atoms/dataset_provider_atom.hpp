#pragma once

#include "test_harness.hpp"
#include "atom_config.hpp"
#include "datasets/provider_factory.hpp"
#include "datasets/dataset_provider.hpp"

#include <iostream>
#include <sstream>

namespace dataset_atoms {

inline vision::GrayImage downscale_for_preview(const vision::GrayImage& src, int max_side = 256) {
    return downscale_max_side(src, max_side);
}

inline vision::GrayImage palette_colormap(const vision::GrayImage& labels) {
    vision::GrayImage out = labels;
    for (uint8_t& p : out.data) {
        if (p == 0) {
            continue;
        }
        p = static_cast<uint8_t>(40 + (static_cast<unsigned>(p) * 47u) % 200u);
    }
    return out;
}

inline std::vector<std::string> render_sample_artifacts(const datasets::VisionSample& sample,
                                                        const std::string& art_dir,
                                                        const std::string& provider_name) {
    std::vector<std::string> files;
    const std::string id = sample.id.empty() ? "sample" : sample.id;

    vision::GrayImage luma;
    if (!sample.luma.empty()) {
        luma = downscale_for_preview(sample.luma);
    } else if (!sample.rgb.empty()) {
        luma = downscale_for_preview(vision::rgb_to_luma(sample.rgb));
    }

    if (!luma.empty()) {
        const std::string name = id + "_luma.pgm";
        vision::save_pgm(vision::join_path(art_dir, name), luma);
        files.push_back(name);
    }

    if (!sample.mask.empty()) {
        const auto mask = downscale_for_preview(sample.mask);
        const std::string mask_name = id + "_mask.pgm";
        vision::save_pgm(vision::join_path(art_dir, mask_name), mask);
        files.push_back(mask_name);

        if (provider_name == "ade20k") {
            const std::string cmap_name = id + "_palette_colormap.pgm";
            vision::save_pgm(vision::join_path(art_dir, cmap_name), palette_colormap(mask));
            files.push_back(cmap_name);
        }
        if (!luma.empty()) {
            const std::string overlay_name = id + "_segmentation_overlay.pgm";
            vision::save_pgm(vision::join_path(art_dir, overlay_name), overlay_mask(luma, mask));
            files.push_back(overlay_name);
        }
    }

    if (!sample.boundary.empty()) {
        const auto boundary = downscale_for_preview(sample.boundary);
        const std::string b_name = id + "_boundary.pgm";
        vision::save_pgm(vision::join_path(art_dir, b_name), boundary);
        files.push_back(b_name);
        if (!luma.empty()) {
            const std::string overlay_name = id + "_boundary_overlay.pgm";
            vision::save_pgm(vision::join_path(art_dir, overlay_name), overlay_mask(luma, boundary));
            files.push_back(overlay_name);
        }
    }

    return files;
}

inline int run_dataset_provider_atom(int argc, char** argv, const char* provider_name,
                                     const char* atom_name) {
    return run_atom_main(argc, argv, provider_name, [&](const AtomCli& cli) -> int {
        const auto vision_root = fs::path(vision::find_vision_root(argc > 0 ? argv[0] : nullptr));
        std::unique_ptr<datasets::DatasetProvider> provider;
        try {
            provider = datasets::make_provider(provider_name, vision_root);
        } catch (const std::exception& ex) {
            std::cerr << "provider error: " << ex.what() << "\n";
            return 1;
        }
        if (!provider || provider->size() == 0) {
            std::cerr << "no samples for provider " << provider_name << "\n";
            return 1;
        }

        const vision::AtomConfig config = vision::load_atom_config_near(argc > 0 ? argv[0] : nullptr);
        std::cout << provider_name << " samples: " << provider->size() << "\n";
        std::cout << "root      : " << provider->root().string() << "\n";

        if (cli.list_only) {
            for (size_t i = 0; i < provider->size(); ++i) {
                const auto s = provider->load_sample(i);
                std::cout << "  [" << i << "] " << s.id;
                if (!s.image_path.empty()) {
                    std::cout << "  " << s.image_path;
                }
                if (!s.gt_path.empty()) {
                    std::cout << "  gt=" << s.gt_path;
                }
                std::cout << "\n";
            }
            return 0;
        }

        const std::string art = make_artifact_dir(cli.artifact_dir);
        AtomDemoReport report{atom_name};
        ScopedTimer timer(&report.elapsed_ms);

        const size_t max_samples = config.artifacts.empty() ? 4 : 4;
        const size_t n = std::min(provider->size(), max_samples);
        report.n_inputs = static_cast<int>(n);
        report.notes.push_back(std::string("provider=") + provider_name);
        report.notes.push_back(std::string("root=") + provider->root().string());
        for (const auto& stage : config.pre_processing_stages) {
            report.notes.push_back("render:" + stage);
        }

        std::ostringstream samples_tsv;
        samples_tsv << "idx\tid\timage_path\tgt_path\thas_mask\thas_boundary\n";
        std::vector<std::string> all_files;

        for (size_t i = 0; i < n; ++i) {
            const auto sample = provider->load_sample(i);
            auto files = render_sample_artifacts(sample, art, provider_name);
            all_files.insert(all_files.end(), files.begin(), files.end());
            samples_tsv << i << '\t' << sample.id << '\t' << sample.image_path << '\t' << sample.gt_path
                        << '\t' << (!sample.mask.empty() ? 1 : 0) << '\t'
                        << (!sample.boundary.empty() ? 1 : 0) << "\n";
            ++report.n_outputs;
        }

        vision::write_text_file(vision::join_path(art, "samples.tsv"), samples_tsv.str());
        all_files.insert(all_files.begin(), "samples.tsv");
        write_atom_manifest(art, report, all_files);
        report.print();
        std::cout << "artifacts -> " << art << "\n";
        return 0;
    });
}

}  // namespace dataset_atoms
