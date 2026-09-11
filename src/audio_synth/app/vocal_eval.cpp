#include "vocal/dataset.hpp"
#include "vocal/evaluator.hpp"
#include "vocal/features.hpp"
#include "vocal/neural_models.hpp"
#include "vocal/onnx_runtime_model.hpp"
#include "vocal/reporting.hpp"
#include "vocal/study.hpp"
#include "vocal/synthesizers.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>

namespace {
void usage() {
    std::cerr << "Usage:\n"
              << "  vocal-eval scan <LibriTTS_R-root>\n"
              << "  vocal-eval score <ref.mcep> <syn.mcep> [ref.f0 syn.f0] [ref-text hyp-text]\n"
              << "  vocal-eval render-all <output-directory> [text]\n"
              << "  vocal-eval compare-neural [text]\n"
              << "  vocal-eval features <input.wav> <output-prefix>\n"
              << "  vocal-eval onnx-infer <model.onnx> <text> <output-prefix>\n"
              << "  vocal-eval summarize <scores.csv> <output-prefix> [bootstrap-samples] [seed]\n"
              << "  vocal-eval study-manifest <wav-directory> <output-prefix> [seed]\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "scan") {
            auto items = vocal::LibriTtsDataset(argv[2]).scan();
            std::cout << "utterances=" << items.size() << '\n';
            for (std::size_t i = 0; i < std::min<std::size_t>(items.size(), 5); ++i) {
                std::cout << items[i].id << '\t' << items[i].speaker_id << '\t'
                          << items[i].normalized_text << '\n';
            }
            return 0;
        }
        if ((argc == 4 || argc == 6 || argc == 8) && std::string(argv[1]) == "score") {
            vocal::EvaluationCase item{"cli", argv[2], argv[3]};
            if (argc >= 6) { item.reference_f0 = argv[4]; item.synthesized_f0 = argv[5]; }
            if (argc == 8) { item.reference_text = argv[6]; item.hypothesis_text = argv[7]; }
            const auto result = vocal::Evaluator{}.evaluate(item);
            std::cout << std::fixed << std::setprecision(4) << "MCD_dB=" << result.mcd_db << '\n';
            if (result.pitch) {
                std::cout << "F0_RMSE_Hz=" << result.pitch->f0_rmse_hz
                          << "\nVUV_error=" << result.pitch->vuv_error_rate << '\n';
            }
            if (result.wer) std::cout << "WER=" << *result.wer << '\n';
            return 0;
        }
        if ((argc == 3 || argc == 4) && std::string(argv[1]) == "render-all") {
            const std::string text = argc == 4 ? argv[3] : "We synthesize a clear acoustic voice.";
            const auto paths = vocal::render_comparison_suite(argv[2], text);
            for (const auto& path : paths) std::cout << path.string() << '\n';
            return 0;
        }
        if ((argc == 2 || argc == 3) && std::string(argv[1]) == "compare-neural") {
            const std::string text = argc == 3 ? argv[2] : "We synthesize a clear acoustic voice.";
            const auto result = vocal::compare_neural_models({text, "demo", 24'000, 80});
            std::cout << std::fixed << std::setprecision(5)
                      << "homebrew_frames=" << result.homebrew_frames
                      << "\nonnx_style_frames=" << result.onnx_style_frames
                      << "\nduration_ratio=" << result.duration_ratio
                      << "\nlog_mel_MAE=" << result.log_mel_mae
                      << "\nlog_mel_RMSE=" << result.log_mel_rmse << '\n';
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "features") {
            const auto audio = vocal::load_wav_mono(argv[2]);
            const auto features = vocal::extract_features(audio);
            const std::string prefix = argv[3];
            vocal::write_feature_matrix(prefix + ".logmel.csv", features.log_mel);
            vocal::write_feature_matrix(prefix + ".mcep.csv", features.mcep);
            std::cout << "frames=" << features.log_mel.size() << "\nmel_bins=80\nmcep_coefficients=25\n";
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "onnx-infer") {
            vocal::OnnxRuntimeAcousticModel model(argv[2]);
            const auto mel = model.infer({argv[3], "", 24'000, 80});
            const std::string prefix = argv[4];
            vocal::metrics::Frames matrix(mel.frames, std::vector<double>(mel.bins));
            for (std::size_t frame = 0; frame < mel.frames; ++frame)
                for (std::size_t bin = 0; bin < mel.bins; ++bin)
                    matrix[frame][bin] = mel.log_mel[frame * mel.bins + bin];
            vocal::write_feature_matrix(prefix + ".logmel.csv", matrix);
            const auto& d = model.diagnostics();
            std::ofstream report(prefix + ".diagnostics.json");
            report << "{\n  \"inference_ms\": " << d.inference_ms
                   << ",\n  \"input_tokens\": " << d.input_tokens
                   << ",\n  \"output_frames\": " << d.output_frames
                   << ",\n  \"frames_per_token\": " << d.frames_per_token()
                   << ",\n  \"duration_values\": " << d.durations.size()
                   << ",\n  \"duration_frame_error\": " << d.duration_frame_error
                   << ",\n  \"zero_duration_tokens\": " << d.zero_duration_tokens
                   << ",\n  \"alignment_values\": " << d.alignment.size()
                   << ",\n  \"alignment_monotonic_violations\": " << d.alignment_monotonic_violations
                   << ",\n  \"alignment_token_coverage\": " << d.alignment_token_coverage << "\n}\n";
            std::cout << "frames=" << mel.frames << "\ninference_ms=" << d.inference_ms << '\n';
            return 0;
        }
        if ((argc >= 4 && argc <= 6) && std::string(argv[1]) == "summarize") {
            const auto bootstrap = argc >= 5 ? static_cast<std::size_t>(std::stoull(argv[4])) : 2000U;
            const auto seed = argc == 6 ? std::stoull(argv[5]) : 0x564f43414cULL;
            const auto summaries = vocal::summarize_by_speaker(vocal::load_score_csv(argv[2]), bootstrap, seed);
            const std::string prefix = argv[3];
            vocal::write_summaries_csv(prefix + ".csv", summaries);
            vocal::write_summaries_json(prefix + ".json", summaries, bootstrap, seed);
            std::cout << "summary_groups=" << summaries.size() << '\n';
            return 0;
        }
        if ((argc == 4 || argc == 5) && std::string(argv[1]) == "study-manifest") {
            vocal::StudyManifestOptions options;
            if (argc == 5) options.seed = std::stoull(argv[4]);
            const std::string prefix = argv[3];
            vocal::write_listening_study(argv[2], prefix + ".json", prefix + ".csv", options);
            std::cout << prefix << ".json\n" << prefix << ".csv\n";
            return 0;
        }
        usage();
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
