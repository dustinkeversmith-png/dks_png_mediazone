#include "vocal/metrics.hpp"
#include "vocal/features.hpp"
#include "vocal/neural_models.hpp"
#include "vocal/reporting.hpp"
#include "vocal/synthesizers.hpp"

#include <cmath>
#include <iostream>
#include <filesystem>

int main() {
    int failures = 0;
    const vocal::metrics::Frames a{{9.0, 1.0, 2.0}, {8.0, 2.0, 3.0}};
    if (std::abs(vocal::metrics::mcd_db(a, a)) > 1e-12) ++failures;
    if (std::abs(vocal::metrics::word_error_rate("Hello, world!", "hello world")) > 1e-12) ++failures;
    if (std::abs(vocal::metrics::word_error_rate("one two three", "one three") - 1.0 / 3.0) > 1e-12) ++failures;
    const double ref[] = {100.0, 0.0, 120.0, 130.0};
    const double syn[] = {110.0, 90.0, 120.0, 0.0};
    const auto score = vocal::metrics::pitch(ref, syn);
    if (score.jointly_voiced_frames != 2 || std::abs(score.vuv_error_rate - 0.5) > 1e-12) ++failures;
    vocal::HomebrewAcousticModel homebrew;
    vocal::OnnxStyleAcousticModel onnx;
    const auto request = vocal::SynthesisRequest{"hello", "test", 24'000, 80};
    const auto homebrew_mel = homebrew.infer(request);
    const auto onnx_mel = onnx.infer(request);
    if (homebrew_mel.frames == 0 || homebrew_mel.log_mel.size() != homebrew_mel.frames * 80) ++failures;
    if (onnx_mel.frames <= homebrew_mel.frames || onnx_mel.log_mel.size() != onnx_mel.frames * 80) ++failures;
    const auto comparison = vocal::compare_neural_models(request);
    if (comparison.log_mel_mae <= 0.0 || comparison.duration_ratio <= 1.0) ++failures;
    const auto audio = vocal::synthesize_demo(vocal::SynthesizerKind::formant, "test", 16'000);
    if (audio.samples.empty() || audio.sample_rate_hz != 16'000) ++failures;
    const auto canonical_audio = vocal::synthesize_demo(vocal::SynthesizerKind::formant, "test", 24'000);
    const auto features = vocal::extract_features(canonical_audio);
    if (features.log_mel.empty() || features.log_mel.front().size() != 80 ||
        features.mcep.front().size() != 25) ++failures;
    const auto wav_path = std::filesystem::current_path() / "vocal_acoustics_roundtrip.wav";
    vocal::write_wav_pcm16(wav_path, canonical_audio);
    const auto loaded = vocal::load_wav_mono(wav_path);
    std::filesystem::remove(wav_path);
    if (loaded.sample_rate_hz != 24'000 || loaded.samples.size() != canonical_audio.samples.size()) ++failures;
    const std::vector<vocal::ScoredUtterance> scored{
        {"a", "speaker-a", 4.0, 10.0, .1, .05},
        {"b", "speaker-a", 6.0, 14.0, .2, .15},
        {"c", "speaker-b", 5.0, std::nullopt, .3, .10}};
    const auto summaries = vocal::summarize_by_speaker(scored, 100, 42);
    if (summaries.size() != 3 || !summaries.front().mcd_db) ++failures;
    if (failures) std::cerr << failures << " test(s) failed\n";
    return failures == 0 ? 0 : 1;
}
