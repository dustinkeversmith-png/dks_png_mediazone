#include "vocal/metrics.hpp"
#include "vocal/features.hpp"
#include "vocal/phonemizer.hpp"
#include "vocal/reporting.hpp"

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
    // A known analytic waveform tests feature extraction without an archived synthesizer.
    vocal::Waveform canonical_audio{24'000, std::vector<float>(2'400)};
    for (std::size_t i = 0; i < canonical_audio.samples.size(); ++i)
        canonical_audio.samples[i] = .2F * static_cast<float>(std::sin(2.0 * 3.141592653589793 * 200.0 * i / 24'000.0));
    const auto features = vocal::extract_features(canonical_audio);
    if (features.log_mel.empty() || features.log_mel.front().size() != 80 ||
        features.mcep.front().size() != 25) ++failures;
    const auto wav_path = std::filesystem::current_path() / "vocal_acoustics_roundtrip.wav";
    vocal::write_wav_pcm16(wav_path, canonical_audio);
    const auto loaded = vocal::load_wav_mono(wav_path);
    std::filesystem::remove(wav_path);
    if (loaded.sample_rate_hz != 24'000 || loaded.samples.size() != canonical_audio.samples.size()) ++failures;
    const auto resampled = vocal::resample_waveform({22'050, std::vector<float>(22'050, .25F)});
    if (resampled.sample_rate_hz != 24'000 || resampled.samples.size() != 24'000 ||
        std::abs(resampled.samples[12'000] - .25F) > 1e-6F) ++failures;
    if (!vocal::resample_waveform({22'050, {}}).samples.empty()) ++failures;
    const std::vector<vocal::ScoredUtterance> scored{
        {"a", "speaker-a", 4.0, 10.0, .1, .05},
        {"b", "speaker-a", 6.0, 14.0, .2, .15},
        {"c", "speaker-b", 5.0, std::nullopt, .3, .10}};
    const auto summaries = vocal::summarize_by_speaker(scored, 100, 42);
    if (summaries.size() != 3 || !summaries.front().mcd_db) ++failures;
    const auto fixture_root = std::filesystem::path(VA_TEST_SOURCE_DIR) / "tests" / "fixtures";
    const vocal::CmuPhonemizer phonemizer(fixture_root / "cmudict.dict", fixture_root / "tokens.tsv");
    const auto pronunciation = phonemizer.phonemize("hello");
    if (pronunciation.dictionary_hits != 1 || pronunciation.fallback_words != 0 ||
        pronunciation.missing_model_symbols != 0 || pronunciation.token_ids.empty()) ++failures;
    if (pronunciation.token_kinds.size() != pronunciation.token_ids.size()) ++failures;
    bool unvoiced_found = false, stressed_found = false;
    for (auto kind : pronunciation.token_kinds) {
        unvoiced_found |= kind == vocal::ProsodyTokenKind::UnvoicedConsonant;
        stressed_found |= kind == vocal::ProsodyTokenKind::StressedVowel;
    }
    if (!unvoiced_found || !stressed_found || pronunciation.token_kinds.back() != vocal::ProsodyTokenKind::Boundary) ++failures;
    if (failures) std::cerr << failures << " test(s) failed\n";
    return failures == 0 ? 0 : 1;
}
