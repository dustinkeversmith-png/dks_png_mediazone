#pragma once
#include "vocal/phonemizer.hpp"
#include <map>
#include <span>

namespace vocal::experiment {
struct WordSpan {
    std::string text;
    std::size_t byte_begin{}, byte_end{}, token_begin{}, token_end{};
};
struct AlignedText {
    PhonemizationResult phonemes;
    std::vector<WordSpan> words;
    std::vector<std::string> phones;
};
// Isolated ARPAbet adapter; the production frontend and token stream are unchanged.
AlignedText align_text(const CmuPhonemizer& frontend, std::string_view text,
                       const std::map<std::int64_t, std::string>& vocabulary);
std::map<std::int64_t, std::string> read_vocabulary(const std::filesystem::path& path);
struct ArticulationEdit {
    std::int64_t word_index{-1}; // -1 selects all words, never boundary tokens.
    std::string selector{"all"}; // all, vowels, stressed, consonants, unvoiced, or ARPAbet phone.
    float duration_scale{1}, pitch_scale{1}, energy_scale{1}, pitch_rise_hz{0};
    float effort_gain{0}, vowel_peak_gain{0};
};
ProsodyControls articulate(const AlignedText& text, const ProsodyControls& base,
                          const ArticulationEdit& edit);
std::vector<ArticulationEdit> read_edits(const std::filesystem::path& path);

struct ArticulationCase {
    std::string name;
    std::vector<ArticulationEdit> edits;
};

// Owns the alignment and predicted contours; no ONNX session or CLI dependency.
// Named setters prepare ONE combined edit. apply() commits it atomically and
// clears the pending edit. Subsequent applications compound existing controls.
class Articulation {
public:
    Articulation(AlignedText text, ProsodyControls predicted);

    Articulation& select_word(std::int64_t index); // Zero-based; -1 means all words.
    Articulation& select_all_words();
    Articulation& select_phone(std::string selector);
    Articulation& select_vowels();
    Articulation& select_stressed_vowels();
    Articulation& select_consonants();
    Articulation& select_unvoiced_consonants();
    Articulation& scale_duration(float scale);
    Articulation& scale_pitch(float scale);
    Articulation& scale_energy(float scale);
    Articulation& ramp_pitch(float rise_hz);
    Articulation& couple_energy_to_pitch(float gain);
    Articulation& couple_pitch_to_vowel_duration(float gain);

    Articulation& apply();
    Articulation& apply(const ArticulationEdit& edit);
    Articulation& apply(std::span<const ArticulationEdit> edits);
    Articulation& apply_file(const std::filesystem::path& path);
    Articulation& clear_pending();
    Articulation& reset(); // Restore prediction and clear pending selection/controls.

    [[nodiscard]] const AlignedText& text() const noexcept;
    [[nodiscard]] const ProsodyControls& baseline() const noexcept;
    [[nodiscard]] const ProsodyControls& controls() const noexcept;
    [[nodiscard]] const ArticulationEdit& pending_edit() const noexcept;
    // Same 21 named applications as the CLI; each is relative to the baseline.
    [[nodiscard]] static std::vector<ArticulationCase> expressive_sweep(std::int64_t word);

private:
    AlignedText text_;
    ProsodyControls baseline_, controls_;
    ArticulationEdit pending_;
};
} // namespace vocal::experiment
