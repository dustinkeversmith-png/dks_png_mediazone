#include "articulation.hpp"
#include "app/cli_options.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <utility>

namespace vocal::experiment {
namespace {
bool vowel(ProsodyTokenKind kind) {
    return kind == ProsodyTokenKind::Vowel || kind == ProsodyTokenKind::StressedVowel;
}
void bounded(float value, float low, float high, const char* name) {
    if (!std::isfinite(value) || value < low || value > high)
        throw std::invalid_argument(std::string(name) + " outside experimental bounds");
}
bool matches(std::string_view selector, std::string phone, ProsodyTokenKind kind) {
    if (kind == ProsodyTokenKind::Boundary || phone == "sp") return false;
    if (selector == "all") return true;
    if (selector == "vowels") return vowel(kind);
    if (selector == "stressed") return kind == ProsodyTokenKind::StressedVowel;
    if (selector == "unvoiced") return kind == ProsodyTokenKind::UnvoicedConsonant;
    if (selector == "consonants") return !vowel(kind);
    if (selector == phone) return true;
    if (!phone.empty() && std::isdigit(static_cast<unsigned char>(phone.back()))) phone.pop_back();
    return selector == phone;
}
}

std::map<std::int64_t, std::string> read_vocabulary(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read experimental vocabulary");
    std::map<std::int64_t, std::string> result;
    bool arpabet = false;
    for (std::string line; std::getline(input, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "# frontend=arpabet") { arpabet = true; continue; }
        if (line.empty() || line[0] == '#') continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos || line.find(',', tab) != std::string::npos)
            throw std::invalid_argument("experiment requires one ID per ARPAbet phone");
        const auto id = cli::Options::parse<std::int64_t>(line.substr(tab + 1));
        if (!result.emplace(id, line.substr(0, tab)).second)
            throw std::invalid_argument("duplicate experimental phone ID");
    }
    if (!arpabet || result.empty()) throw std::invalid_argument("experiment requires the trained ARPAbet vocabulary");
    return result;
}

AlignedText align_text(const CmuPhonemizer& frontend, std::string_view text,
                       const std::map<std::int64_t, std::string>& vocabulary) {
    AlignedText result;
    result.phonemes = frontend.phonemize(text);
    if (result.phonemes.words == 0 || result.phonemes.missing_model_symbols)
        throw std::invalid_argument("text must contain words with complete model symbols");
    for (auto id : result.phonemes.token_ids) result.phones.push_back(vocabulary.at(id));
    std::size_t cursor = 0, begin = 0;
    std::string word;
    auto flush = [&](std::size_t end) {
        if (word.empty()) return;
        while (cursor < result.phones.size() && result.phones[cursor] == "sp") ++cursor;
        const auto part = frontend.phonemize(word);
        if (part.token_ids.empty() || part.missing_model_symbols || cursor + part.token_ids.size() > result.phones.size() ||
            !std::equal(part.token_ids.begin(), part.token_ids.end(), result.phonemes.token_ids.begin() + cursor))
            throw std::invalid_argument("word alignment disagrees with whole-text frontend");
        result.words.push_back({word, begin, end, cursor, cursor + part.token_ids.size()});
        cursor += part.token_ids.size(); word.clear();
    };
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto raw = static_cast<unsigned char>(text[i]);
        const char c = static_cast<char>(std::tolower(raw));
        if (std::isalnum(raw) || (c == '\'' && !word.empty())) {
            if (word.empty()) begin = i;
            word += c;
        } else flush(i);
    }
    flush(text.size());
    while (cursor < result.phones.size() && result.phones[cursor] == "sp") ++cursor;
    if (cursor != result.phones.size() || result.words.size() != result.phonemes.words)
        throw std::invalid_argument("incomplete experimental word alignment");
    return result;
}

ProsodyControls articulate(const AlignedText& text, const ProsodyControls& base, const ArticulationEdit& edit) {
    validate_prosody(base, text.phones.size());
    if (base.token_kinds.size() != text.phones.size()) throw std::invalid_argument("articulation needs phonetic annotations");
    if (edit.word_index < -1 || (edit.word_index >= 0 && static_cast<std::size_t>(edit.word_index) >= text.words.size()))
        throw std::invalid_argument("word index outside text");
    bounded(edit.duration_scale, .25F, 4.F, "duration scale");
    bounded(edit.pitch_scale, .45F, 1.8F, "pitch scale");
    bounded(edit.energy_scale, .1F, 2.5F, "energy scale");
    bounded(edit.pitch_rise_hz, -100.F, 100.F, "pitch rise");
    bounded(edit.effort_gain, 0.F, .2F, "effort gain");
    bounded(edit.vowel_peak_gain, 0.F, 2.F, "vowel peak gain");
    std::vector<std::size_t> offsets(base.durations.size() + 1);
    for (std::size_t i = 0; i < base.durations.size(); ++i)
        offsets[i + 1] = offsets[i] + static_cast<std::size_t>(base.durations[i]);
    auto edited = base;
    bool selected = false;
    for (std::size_t w = 0; w < text.words.size(); ++w) {
        if (edit.word_index >= 0 && w != static_cast<std::size_t>(edit.word_index)) continue;
        const auto& word = text.words[w];
        const auto word_frames = offsets[word.token_end] - offsets[word.token_begin];
        for (std::size_t i = word.token_begin; i < word.token_end; ++i) {
            if (!matches(edit.selector, text.phones[i], base.token_kinds[i])) continue;
            selected = true;
            if (base.durations[i] == 0) continue;
            double peak = 0;
            for (std::size_t f = offsets[i]; f < offsets[i + 1]; ++f) {
                const double progress = static_cast<double>(f - offsets[word.token_begin]) /
                                        static_cast<double>(std::max<std::size_t>(1, word_frames - 1));
                const double energy = base.energy_contour[f] * edit.energy_scale;
                double pitch = base.f0_contour[f] * edit.pitch_scale + edit.pitch_rise_hz * progress;
                if (vowel(base.token_kinds[i])) {
                    const double effort = std::clamp(std::log((energy + .01) / (base.energy_contour[f] + .01)), -1., 1.);
                    pitch *= std::exp(edit.effort_gain * effort);
                }
                if (base.f0_contour[f] > 0) {
                    if (pitch != static_cast<double>(base.f0_contour[f]))
                        edited.f0_contour[f] = static_cast<float>(std::clamp(pitch, 40., 500.));
                    peak = std::max(peak, std::max(0., std::log(static_cast<double>(edited.f0_contour[f]) / base.f0_contour[f])));
                }
                edited.energy_contour[f] = static_cast<float>(energy);
            }
            const double elongation = vowel(base.token_kinds[i]) ?
                1. + edit.vowel_peak_gain * std::pow(std::min(peak, .7), 2.) : 1.;
            edited.durations[i] = std::max<std::int64_t>(1, std::llround(base.durations[i] * edit.duration_scale * elongation));
        }
    }
    if (!selected) throw std::invalid_argument("phone selector matched no tokens in selected words");
    const auto frames = duration_frames(edited.durations);
    auto output = edited;
    output.f0_contour.clear(); output.energy_contour.clear();
    output.f0_contour.reserve(frames); output.energy_contour.reserve(frames);
    for (std::size_t i = 0; i < base.durations.size(); ++i) {
        const auto old_count = static_cast<std::size_t>(base.durations[i]);
        const auto count = static_cast<std::size_t>(edited.durations[i]);
        for (std::size_t f = 0; f < count; ++f) {
            const auto source = offsets[i] + std::min(old_count - 1, f * old_count / count);
            output.f0_contour.push_back(edited.f0_contour[source]);
            output.energy_contour.push_back(edited.energy_contour[source]);
        }
    }
    validate_prosody(output, text.phones.size());
    return output;
}

std::vector<ArticulationEdit> read_edits(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read articulation edit table");
    std::vector<ArticulationEdit> result;
    for (std::string line; std::getline(input, line);) {
        line.resize(line.find('#') == std::string::npos ? line.size() : line.find('#'));
        std::istringstream row(line);
        std::vector<std::string> fields;
        for (std::string value; row >> value;) fields.push_back(value);
        if (fields.empty()) continue;
        if (fields.size() != 8 || result.size() >= 128) throw std::invalid_argument("edit rows require 8 fields; maximum 128 rows");
        result.push_back({cli::Options::parse<std::int64_t>(fields[0]), fields[1],
            cli::Options::parse<float>(fields[2]), cli::Options::parse<float>(fields[3]),
            cli::Options::parse<float>(fields[4]), cli::Options::parse<float>(fields[5]),
            cli::Options::parse<float>(fields[6]), cli::Options::parse<float>(fields[7])});
    }
    if (result.empty()) throw std::invalid_argument("empty articulation edit table");
    return result;
}
Articulation::Articulation(AlignedText text, ProsodyControls predicted)
    : text_(std::move(text)), baseline_(std::move(predicted)), controls_(baseline_) {
    validate_prosody(baseline_, text_.phones.size());
    if (baseline_.token_kinds.size() != text_.phones.size() ||
        text_.phonemes.token_ids.size() != text_.phones.size() || text_.words.empty())
        throw std::invalid_argument("articulation needs aligned text and phonetic annotations");
    std::size_t end = 0;
    for (const auto& word : text_.words) {
        if (word.token_begin < end || word.token_begin >= word.token_end || word.token_end > text_.phones.size())
            throw std::invalid_argument("invalid articulation word interval");
        end = word.token_end;
    }
}

Articulation& Articulation::select_word(std::int64_t index) {
    if (index < -1 || (index >= 0 && static_cast<std::size_t>(index) >= text_.words.size()))
        throw std::invalid_argument("word index outside text");
    pending_.word_index = index;
    return *this;
}
Articulation& Articulation::select_all_words() { return select_word(-1); }
Articulation& Articulation::select_phone(std::string selector) {
    if (selector.empty()) throw std::invalid_argument("empty articulation phone selector");
    pending_.selector = std::move(selector);
    return *this;
}
Articulation& Articulation::select_vowels() { return select_phone("vowels"); }
Articulation& Articulation::select_stressed_vowels() { return select_phone("stressed"); }
Articulation& Articulation::select_consonants() { return select_phone("consonants"); }
Articulation& Articulation::select_unvoiced_consonants() { return select_phone("unvoiced"); }
Articulation& Articulation::scale_duration(float scale) {
    bounded(scale, .25F, 4.F, "duration scale"); pending_.duration_scale = scale; return *this;
}
Articulation& Articulation::scale_pitch(float scale) {
    bounded(scale, .45F, 1.8F, "pitch scale"); pending_.pitch_scale = scale; return *this;
}
Articulation& Articulation::scale_energy(float scale) {
    bounded(scale, .1F, 2.5F, "energy scale"); pending_.energy_scale = scale; return *this;
}
Articulation& Articulation::ramp_pitch(float rise_hz) {
    bounded(rise_hz, -100.F, 100.F, "pitch rise"); pending_.pitch_rise_hz = rise_hz; return *this;
}
Articulation& Articulation::couple_energy_to_pitch(float gain) {
    bounded(gain, 0.F, .2F, "effort gain"); pending_.effort_gain = gain; return *this;
}
Articulation& Articulation::couple_pitch_to_vowel_duration(float gain) {
    bounded(gain, 0.F, 2.F, "vowel peak gain"); pending_.vowel_peak_gain = gain; return *this;
}
Articulation& Articulation::apply() { return apply(pending_); }
Articulation& Articulation::apply(const ArticulationEdit& edit) {
    return apply(std::span<const ArticulationEdit>(&edit, 1));
}
Articulation& Articulation::apply(std::span<const ArticulationEdit> edits) {
    auto next = controls_;
    for (const auto& edit : edits) next = articulate(text_, next, edit);
    controls_ = std::move(next);
    return clear_pending();
}
Articulation& Articulation::apply_file(const std::filesystem::path& path) {
    const auto edits = read_edits(path);
    return apply(edits);
}
Articulation& Articulation::clear_pending() { pending_ = {}; return *this; }
Articulation& Articulation::reset() { controls_ = baseline_; return clear_pending(); }
const AlignedText& Articulation::text() const noexcept { return text_; }
const ProsodyControls& Articulation::baseline() const noexcept { return baseline_; }
const ProsodyControls& Articulation::controls() const noexcept { return controls_; }
const ArticulationEdit& Articulation::pending_edit() const noexcept { return pending_; }

std::vector<ArticulationCase> Articulation::expressive_sweep(std::int64_t word) {
    if (word < 0) throw std::invalid_argument("sweep requires a nonnegative word index");
    return {
        {"baseline", {}},
        {"word_short", {{word,"all",.6F}}}, {"word_long", {{word,"all",1.8F}}},
        {"pitch_low", {{word,"all",1,.7F}}}, {"pitch_high", {{word,"all",1,1.4F}}},
        {"energy_soft", {{word,"all",1,1,.5F}}}, {"energy_strong", {{word,"all",1,1,1.8F}}},
        {"vowels_short", {{word,"vowels",.6F}}}, {"vowels_long", {{word,"vowels",1.8F}}},
        {"stressed_emphasis", {{word,"stressed",1.4F,1.12F,1.3F}}},
        {"consonants_short", {{-1,"consonants",.6F}}},
        {"consonants_long", {{-1,"consonants",1.8F}}},
        {"unvoiced_long", {{-1,"unvoiced",1.8F}}},
        {"sibilant_short", {{-1,"S",.5F}}}, {"sibilant_long", {{-1,"S",2.5F,1,1.1F}}},
        {"word_rise", {{word,"all",1,1,1,60}}}, {"word_fall", {{word,"all",1,1,1,-60}}},
        {"effort_coupled", {{word,"all",1,1,1.6F,0,.18F}}},
        {"vowel_peak_coupled", {{word,"vowels",1,1.3F,1.2F,30,.12F,2.F}}},
        {"stress_test_low", {{word,"all",.35F,.45F,.15F}}},
        {"stress_test_high", {{word,"all",3.F,1.8F,2.5F}}},
    };
}
} // namespace vocal::experiment
