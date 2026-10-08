#include "articulation.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace vocal;
    using namespace vocal::experiment;
    auto require = [](bool value) { if (!value) throw std::runtime_error("articulation invariant failed"); };
    try {
        const std::filesystem::path fixtures = std::filesystem::path(VA_TEST_SOURCE_DIR) / "experiments/articulation/fixtures";
        CmuPhonemizer frontend(fixtures / "cmudict.dict", fixtures / "tokens.tsv");
        const auto aligned = align_text(frontend,"Hello, world hello!",read_vocabulary(fixtures / "tokens.tsv"));
        require(aligned.words.size() == 3 && aligned.words[0].token_begin == 0 && aligned.words[0].token_end == 4);
        require(aligned.words[1].token_begin == 5 && aligned.words[1].token_end == 9);
        require(aligned.words[2].token_begin == 9 && aligned.words[2].token_end == 13);
        require(aligned.phones[4] == "sp" && aligned.words[2].text == "hello");
        auto base = baseline_prosody(aligned.phones.size(),4,180,1);
        base.token_kinds = aligned.phonemes.token_kinds;
        const auto identity = articulate(aligned,base,{});
        require(identity.durations == base.durations && identity.f0_contour == base.f0_contour && identity.energy_contour == base.energy_contour);
        const auto word = articulate(aligned,base,{1,"all",1,1.2F,1.5F});
        for (std::size_t f = 0; f < base.f0_contour.size(); ++f) {
            const bool selected = f >= 20 && f < 36;
            require(std::abs(word.f0_contour[f] - (selected ? 216.F : 180.F)) < .001F);
            require(word.energy_contour[f] == (selected ? 1.5F : 1.F));
        }
        const auto phone = articulate(aligned,base,{2,"HH",2});
        require(phone.durations[9] == 8 && duration_frames(phone.durations) == duration_frames(base.durations)+4);
        for (std::size_t i = 0; i < base.durations.size(); ++i) if (i != 9) require(phone.durations[i] == base.durations[i]);
        const auto vowel = articulate(aligned,base,{0,"stressed",2});
        require(vowel.durations[3] == 8 && vowel.durations[0] == 4);
        const auto coupled = articulate(aligned,base,{0,"vowels",1,1,1.8F,0,.2F,2});
        require(coupled.f0_contour[4] > base.f0_contour[4]);
        Articulation voice(aligned, base);
        voice.select_word(0).select_vowels().scale_duration(1.3F).scale_pitch(1.2F)
             .scale_energy(1.8F).ramp_pitch(25).couple_energy_to_pitch(.2F)
             .couple_pitch_to_vowel_duration(2).apply();
        const auto combined = articulate(aligned,base,{0,"vowels",1.3F,1.2F,1.8F,25,.2F,2});
        require(voice.controls().durations == combined.durations && voice.controls().f0_contour == combined.f0_contour &&
                voice.controls().energy_contour == combined.energy_contour);
        require(voice.baseline().f0_contour == base.f0_contour && voice.pending_edit().word_index == -1);
        voice.reset().select_word(2).select_phone("HH").scale_duration(2).apply();
        require(voice.controls().durations == phone.durations);
        const auto committed = voice.controls();
        const std::vector<ArticulationEdit> failing_batch{{0,"all",2},{0,"NOTAPHONE"}};
        bool batch_rejected = false;
        try { voice.apply(failing_batch); } catch (const std::invalid_argument&) { batch_rejected = true; }
        require(batch_rejected && voice.controls().durations == committed.durations &&
                voice.controls().f0_contour == committed.f0_contour);
        voice.reset().select_word(0).select_stressed_vowels().scale_duration(2).apply();
        require(voice.controls().durations == vowel.durations);
        voice.reset().select_all_words().select_consonants().scale_duration(2).apply();
        require(voice.controls().durations[0] == 8 && voice.controls().durations[1] == 4);
        voice.reset().select_unvoiced_consonants().scale_duration(2).apply();
        require(voice.controls().durations[0] == 8 && voice.controls().durations[2] == 4);
        voice.select_word(1).clear_pending();
        require(voice.pending_edit().word_index == -1);
        voice.reset();
        for (const auto& item : Articulation::expressive_sweep(0)) {
            // S sweep requires a different sentence; exercise all other reusable presets here.
            if (item.name.starts_with("sibilant")) continue;
            voice.reset().apply(item.edits);
            validate_prosody(voice.controls(), aligned.phones.size());
        }
        require(Articulation::expressive_sweep(0).size() == 21);
        auto zero = base; zero.durations[0] = 0;
        zero.f0_contour.resize(duration_frames(zero.durations)); zero.energy_contour.resize(duration_frames(zero.durations));
        const auto preserved = articulate(aligned,zero,{0,"all",2});
        require(preserved.durations[0] == 0);
        for (auto bad : std::vector<ArticulationEdit>{{3,"all"},{0,"NOTAPHONE"},{0,"all",0},{0,"all",1,1,3}}) {
            bool rejected = false;
            try { (void)articulate(aligned,base,bad); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected);
        }
        std::cout << "Alignment, word/phone isolation, fluent controls, atomic batches, sweep presets, zero durations and bounds passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
