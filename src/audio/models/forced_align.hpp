// Forced Viterbi alignment (Path 2): given an utterance and the word sequence
// that was actually spoken, find the maximum-likelihood assignment of frames to
// HMM states.
//
// Why this exists: TIMIT ships hand-marked phone boundaries, so the first
// acoustic model could be trained by splitting each labelled segment into three
// equal parts. LibriSpeech ships only sentence text, so there are no boundaries
// to split - they have to be inferred. Forced alignment is also better than
// uniform splitting even when boundaries *are* available, because real phone
// durations vary with stress and position: a stop burst is 2 frames, a stressed
// vowel is 20, and equal thirds smear the models of both.
//
// The search is a plain Viterbi over a linear chain (no LM, no beam): the word
// sequence is known, so the only unknowns are the boundaries. Cost is
// O(frames * states) with states = 3 * number of phones in the utterance, which
// for a 30 s utterance is ~3000 x ~900 - small enough to do exactly, with a
// full backtrace, rather than approximately.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "acoustic_model.hpp"
#include "g2p.hpp"
#include "lexicon.hpp"
#include "mfcc.hpp"
#include "phone_set.hpp"
#include "triphone.hpp"

namespace models {

// One aligned segment of an utterance.
struct AlignedSegment {
    int phone = -1;
    int start_frame = 0;
    int stop_frame = 0;
};

// Maps words to phone sequences, inserting optional silence around words.
class PronunciationTable {
public:
    void build(const Lexicon& lexicon) {
        table_.clear();
        for (const Pronunciation& pron : lexicon.pronunciations()) {
            // Keep the first pronunciation per word: alignment does not need
            // variants, and a single choice keeps the chain deterministic.
            const std::string& word = lexicon.word(pron.word_id);
            table_.emplace(word, pron.phones);
        }
    }

    const std::vector<int>* find(const std::string& word) const {
        const auto it = table_.find(word);
        return it == table_.end() ? nullptr : &it->second;
    }

    // Word sequence -> phone sequence with leading/trailing silence.
    //
    // Words missing from CMUDict fall back to letter-to-sound rules and are
    // counted in `guessed`. Alignment needs *a* pronunciation for every word in
    // the audio: skipping the word would leave its frames to be absorbed by its
    // neighbours, which corrupts their models worse than an approximate
    // spelling of one rare proper noun does.
    bool phones_for(const std::vector<std::string>& words, std::vector<int>& out,
                    int* guessed = nullptr, bool silence_between_words = false) const {
        out.clear();
        const int silence = phone_id("SIL");
        out.push_back(silence);
        for (const std::string& word : words) {
            const std::vector<int>* phones = find(word);
            if (phones) {
                out.insert(out.end(), phones->begin(), phones->end());
            } else {
                const std::vector<int> fallback = g2p_.convert(word);
                if (fallback.empty()) return false;
                out.insert(out.end(), fallback.begin(), fallback.end());
                if (guessed) ++*guessed;
            }
            if (silence_between_words) out.push_back(silence);
        }
        if (out.back() != silence) out.push_back(silence);
        return true;
    }

    size_t size() const { return table_.size(); }

private:
    std::unordered_map<std::string, std::vector<int>> table_;
    GraphemeToPhoneme g2p_;
};

class ForcedAligner {
public:
    // Aligns `features` to `phones`. Returns segments in time order, or an
    // empty vector if the utterance is too short for the phone sequence.
    // `tree` is optional: with it, each chain position emits from the tied
    // triphone state for its context instead of the monophone state, so
    // re-alignment benefits from the sharper context-dependent models.
    std::vector<AlignedSegment> align(const FeatureMatrix& features,
                                      const std::vector<int>& phones,
                                      const AcousticModel& model,
                                      const TriphoneTree* tree = nullptr) {
        const int T = features.num_frames;
        const int S = static_cast<int>(phones.size()) * kNumStatesPerPhone;
        if (T == 0 || S == 0 || T < S / 2) return {};

        constexpr float kNegInf = -std::numeric_limits<float>::infinity();
        scores_.assign(S, kNegInf);
        next_.assign(S, kNegInf);
        // Backpointer per (frame, state): 0 = self-loop, 1 = came from s-1.
        back_.assign(static_cast<size_t>(T) * S, 0);

        // Precomputed per-state emission indices and transition costs.
        state_phone_.resize(S);
        state_hmm_.resize(S);
        const int silence = phone_id("SIL");
        for (int s = 0; s < S; ++s) {
            const int index = s / kNumStatesPerPhone;
            const int sub = s % kNumStatesPerPhone;
            const int phone = phones[index];
            state_phone_[s] = phone;
            if (tree) {
                const int left = index > 0 ? phones[index - 1] : silence;
                const int right = index + 1 < static_cast<int>(phones.size())
                                      ? phones[index + 1]
                                      : silence;
                state_hmm_[s] = tree->senone(left, phone, sub, right);
            } else {
                state_hmm_[s] = state_index(phone, sub);
            }
        }

        for (int t = 0; t < T; ++t) {
            model.score_frame(features.frame(t), am_);
            uint8_t* back = back_.data() + static_cast<size_t>(t) * S;

            // A path must reach state s by frame t, and can still reach the
            // last state by frame T: outside that diagonal the cell is dead.
            const int lo = std::max(0, S - (T - t) * 2);
            const int hi = std::min(S - 1, t);

            for (int s = hi; s >= lo; --s) {
                float best;
                uint8_t choice;
                if (t == 0) {
                    // Only the first state can be occupied at frame 0.
                    best = (s == 0) ? 0.0f : kNegInf;
                    choice = 0;
                } else {
                    const float stay = scores_[s] == kNegInf
                                           ? kNegInf
                                           : scores_[s] + model.log_self_loop(state_phone_[s]);
                    const float enter =
                        (s > 0 && scores_[s - 1] != kNegInf)
                            ? scores_[s - 1] + model.log_exit(state_phone_[s - 1])
                            : kNegInf;
                    if (enter > stay) {
                        best = enter;
                        choice = 1;
                    } else {
                        best = stay;
                        choice = 0;
                    }
                }
                back[s] = choice;
                next_[s] = best == kNegInf ? kNegInf : best + am_[state_hmm_[s]];
            }
            for (int s = 0; s < lo; ++s) next_[s] = kNegInf;
            for (int s = hi + 1; s < S; ++s) next_[s] = kNegInf;
            scores_.swap(next_);
        }

        if (scores_[S - 1] == kNegInf) return {};  // no complete path

        // Backtrace over the chain. The path gives the exact state occupied at
        // every frame, including the within-phone state boundaries - keep them
        // rather than re-splitting each phone uniformly afterwards, which would
        // throw away most of what the alignment just computed.
        chain_position_.assign(T, 0);
        int state = S - 1;
        for (int t = T - 1; t >= 0; --t) {
            chain_position_[t] = state;
            if (back_[static_cast<size_t>(t) * S + state] == 1 && state > 0) --state;
        }

        state_path_.resize(T);
        phone_of_frame_.resize(T);
        for (int t = 0; t < T; ++t) {
            const int position = chain_position_[t];
            const int phone = phones[position / kNumStatesPerPhone];
            phone_of_frame_[t] = phone;
            state_path_[t] = state_index(phone, position % kNumStatesPerPhone);
        }

        // Phone-level segments, for triphone context and for reporting.
        std::vector<AlignedSegment> segments;
        int current = chain_position_[0] / kNumStatesPerPhone;
        int start = 0;
        for (int t = 1; t < T; ++t) {
            const int phone_index = chain_position_[t] / kNumStatesPerPhone;
            if (phone_index != current) {
                segments.push_back({phones[current], start, t});
                current = phone_index;
                start = t;
            }
        }
        segments.push_back({phones[current], start, T});
        return segments;
    }

    // Per-frame HMM state indices from the last align() call, in the model's
    // global state numbering - what GMM re-estimation consumes.
    const std::vector<int>& state_path() const { return state_path_; }
    const std::vector<int>& phone_of_frame() const { return phone_of_frame_; }

private:
    std::vector<int> chain_position_, state_path_, phone_of_frame_;
    std::vector<float> scores_, next_, am_;
    std::vector<uint8_t> back_;
    std::vector<int> state_phone_, state_hmm_;
};

}  // namespace models
