// Single-pass token-passing Viterbi beam decoder (Step 3 of models/README.md).
//
// Search space: every pronunciation in the lexicon becomes a linear chain of
// 3-state phone HMMs. Tokens propagate through those chains frame by frame;
// when a token leaves a word's last state it is re-entered at the start of the
// next word with a bigram LM cost. Backtracking runs over word-link records,
// so the decoder emits a word sequence, not a frame-level phone string - which
// is what makes continuous captioning possible at all.
//
// The three costs that make this tractable ("optimality and speed"):
//   1. Acoustic scores are computed once per frame for all 120 HMM states into
//      a flat table; token propagation then reads that table, so widening the
//      beam never re-runs a Gaussian.
//   2. Beam + histogram pruning bound the active state set per frame.
//   3. LM expansion uses the backoff structure: explicit bigrams are scanned
//      only for the top-K surviving word ends, and the backed-off mass is
//      applied to the whole vocabulary in one pass via a single best-backoff
//      value. That makes exact-under-backoff entry scores cost
//      O(K * successors + V) instead of O(K * V).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "acoustic_model.hpp"
#include "lexicon.hpp"
#include "mfcc.hpp"
#include "ngram_lm.hpp"
#include "phone_set.hpp"

namespace models {

struct DecoderConfig {
    float acoustic_scale = 0.06f;   // Gaussian log-likelihoods are sharp; scale them
    float word_insertion_penalty = 0.0f;
    float beam = 120.0f;            // log-prob width below the frame's best token
    float word_beam = 10.0f;        // tighter width for *entering* a new word
    int max_active = 4000;          // histogram pruning cap
    int max_word_ends = 24;         // word ends expanded per frame
    bool allow_silence = true;      // optional inter-word silence model
};

struct DecodeResult {
    std::vector<std::string> words;
    float score = 0.0f;
    int frames = 0;
    // search statistics, for the speed report
    int64_t states_visited = 0;
    int64_t word_entries = 0;
    double decode_seconds = 0.0;
};

class ViterbiDecoder {
public:
    // Builds the search network. Words are addressed by LM id, so the lexicon
    // and the LM must agree on spelling; words missing from either side are
    // dropped (and counted) rather than silently mismatched.
    void build(const Lexicon& lexicon, const NgramLm& lm, const AcousticModel& model,
               const DecoderConfig& config = {}) {
        lm_ = &lm;
        model_ = &model;
        config_ = config;

        state_phone_.clear();
        state_hmm_.clear();
        state_word_.clear();
        state_last_.clear();
        word_entry_.assign(lm.vocabulary_size(), std::vector<int>());
        skipped_words_ = 0;

        for (const Pronunciation& pron : lexicon.pronunciations()) {
            const std::string& spelling = lexicon.word(pron.word_id);
            const int lm_word = lm.word_id(spelling);
            if (lm_word < 0 || lm_word == lm.bos() || lm_word == lm.unk()) {
                ++skipped_words_;
                continue;
            }
            add_chain(pron.phones, lm_word);
        }

        // Optional silence chain: entered like a word, but leaves the LM
        // history untouched so "cat <sil> sat" still scores P(sat|cat).
        silence_entry_ = -1;
        if (config_.allow_silence) {
            silence_entry_ = static_cast<int>(state_phone_.size());
            add_chain({phone_id("SIL")}, kSilenceWord);
        }

        // Words ordered by unigram probability. The backed-off entry score is
        // best_backoff + log P_uni(w), which is monotonically decreasing in
        // this order, so the vocabulary sweep can stop as soon as it drops
        // below the beam floor instead of touching all 20k words every frame.
        unigram_order_.clear();
        for (int word = 0; word < lm.vocabulary_size(); ++word) {
            if (!word_entry_[word].empty()) unigram_order_.push_back(word);
        }
        std::sort(unigram_order_.begin(), unigram_order_.end(), [&lm](int a, int b) {
            return lm.log_unigram(a) > lm.log_unigram(b);
        });

        const int total = static_cast<int>(state_phone_.size());
        scores_.assign(total, kNegInf);
        next_scores_.assign(total, kNegInf);
        links_.assign(total, -1);
        next_links_.assign(total, -1);
        stamp_.assign(total, -1);
        entry_score_.assign(lm.vocabulary_size(), kNegInf);
        entry_link_.assign(lm.vocabulary_size(), -1);
        entry_mark_.assign(lm.vocabulary_size(), -1);
    }

    int num_states() const { return static_cast<int>(state_phone_.size()); }
    int skipped_words() const { return skipped_words_; }

    // Decode one utterance.
    DecodeResult decode(const FeatureMatrix& features) {
        DecodeResult result;
        result.frames = features.num_frames;
        if (!lm_ || !model_ || features.empty()) return result;

        const int total = num_states();
        std::fill(scores_.begin(), scores_.end(), kNegInf);
        std::fill(links_.begin(), links_.end(), -1);
        std::fill(stamp_.begin(), stamp_.end(), -1);
        active_.clear();
        word_links_.clear();

        // Frame -1: a virtual sentence-start word end seeds the first entries.
        ends_.clear();
        ends_.push_back({lm_->bos(), 0.0f, -1});
        previous_best_ = 0.0f;
        std::fill(entry_mark_.begin(), entry_mark_.end(), -1);

        for (int t = 0; t < features.num_frames; ++t) {
            const float* frame = features.frame(t);
            model_->score_frame(frame, am_scores_);

            next_active_.clear();
            ++frame_stamp_;

            // 1. Word entries from the previous frame's word ends.
            expand_entries(t, result);

            // 2. Transitions inside the active chains.
            for (int state : active_) {
                const float score = scores_[state];
                if (score == kNegInf) continue;
                const int phone = state_phone_[state];

                relax(state, score + model_->log_self_loop(phone), links_[state]);
                if (!state_last_[state]) {
                    relax(state + 1, score + model_->log_exit(phone), links_[state]);
                }
            }

            // 3. Emission + pruning.
            float best = kNegInf;
            for (int state : next_active_) {
                next_scores_[state] += config_.acoustic_scale * am_scores_[state_hmm_[state]];
                if (next_scores_[state] > best) best = next_scores_[state];
            }
            previous_best_ = best;
            result.states_visited += static_cast<int64_t>(next_active_.size());
            prune(best);

            // 4. Collect word ends for the next frame's entries.
            collect_word_ends(t, best);

            scores_.swap(next_scores_);
            links_.swap(next_links_);
            active_.swap(next_active_);
            for (int state : next_active_) next_scores_[state] = kNegInf;
        }

        finalize(result);
        return result;
    }

    void set_config(const DecoderConfig& config) { config_ = config; }
    const DecoderConfig& config() const { return config_; }

private:
    static constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    static constexpr int kSilenceWord = -2;

    struct WordEnd {
        int word;    // LM id used as the history for the next word
        float score;
        int link;    // word-link record of the predecessor
        bool emit = true;  // false for silence: it ends no word, so it writes
                           // no link record and contributes no transcript token
    };
    struct WordLink {
        int32_t word;
        int32_t frame;
        int32_t parent;
    };

    void add_chain(const std::vector<int>& phones, int word) {
        const int first = static_cast<int>(state_phone_.size());
        for (size_t p = 0; p < phones.size(); ++p) {
            for (int s = 0; s < kNumStatesPerPhone; ++s) {
                state_phone_.push_back(phones[p]);
                state_hmm_.push_back(state_index(phones[p], s));
                state_word_.push_back(word);
                state_last_.push_back(false);
            }
        }
        state_last_.back() = true;
        if (word >= 0) word_entry_[word].push_back(first);
    }

    // Writes into the next frame's arrays, keeping only the best predecessor.
    inline void relax(int state, float score, int link) {
        if (stamp_[state] != frame_stamp_) {
            stamp_[state] = frame_stamp_;
            next_scores_[state] = score;
            next_links_[state] = link;
            next_active_.push_back(state);
        } else if (score > next_scores_[state]) {
            next_scores_[state] = score;
            next_links_[state] = link;
        }
    }

    // Bigram expansion over the surviving word ends. Explicit bigrams are
    // enumerated per end; the backed-off mass is folded into one best value
    // that is applied to every word in a single vocabulary sweep.
    void expand_entries(int frame, DecodeResult& result) {
        if (ends_.empty()) return;

        // Pass 1: explicit bigram successors of the surviving word ends, plus
        // the single best backed-off predecessor.
        touched_.clear();
        float best_backoff = kNegInf;
        int best_backoff_link = -1;
        for (const WordEnd& end : ends_) {
            const float backoff = end.score + lm_->log_backoff(end.word);
            if (backoff > best_backoff) {
                best_backoff = backoff;
                best_backoff_link = end.link;
            }
            for (const NgramLm::Bigram* it = lm_->successors_begin(end.word);
                 it != lm_->successors_end(end.word); ++it) {
                const int word = it->word;
                if (word_entry_[word].empty()) continue;
                const float candidate = end.score + it->log_prob;
                if (entry_mark_[word] != frame_stamp_) {
                    entry_mark_[word] = frame_stamp_;
                    entry_score_[word] = candidate;
                    entry_link_[word] = end.link;
                    touched_.push_back(word);
                } else if (candidate > entry_score_[word]) {
                    entry_score_[word] = candidate;
                    entry_link_[word] = end.link;
                }
            }
        }

        const float penalty = config_.word_insertion_penalty;

        // Pass 2: enter every word that an explicit bigram reached, taking the
        // better of the bigram and the backed-off score.
        for (int word : touched_) {
            float score = entry_score_[word];
            int link = entry_link_[word];
            const float backed_off = best_backoff + lm_->log_unigram(word);
            if (backed_off > score) {
                score = backed_off;
                link = best_backoff_link;
            }
            score += penalty;
            for (int state : word_entry_[word]) relax(state, score, link);
            ++result.word_entries;
        }

        // Pass 3: the rest of the vocabulary can only be entered through the
        // backoff path, whose score decreases monotonically in unigram order -
        // so stop as soon as it falls below the beam floor instead of sweeping
        // all 20k words on every frame.
        const float floor = previous_best_ - config_.word_beam;
        for (int word : unigram_order_) {
            const float score = best_backoff + lm_->log_unigram(word) + penalty;
            if (score < floor) break;
            if (entry_mark_[word] == frame_stamp_) continue;  // entered in pass 2
            for (int state : word_entry_[word]) relax(state, score, best_backoff_link);
            ++result.word_entries;
        }

        // Silence inherits the best word end without an LM cost.
        if (silence_entry_ >= 0) {
            const WordEnd* best = &ends_[0];
            for (const WordEnd& end : ends_) {
                if (end.score > best->score) best = &end;
            }
            relax(silence_entry_, best->score, best->link);
        }
    }

    void prune(float best) {
        if (next_active_.empty()) return;
        const float floor = best - config_.beam;

        size_t kept = 0;
        for (size_t i = 0; i < next_active_.size(); ++i) {
            const int state = next_active_[i];
            if (next_scores_[state] >= floor) {
                next_active_[kept++] = state;
            } else {
                next_scores_[state] = kNegInf;
                stamp_[state] = -1;
            }
        }
        next_active_.resize(kept);

        // Histogram pruning: cap the active set by score rank.
        if (static_cast<int>(next_active_.size()) > config_.max_active) {
            std::nth_element(next_active_.begin(),
                             next_active_.begin() + config_.max_active,
                             next_active_.end(),
                             [this](int a, int b) { return next_scores_[a] > next_scores_[b]; });
            for (size_t i = config_.max_active; i < next_active_.size(); ++i) {
                next_scores_[next_active_[i]] = kNegInf;
                stamp_[next_active_[i]] = -1;
            }
            next_active_.resize(config_.max_active);
        }
    }

    void collect_word_ends(int frame, float best) {
        ends_.clear();
        candidates_.clear();

        for (int state : next_active_) {
            if (!state_last_[state]) continue;
            const int word = state_word_[state];
            const float exit = next_scores_[state] + model_->log_exit(state_phone_[state]);
            if (exit < best - config_.beam) continue;

            if (word == kSilenceWord) {
                // A pause ends no word, so it emits no link record and no
                // transcript token - but it must still re-enter the word-entry
                // pool, otherwise silence is a trap that swallows the token.
                // Its LM history is <unk>, which has no bigram successors, so
                // the next word is scored through the backoff (unigram) path -
                // the right behaviour after a pause of unknown context.
                candidates_.push_back({lm_->unk(), exit, next_links_[state], false});
                continue;
            }
            candidates_.push_back({word, exit, next_links_[state], true});
        }

        if (candidates_.empty()) return;

        // Keep only the best token per word, then the top-K words overall.
        std::sort(candidates_.begin(), candidates_.end(),
                  [](const WordEnd& a, const WordEnd& b) {
                      return a.word != b.word ? a.word < b.word : a.score > b.score;
                  });
        int previous = -1;
        for (const WordEnd& candidate : candidates_) {
            if (candidate.word == previous) continue;
            previous = candidate.word;
            ends_.push_back(candidate);
        }
        if (static_cast<int>(ends_.size()) > config_.max_word_ends) {
            std::nth_element(ends_.begin(), ends_.begin() + config_.max_word_ends, ends_.end(),
                             [](const WordEnd& a, const WordEnd& b) { return a.score > b.score; });
            ends_.resize(config_.max_word_ends);
        }

        // Record each surviving word end so the backtrace can recover the
        // transcript. Silence exits keep their predecessor's link instead.
        for (WordEnd& end : ends_) {
            if (!end.emit) continue;
            word_links_.push_back({static_cast<int32_t>(end.word), static_cast<int32_t>(frame),
                                   static_cast<int32_t>(end.link)});
            end.link = static_cast<int>(word_links_.size()) - 1;
        }
    }

    void finalize(DecodeResult& result) {
        float best = kNegInf;
        int best_link = -1;
        for (const WordEnd& end : ends_) {
            const float score = end.score + lm_->log_probability(end.word, lm_->eos());
            if (score > best) {
                best = score;
                best_link = end.link;
            }
        }
        if (best_link < 0) {
            // No word finished inside the beam; fall back to the best partial
            // hypothesis so the caller still gets a (short) transcript.
            for (int state : active_) {
                if (scores_[state] > best) {
                    best = scores_[state];
                    best_link = links_[state];
                }
            }
        }
        result.score = best;

        std::vector<std::string> reversed;
        int link = best_link;
        while (link >= 0 && link < static_cast<int>(word_links_.size())) {
            const WordLink& record = word_links_[link];
            reversed.push_back(lm_->word(record.word));
            link = record.parent;
        }
        result.words.assign(reversed.rbegin(), reversed.rend());
    }

    // network
    std::vector<int> state_phone_;
    std::vector<int> state_hmm_;
    std::vector<int> state_word_;
    std::vector<char> state_last_;
    std::vector<std::vector<int>> word_entry_;
    int silence_entry_ = -1;
    int skipped_words_ = 0;

    // per-utterance search state
    std::vector<float> scores_, next_scores_;
    std::vector<int> links_, next_links_;
    std::vector<int> stamp_;
    std::vector<int> active_, next_active_;
    std::vector<float> am_scores_;
    std::vector<float> entry_score_;
    std::vector<int> entry_link_;
    std::vector<int> entry_mark_;
    std::vector<int> touched_;
    std::vector<int> unigram_order_;
    float previous_best_ = -std::numeric_limits<float>::infinity();
    std::vector<WordEnd> ends_, candidates_;
    std::vector<WordLink> word_links_;
    int frame_stamp_ = 0;

    const NgramLm* lm_ = nullptr;
    const AcousticModel* model_ = nullptr;
    DecoderConfig config_;
};

}  // namespace models
