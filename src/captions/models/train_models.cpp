// Trains the networkless captioning stack described in models/README.md.
//
// Two acoustic training routes:
//
//   --train-source timit        bootstrap from TIMIT's hand-marked .phn
//                               boundaries (uniform split inside each phone)
//   --train-source librispeech  matched-domain training on train-clean-100:
//                               no boundaries exist, so a TIMIT-bootstrapped
//                               model forced-aligns the audio against the known
//                               word sequence, the GMMs are re-estimated from
//                               that alignment, and the loop repeats
//
// The second route exists because training on TIMIT and testing on LibriSpeech
// pays a channel-mismatch penalty (1986 close-mic studio vs modern audiobook
// recordings) that no decoder tuning can recover.
//
// Also trains: a word bigram (LibriSpeech train-clean-100 transcripts, disjoint
// from test-clean) and a phone bigram (from the TIMIT alignments).
//
// Usage:
//   train_models [--data-root data] [--out data/models] [--mixtures 8]
//                [--train-source timit|librispeech] [--align-iters 4]
//                [--holdout 0.15] [--vocab 20000]

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "audio_loadnorm.hpp"

#include <toy_pruned_hmm/acoustic_model.hpp>
#include <toy_pruned_hmm/forced_align.hpp>
#include <toy_pruned_hmm/phonemes/lexicon.hpp>
#include <filter/mfcc.hpp>
#include <toy_pruned_hmm/ngram/ngram_lm.hpp>
#include <toy_pruned_hmm/phonemes/phone_set.hpp>
#include <benchmarks/scoring.hpp>
#include <toy_pruned_hmm/triphone/triphone.hpp>

namespace fs = std::filesystem;

namespace {

struct Segment {
    int start_sample = 0;
    int stop_sample = 0;
    int phone = -1;
};

// TIMIT .phn lines are "<start> <stop> <label>" in samples at 16 kHz.
std::vector<Segment> read_alignment(const fs::path& path) {
    std::vector<Segment> segments;
    std::ifstream file(path);
    if (!file) return segments;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        Segment segment;
        std::string label;
        if (!(stream >> segment.start_sample >> segment.stop_sample >> label)) continue;
        segment.phone = models::timit_phone_id(label);
        if (segment.phone >= 0) segments.push_back(segment);
    }
    return segments;
}

std::string read_file(const fs::path& path) {
    std::ifstream file(path);
    if (!file) return {};
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::string json_string_field(const std::string& content, const std::string& key) {
    const size_t at = content.find("\"" + key + "\"");
    if (at == std::string::npos) return {};
    const size_t colon = content.find(':', at);
    if (colon == std::string::npos) return {};
    const size_t first = content.find('"', colon);
    if (first == std::string::npos) return {};
    const size_t last = content.find('"', first + 1);
    if (last == std::string::npos) return {};
    return content.substr(first + 1, last - first - 1);
}

std::string argument(int argc, char** argv, const std::string& flag,
                     const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return fallback;
}

// A corpus held in memory: all frames in one flat array, plus per-utterance
// offsets and the phone sequence each utterance should align to.
struct FeatureStore {
    std::vector<float> frames;          // num_frames x kFeatureDim, row-major
    std::vector<int> offsets;           // utterance i occupies [offsets[i], offsets[i+1])
    std::vector<std::vector<std::pair<int, int>>> contexts;
    std::vector<std::vector<int>> phones;  // expected phone sequence per utterance

    int utterances() const { return static_cast<int>(offsets.size()) - 1; }
    int total_frames() const { return offsets.empty() ? 0 : offsets.back(); }

    models::FeatureMatrix view(int index) const {
        models::FeatureMatrix matrix;
        const int start = offsets[index];
        const int stop = offsets[index + 1];
        matrix.num_frames = stop - start;
        matrix.dim = models::kFeatureDim;
        matrix.data.assign(frames.begin() + static_cast<size_t>(start) * models::kFeatureDim,
                           frames.begin() + static_cast<size_t>(stop) * models::kFeatureDim);
        return matrix;
    }

    void append(const models::FeatureMatrix& features, std::vector<int> phone_sequence,
                std::vector<std::pair<int, int>> phone_contexts) {
        if (offsets.empty()) offsets.push_back(0);
        frames.insert(frames.end(), features.data.begin(), features.data.end());
        offsets.push_back(offsets.back() + features.num_frames);
        phones.push_back(std::move(phone_sequence));
        contexts.push_back(std::move(phone_contexts));
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    const fs::path data_root = argument(argc, argv, "--data-root", "data");
    const fs::path out_dir = argument(argc, argv, "--out", "data/models");
    const double holdout = std::stod(argument(argc, argv, "--holdout", "0.15"));
    const int mixtures = std::stoi(argument(argc, argv, "--mixtures", "8"));
    const int align_iterations = std::stoi(argument(argc, argv, "--align-iters", "4"));
    const std::string train_source = argument(argc, argv, "--train-source", "auto");
    // --lm-only retrains just the language model (e.g. at a different
    // vocabulary size) without touching the acoustic model.
    const bool lm_only = !!std::count(argv, argv + argc, std::string("--lm-only"));
    // Triphone state tying (Path 3): off with --monophone, so the monophone
    // baseline stays reproducible for comparison.
    const bool triphones = !std::count(argv, argv + argc, std::string("--monophone"));
    const int max_senones = std::stoi(argument(argc, argv, "--senones", "2200"));
    const double min_gain = std::stod(argument(argc, argv, "--split-gain", "1200"));
    const double min_occupancy = std::stod(argument(argc, argv, "--min-occupancy", "600"));
    const int triphone_align_iters = std::stoi(argument(argc, argv, "--triphone-iters", "2"));

    fs::create_directories(out_dir);

    std::cout << "======================================================================\n";
    std::cout << "TRAINING NETWORKLESS CAPTIONING MODELS\n";
    std::cout << "======================================================================\n";

    if (lm_only) {
        models::NgramLm lm;
        const int limit = std::stoi(argument(argc, argv, "--vocab", "20000"));
        if (!lm.train_from_text((out_dir / "lm_corpus.txt").string(), limit)) {
            std::cerr << "[ERROR] LM training failed\n";
            return 1;
        }
        lm.save((out_dir / "bigram.lm").string());
        std::cout << "\nWord bigram LM only: " << lm.vocabulary_size() << " words -> "
                  << (out_dir / "bigram.lm") << "\n";
        return 0;
    }

    // ---------------------------------------------------------------- TIMIT
    const fs::path timit_dir = data_root / "audio" / "timit";
    std::vector<fs::path> timit_wavs;
    if (fs::exists(timit_dir)) {
        for (const auto& entry : fs::directory_iterator(timit_dir)) {
            if (entry.path().extension() == ".wav") timit_wavs.push_back(entry.path());
        }
    }
    std::sort(timit_wavs.begin(), timit_wavs.end());

    if (timit_wavs.empty()) {
        std::cerr << "[ERROR] No TIMIT audio in " << timit_dir << "\n";
        std::cerr << "        Run: python scripts/download_datasets.py --tiers timit\n";
        return 1;
    }

    // Speaker-disjoint split: a held-out speaker's utterances never train the
    // model that will later be scored on them.
    std::vector<std::string> speakers;
    for (const fs::path& wav : timit_wavs) {
        speakers.push_back(json_string_field(read_file(fs::path(wav).replace_extension(".json")),
                                             "speaker_id"));
    }
    std::unordered_set<std::string> unique(speakers.begin(), speakers.end());
    std::vector<std::string> ordered(unique.begin(), unique.end());
    std::sort(ordered.begin(), ordered.end());
    std::mt19937 rng(20240910);
    std::shuffle(ordered.begin(), ordered.end(), rng);
    const size_t held = static_cast<size_t>(ordered.size() * holdout);
    std::unordered_set<std::string> test_speakers(ordered.begin(), ordered.begin() + held);

    {
        std::ofstream split(out_dir / "timit_test_speakers.txt");
        for (const std::string& speaker : test_speakers) split << speaker << "\n";
    }

    std::cout << "\nTIMIT bootstrap: " << timit_wavs.size() << " utterances, " << unique.size()
              << " speakers (" << test_speakers.size() << " held out)\n";

    models::MfccExtractor extractor;
    models::AcousticModel acoustic;
    acoustic.begin_training();

    std::vector<std::vector<std::string>> phone_sentences;
    std::vector<float> timit_frames;
    std::vector<int> timit_states;
    models::RtfTimer front_end_timer;
    int trained = 0;
    int64_t frames = 0;

    for (size_t i = 0; i < timit_wavs.size(); ++i) {
        if (test_speakers.count(speakers[i])) continue;

        std::vector<float> audio;
        if (!load_and_preprocess_audio(timit_wavs[i].string(), audio) || audio.empty()) continue;

        const std::vector<Segment> segments =
            read_alignment(fs::path(timit_wavs[i]).replace_extension(".phn"));
        if (segments.empty()) continue;

        front_end_timer.start();
        const models::FeatureMatrix features = extractor.extract(audio);
        front_end_timer.stop();
        front_end_timer.add_audio(static_cast<double>(audio.size()) / models::kSampleRate);
        if (features.empty()) continue;

        std::vector<std::string> phone_sequence;
        for (const Segment& segment : segments) {
            const int start = segment.start_sample / models::kHopSize;
            const int stop = std::min(segment.stop_sample / models::kHopSize, features.num_frames);
            if (stop <= start) continue;
            acoustic.accumulate_segment(features, start, stop, segment.phone);
            const int length = stop - start;
            for (int t = start; t < stop; ++t) {
                int sub = ((t - start) * models::kNumStatesPerPhone) / length;
                if (sub >= models::kNumStatesPerPhone) sub = models::kNumStatesPerPhone - 1;
                timit_states.push_back(models::state_index(segment.phone, sub));
                timit_frames.insert(timit_frames.end(), features.frame(t),
                                    features.frame(t) + models::kFeatureDim);
            }
            frames += length;
            phone_sequence.push_back(models::phone_names()[segment.phone]);
        }
        if (!phone_sequence.empty()) phone_sentences.push_back(std::move(phone_sequence));
        ++trained;
    }

    acoustic.finish_training();
    std::cout << "  bootstrap model: " << trained << " utterances, " << frames << " frames ("
              << frames / 6000.0 << " min)\n";

    // Phone bigram over the TIMIT alignments, used by the Tier 2 phone loop.
    {
        const fs::path phone_text = out_dir / "phone_corpus.txt";
        std::ofstream out(phone_text);
        for (const std::vector<std::string>& sentence : phone_sentences) {
            for (size_t i = 0; i < sentence.size(); ++i) {
                out << sentence[i] << (i + 1 == sentence.size() ? '\n' : ' ');
            }
        }
        out.close();

        models::NgramLm phone_lm;
        if (phone_lm.train_from_text(phone_text.string(), 100)) {
            phone_lm.save((out_dir / "phone.lm").string());
            std::cout << "  wrote " << (out_dir / "phone.lm") << " (" << phone_lm.vocabulary_size()
                      << " units)\n";
        }
    }

    // ------------------------------------------- matched-domain re-training
    const fs::path libri_dir = data_root / "audio" / "librispeech_train";
    const bool want_librispeech =
        train_source == "librispeech" || (train_source == "auto" && fs::exists(libri_dir));

    std::vector<float>* mixture_frames = &timit_frames;
    std::vector<int>* mixture_states = &timit_states;
    std::vector<int> aligned_states;
    FeatureStore store;
    models::TriphoneTree tree;

    if (want_librispeech && fs::exists(libri_dir)) {
        // Lexicon for alignment: every word in the training transcripts.
        models::Lexicon lexicon;
        if (!lexicon.load_cmudict((out_dir / "cmudict.dict").string())) {
            std::cerr << "[ERROR] need " << (out_dir / "cmudict.dict") << " to align\n";
            return 1;
        }
        models::PronunciationTable pronunciations;
        pronunciations.build(lexicon);

        std::vector<fs::path> wavs;
        for (const auto& entry : fs::directory_iterator(libri_dir)) {
            if (entry.path().extension() == ".wav") wavs.push_back(entry.path());
        }
        std::sort(wavs.begin(), wavs.end());

        std::cout << "\nMatched-domain training: " << wavs.size() << " LibriSpeech utterances\n";
        std::cout << "  lexicon: " << pronunciations.size() << " words\n";

        int skipped_oov = 0;
        int guessed_words = 0;
        for (const fs::path& wav : wavs) {
            const std::vector<std::string> words =
                models::tokenize(read_file(fs::path(wav).replace_extension(".txt")));
            if (words.empty()) continue;

            std::vector<int> phone_sequence;
            std::vector<std::pair<int, int>> phone_contexts;
            if (!pronunciations.phones_for(words, phone_sequence, &guessed_words, false, &phone_contexts)) {
                ++skipped_oov;  // no pronunciation at all, even by rule
                continue;
            }

            std::vector<float> audio;
            if (!load_and_preprocess_audio(wav.string(), audio) || audio.empty()) continue;

            front_end_timer.start();
            const models::FeatureMatrix features = extractor.extract(audio);
            front_end_timer.stop();
            front_end_timer.add_audio(static_cast<double>(audio.size()) / models::kSampleRate);
            if (features.empty()) continue;

            store.append(features, std::move(phone_sequence), std::move(phone_contexts));
        }

        std::cout << "  loaded " << store.utterances() << " utterances, " << store.total_frames()
                  << " frames (" << store.total_frames() / 6000.0 << " min), " << skipped_oov
                  << " skipped, " << guessed_words
                  << " words pronounced by letter-to-sound fallback\n";
        std::cout << "  feature cache: " << store.frames.size() * sizeof(float) / (1024 * 1024)
                  << " MB\n";

        // Forced-alignment loop: align with the current model, re-estimate the
        // model from that alignment, repeat. The TIMIT model is only a
        // bootstrap - after the first iteration every parameter has been
        // re-estimated on LibriSpeech audio.
        models::ForcedAligner aligner;
        for (int iteration = 1; iteration <= align_iterations; ++iteration) {
            models::RtfTimer timer;
            timer.start();

            models::AcousticModel next;
            next.begin_training();
            aligned_states.clear();
            aligned_states.reserve(store.total_frames());

            int aligned = 0;
            int64_t aligned_frames = 0;
            for (int u = 0; u < store.utterances(); ++u) {
                const models::FeatureMatrix features = store.view(u);
                const std::vector<models::AlignedSegment> segments =
                    aligner.align(features, store.phones[u], acoustic);
                if (segments.empty()) {
                    // No complete path: keep the frames out of this iteration.
                    aligned_states.resize(aligned_states.size() + features.num_frames, -1);
                    continue;
                }
                const std::vector<int>& path = aligner.state_path();
                for (int t = 0; t < features.num_frames; ++t) {
                    next.accumulate_frame(features.frame(t), path[t]);
                    aligned_states.push_back(path[t]);
                }
                for (const models::AlignedSegment& segment : segments) {
                    next.accumulate_duration(segment.phone,
                                             segment.stop_frame - segment.start_frame);
                }
                ++aligned;
                aligned_frames += features.num_frames;
            }

            next.finish_training();
            acoustic = next;
            const double seconds = timer.stop();
            std::cout << "  align pass " << iteration << "/" << align_iterations << ": " << aligned
                      << "/" << store.utterances() << " utterances, " << aligned_frames
                      << " frames in " << seconds << " s\n";
        }

        mixture_frames = &store.frames;
        mixture_states = &aligned_states;

        // ------------------------------------------ tied-state triphones
        if (triphones) {
            models::RtfTimer timer;
            timer.start();

            // Collect per-context statistics from the final monophone
            // alignment: one entry per (centre, state) x (left, right).
            std::vector<std::vector<models::ContextStats>> contexts(
                models::TriphoneTree::num_tree_keys());
            std::vector<std::unordered_map<int, int>> lookup(contexts.size());
            const int silence = models::phone_id("SIL");

            std::vector<int> senone_of_frame(store.total_frames(), -1);
            std::vector<std::array<int, 4>> frame_context(store.total_frames(),
                                                          {{-1, -1, -1, -1}});
            // (phone, frames) for every aligned segment, replayed into the
            // triphone model so transitions do not have to be re-derived.
            std::vector<std::pair<int, int>> durations;

            for (int u = 0; u < store.utterances(); ++u) {
                const models::FeatureMatrix features = store.view(u);
                const std::vector<models::AlignedSegment> segments =
                    aligner.align(features, store.phones[u], acoustic);
                if (segments.empty()) continue;
                const std::vector<int>& path = aligner.state_path();
                const int base = store.offsets[u];

                for (size_t s = 0; s < segments.size(); ++s) {
                    const int centre = segments[s].phone;
                    const int left = store.contexts[u][s].first;
                    const int right = store.contexts[u][s].second;

                    for (int t = segments[s].start_frame; t < segments[s].stop_frame; ++t) {
                        const int sub = path[t] % models::kNumStatesPerPhone;
                        const int key = models::TriphoneTree::tree_key(centre, sub);
                        const int context_id = left * 64 + right;

                        auto it = lookup[key].find(context_id);
                        if (it == lookup[key].end()) {
                            it = lookup[key].emplace(context_id,
                                                     static_cast<int>(contexts[key].size())).first;
                            models::ContextStats stats;
                            stats.left = left;
                            stats.right = right;
                            contexts[key].push_back(std::move(stats));
                        }
                        contexts[key][it->second].add(features.frame(t));
                        frame_context[base + t] = {{left, centre, sub, right}};
                    }
                    durations.emplace_back(centre,
                                           segments[s].stop_frame - segments[s].start_frame);
                }
            }

            int distinct = 0;
            for (const auto& list : contexts) distinct += static_cast<int>(list.size());

            tree.build(contexts, min_gain, min_occupancy, max_senones);
            std::cout << "\n  triphone tying: " << distinct
                      << " observed contexts -> " << tree.num_senones() << " tied states in "
                      << timer.stop() << " s\n";

            // Re-label every frame with its senone and re-estimate from scratch.
            for (int i = 0; i < store.total_frames(); ++i) {
                const std::array<int, 4>& context = frame_context[i];
                if (context[1] < 0) continue;
                senone_of_frame[i] = tree.senone(context[0], context[1], context[2], context[3]);
            }

            models::AcousticModel triphone_model;
            triphone_model.begin_training(tree.num_senones());
            for (int i = 0; i < store.total_frames(); ++i) {
                if (senone_of_frame[i] < 0) continue;
                triphone_model.accumulate_frame(
                    store.frames.data() + static_cast<size_t>(i) * models::kFeatureDim,
                    senone_of_frame[i]);
            }
            for (const std::pair<int, int>& duration : durations) {
                triphone_model.accumulate_duration(duration.first, duration.second);
            }
            triphone_model.finish_training();
            acoustic = triphone_model;

            // One more alignment pass, now scored with the triphone model.
            for (int iteration = 1; iteration <= triphone_align_iters; ++iteration) {
                models::RtfTimer pass;
                pass.start();
                models::AcousticModel next;
                next.begin_training(tree.num_senones());
                std::fill(senone_of_frame.begin(), senone_of_frame.end(), -1);
                int aligned = 0;

                for (int u = 0; u < store.utterances(); ++u) {
                    const models::FeatureMatrix features = store.view(u);
                    const std::vector<models::AlignedSegment> segments =
                        aligner.align(features, store.phones[u], acoustic, &tree, &store.contexts[u]);
                    if (segments.empty()) continue;
                    const int base = store.offsets[u];
                    const std::vector<int>& path = aligner.state_path();

                    for (size_t s = 0; s < segments.size(); ++s) {
                        const int centre = segments[s].phone;
                        const int left = store.contexts[u][s].first;
                        const int right = store.contexts[u][s].second;
                        for (int t = segments[s].start_frame; t < segments[s].stop_frame; ++t) {
                            const int sub = path[t] % models::kNumStatesPerPhone;
                            const int senone = tree.senone(left, centre, sub, right);
                            senone_of_frame[base + t] = senone;
                            next.accumulate_frame(features.frame(t), senone);
                        }
                        next.accumulate_duration(centre,
                                                 segments[s].stop_frame - segments[s].start_frame);
                    }
                    ++aligned;
                }
                next.finish_training();
                acoustic = next;
                std::cout << "  triphone align pass " << iteration << "/" << triphone_align_iters
                          << ": " << aligned << " utterances in " << pass.stop() << " s\n";
            }

            aligned_states = senone_of_frame;
            tree.save((out_dir / "triphone.tree").string());
            std::cout << "  wrote " << (out_dir / "triphone.tree") << "\n";
        }
    }

    // ------------------------------------------------------------ mixtures
    if (mixtures > 1 && !mixture_states->empty()) {
        models::RtfTimer em_timer;
        em_timer.start();
        acoustic.train_mixtures(*mixture_frames, *mixture_states, mixtures);
        const double seconds = em_timer.stop();
        std::cout << "\n  mixture EM: " << mixtures << " components/state over "
                  << mixture_states->size() << " frames in " << seconds << " s\n";
    }

    const fs::path am_path = out_dir / "monophone.am";
    if (!acoustic.save(am_path.string())) {
        std::cerr << "[ERROR] Could not write " << am_path << "\n";
        return 1;
    }
    std::cout << "  wrote " << am_path << " (" << acoustic.units() << " units x "
              << acoustic.mixtures() << " mixtures x " << models::kFeatureDim << " dims)\n";
    std::cout << "  front-end speed: " << front_end_timer.times_real_time() << "x real time\n";

    // ------------------------------------------------------------ word LM
    const fs::path corpus = out_dir / "lm_corpus.txt";
    if (!fs::exists(corpus)) {
        std::cerr << "\n[WARN] " << corpus << " not found; skipping word LM.\n";
        return 0;
    }

    std::cout << "\nWord bigram LM from " << corpus << "\n";
    models::NgramLm lm;
    const int vocabulary_limit = std::stoi(argument(argc, argv, "--vocab", "20000"));
    if (!lm.train_from_text(corpus.string(), vocabulary_limit)) {
        std::cerr << "[ERROR] LM training failed\n";
        return 1;
    }
    lm.save((out_dir / "bigram.lm").string());
    std::cout << "  vocabulary: " << lm.vocabulary_size() << " words\n";
    std::cout << "  wrote " << (out_dir / "bigram.lm") << "\n";

    std::cout << "\n[OK] Models written to " << out_dir << "\n";
    return 0;
}
