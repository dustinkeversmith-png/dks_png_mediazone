// End-to-end evaluation of the networkless captioning stack against the three
// downloaded tiers, alongside the two legacy frame-level baselines.
//
//   Tier 1  data/audio/digits       isolated digits   -> accuracy
//   Tier 2  data/audio/timit        phone recognition -> phone error rate
//   Tier 3  data/audio/librispeech  captioning        -> word error rate
//
// Every tier reports speed as a real-time factor (RTF = compute seconds per
// audio second) measured on one CPU thread, so accuracy and cost are always
// read together.
//
// Usage:
//   benchmark_models [--tiers digits,timit,librispeech] [--models data/models]
//                    [--limit N] [--templates-per-digit K] [--dev]
//                    [--acoustic-scale F] [--beam F] [--max-active N]
//                    [--word-penalty F] [--vocab-lexicon 1]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "audio_loadnorm.hpp"

#include <models/acoustic_model.hpp>
#include <models/baseline_frontends.hpp>
#include <models/dtw.hpp>
#include <models/lexicon.hpp>
#include <models/mfcc.hpp>
#include <models/ngram_lm.hpp>
#include <models/phone_set.hpp>
#include <models/scoring.hpp>
#include <models/triphone.hpp>
#include <models/viterbi_decoder.hpp>

namespace fs = std::filesystem;

namespace {

std::string argument(int argc, char** argv, const std::string& flag,
                     const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return fallback;
}

bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) return true;
    }
    return false;
}

std::vector<std::string> split_csv(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) parts.push_back(item);
    }
    return parts;
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

void print_header(const std::string& title) {
    std::cout << "\n======================================================================\n";
    std::cout << title << "\n";
    std::cout << "======================================================================\n";
}

// ---------------------------------------------------------------- Tier 1

struct DigitClip {
    fs::path path;
    std::string label;
};

void run_digits(const fs::path& data_root, int templates_per_digit, int test_per_digit,
                bool run_baselines, int neighbours, int radius, bool use_endpoint) {
    print_header("TIER 1  Isolated digits  -  MFCC + DTW template matching");

    const fs::path digits_dir = data_root / "audio" / "digits";
    if (!fs::exists(digits_dir)) {
        std::cout << "[SKIP] " << digits_dir << " not found\n";
        return;
    }

    std::vector<DigitClip> train, test;
    for (int digit = 0; digit < 10; ++digit) {
        const fs::path class_dir = digits_dir / std::to_string(digit);
        if (!fs::exists(class_dir)) continue;
        std::vector<fs::path> clips;
        for (const auto& entry : fs::directory_iterator(class_dir)) {
            if (entry.path().extension() == ".wav") clips.push_back(entry.path());
        }
        std::sort(clips.begin(), clips.end());
        // Deterministic shuffle so the split does not track file order.
        std::mt19937 rng(1234 + digit);
        std::shuffle(clips.begin(), clips.end(), rng);

        const std::string label = std::to_string(digit);
        for (size_t i = 0; i < clips.size(); ++i) {
            if (static_cast<int>(i) < templates_per_digit) {
                train.push_back({clips[i], label});
            } else if (static_cast<int>(i) < templates_per_digit + test_per_digit) {
                test.push_back({clips[i], label});
            }
        }
    }

    if (train.empty() || test.empty()) {
        std::cout << "[SKIP] not enough digit clips (train=" << train.size()
                  << ", test=" << test.size() << ")\n";
        return;
    }

    std::cout << "Templates: " << train.size() << "   Test clips: " << test.size() << "\n";
    std::cout << "Note: this corpus carries no speaker ids, so the split is clip-disjoint\n"
              << "      but not speaker-disjoint.\n\n";

    models::MfccExtractor extractor;
    models::DtwRecognizer recognizer;
    models::RtfTimer enroll_timer;

    for (const DigitClip& clip : train) {
        std::vector<float> audio;
        if (!load_and_preprocess_audio(clip.path.string(), audio) || audio.empty()) continue;
        enroll_timer.start();
        models::FeatureMatrix features = extractor.extract(audio);
        if (use_endpoint) features = models::endpoint(features);
        enroll_timer.stop();
        enroll_timer.add_audio(static_cast<double>(audio.size()) / models::kSampleRate);
        if (!features.empty()) recognizer.add_template(clip.label, std::move(features));
    }

    struct Outcome {
        int correct = 0;
        int total = 0;
        models::RtfTimer timer;
    };
    Outcome dtw;
    int64_t scored = 0, pruned = 0, abandoned = 0;
    std::map<std::string, std::map<std::string, int>> confusion;

    for (const DigitClip& clip : test) {
        std::vector<float> audio;
        if (!load_and_preprocess_audio(clip.path.string(), audio) || audio.empty()) continue;
        const double duration = static_cast<double>(audio.size()) / models::kSampleRate;

        dtw.timer.start();
        models::FeatureMatrix features = extractor.extract(audio);
        if (use_endpoint) features = models::endpoint(features);
        const models::DtwRecognizer::Result result =
            recognizer.classify(features, neighbours, radius);
        dtw.timer.stop();
        dtw.timer.add_audio(duration);

        scored += result.templates_scored;
        pruned += result.templates_pruned;
        abandoned += result.templates_abandoned;
        ++dtw.total;
        if (result.label == clip.label) ++dtw.correct;
        confusion[clip.label][result.label]++;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "MFCC + banded DTW (" << neighbours << "-NN, "
              << (use_endpoint ? "endpointed" : "no endpointing") << ", band radius " << radius
              << ")\n";
    std::cout << "  accuracy      : " << 100.0 * dtw.correct / std::max(dtw.total, 1) << " %  ("
              << dtw.correct << "/" << dtw.total << ")\n";
    std::cout << "  speed         : " << dtw.timer.times_real_time() << "x real time  (RTF "
              << dtw.timer.rtf() << ")\n";
    std::cout << "  per clip      : " << 1000.0 * dtw.timer.elapsed() / std::max(dtw.total, 1)
              << " ms against " << recognizer.size() << " templates\n";
    std::cout << "  search savings: lower bound skipped "
              << 100.0 * pruned / std::max<int64_t>(scored + pruned, 1)
              << " % of templates, early abandon dropped "
              << 100.0 * abandoned / std::max<int64_t>(scored + pruned, 1)
              << " % mid-alignment\n";
    std::cout << "  enrolment     : " << enroll_timer.times_real_time() << "x real time\n";

    if (!run_baselines) return;

    // Legacy frame-level pipelines on the same split: their collapsed vowel
    // strings are classified by nearest-neighbour edit distance, which is the
    // most favourable classifier those features admit.
    struct BaselineRun {
        std::string name;
        std::vector<std::vector<std::string>> templates;
        std::vector<std::string> labels;
        int correct = 0;
        int total = 0;
        models::RtfTimer timer;
    };

    models::baseline::LpcFrontend lpc;
    models::baseline::FourierFrontend fourier;
    BaselineRun runs[2] = {{"LPC formants + Bark lookup"}, {"FFT envelope + Bark lookup"}};

    for (const DigitClip& clip : train) {
        std::vector<float> audio;
        if (!load_and_preprocess_audio(clip.path.string(), audio) || audio.empty()) continue;
        runs[0].templates.push_back(models::baseline::collapse(lpc.vowel_string(audio)));
        runs[0].labels.push_back(clip.label);
        runs[1].templates.push_back(models::baseline::collapse(fourier.vowel_string(audio)));
        runs[1].labels.push_back(clip.label);
    }

    for (const DigitClip& clip : test) {
        std::vector<float> audio;
        if (!load_and_preprocess_audio(clip.path.string(), audio) || audio.empty()) continue;
        const double duration = static_cast<double>(audio.size()) / models::kSampleRate;

        for (int which = 0; which < 2; ++which) {
            BaselineRun& run = runs[which];
            run.timer.start();
            const std::vector<std::string> hypothesis =
                models::baseline::collapse(which == 0 ? lpc.vowel_string(audio)
                                                      : fourier.vowel_string(audio));
            int best = 1 << 30;
            std::string best_label;
            for (size_t i = 0; i < run.templates.size(); ++i) {
                const models::EditCounts counts =
                    models::edit_distance(run.templates[i], hypothesis);
                if (static_cast<int>(counts.errors()) < best) {
                    best = static_cast<int>(counts.errors());
                    best_label = run.labels[i];
                }
            }
            run.timer.stop();
            run.timer.add_audio(duration);
            ++run.total;
            if (best_label == clip.label) ++run.correct;
        }
    }

    for (const BaselineRun& run : runs) {
        std::cout << "\n" << run.name << " (legacy frame classifier, 1-NN edit distance)\n";
        std::cout << "  accuracy      : " << 100.0 * run.correct / std::max(run.total, 1) << " %  ("
                  << run.correct << "/" << run.total << ")\n";
        std::cout << "  speed         : " << run.timer.times_real_time() << "x real time  (RTF "
                  << run.timer.rtf() << ")\n";
    }

    std::cout << "\nConfusions (reference -> most frequent hypothesis):\n";
    for (const auto& row : confusion) {
        const auto best = std::max_element(
            row.second.begin(), row.second.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        int total = 0;
        for (const auto& cell : row.second) total += cell.second;
        std::cout << "  " << row.first << " -> " << best->first << "  (" << best->second << "/"
                  << total << ")\n";
    }
}

// ---------------------------------------------------------------- Tier 2

void run_timit(const fs::path& data_root, const fs::path& models_dir, int limit,
               const models::DecoderConfig& base_config) {
    print_header("TIER 2  Phonetic recognition  -  monophone HMM + phone bigram Viterbi");

    models::AcousticModel acoustic;
    if (!acoustic.load((models_dir / "monophone.am").string())) {
        std::cout << "[SKIP] " << (models_dir / "monophone.am") << " not found; run train_models\n";
        return;
    }
    models::NgramLm phone_lm;
    if (!phone_lm.load((models_dir / "phone.lm").string())) {
        std::cout << "[SKIP] " << (models_dir / "phone.lm") << " not found; run train_models\n";
        return;
    }

    // Phone loop: a lexicon whose words are single phones.
    models::Lexicon phone_lexicon;
    for (int p = 0; p < models::num_phones(); ++p) {
        phone_lexicon.add_entry(models::phone_names()[p], {p});
    }

    models::DecoderConfig config = base_config;
    config.allow_silence = false;  // SIL is already a unit in the loop
    models::ViterbiDecoder decoder;
    decoder.build(phone_lexicon, phone_lm, acoustic, config);

    std::unordered_set<std::string> test_speakers;
    {
        std::ifstream file(models_dir / "timit_test_speakers.txt");
        std::string speaker;
        while (file >> speaker) test_speakers.insert(speaker);
    }
    if (test_speakers.empty()) {
        std::cout << "[SKIP] no held-out speaker list; run train_models first\n";
        return;
    }

    const fs::path timit_dir = data_root / "audio" / "timit";
    std::vector<fs::path> wavs;
    for (const auto& entry : fs::directory_iterator(timit_dir)) {
        if (entry.path().extension() == ".wav") wavs.push_back(entry.path());
    }
    std::sort(wavs.begin(), wavs.end());

    models::MfccExtractor extractor;
    models::EditCounts phone_errors;
    models::RtfTimer timer;
    int64_t frames_correct = 0, frames_total = 0;
    int utterances = 0;
    int64_t states_visited = 0;

    for (const fs::path& wav : wavs) {
        const std::string meta = read_file(fs::path(wav).replace_extension(".json"));
        if (!test_speakers.count(json_string_field(meta, "speaker_id"))) continue;
        if (limit > 0 && utterances >= limit) break;

        std::vector<float> audio;
        if (!load_and_preprocess_audio(wav.string(), audio) || audio.empty()) continue;

        // Reference phone sequence, folded to the 39-phone inventory.
        std::vector<std::string> reference;
        std::vector<int> frame_labels;
        {
            std::ifstream alignment(fs::path(wav).replace_extension(".phn"));
            std::string line;
            while (std::getline(alignment, line)) {
                std::istringstream stream(line);
                int start = 0, stop = 0;
                std::string label;
                if (!(stream >> start >> stop >> label)) continue;
                const int phone = models::timit_phone_id(label);
                if (phone < 0) continue;
                const std::string& name = models::phone_names()[phone];
                if (name != "SIL" && (reference.empty() || reference.back() != name)) {
                    reference.push_back(name);
                }
                for (int f = start / models::kHopSize; f < stop / models::kHopSize; ++f) {
                    if (f >= static_cast<int>(frame_labels.size())) frame_labels.resize(f + 1, -1);
                    frame_labels[f] = phone;
                }
            }
        }
        if (reference.empty()) continue;

        timer.start();
        const models::FeatureMatrix features = extractor.extract(audio);
        const models::DecodeResult result = decoder.decode(features);
        timer.stop();
        timer.add_audio(static_cast<double>(audio.size()) / models::kSampleRate);
        states_visited += result.states_visited;

        std::vector<std::string> hypothesis;
        for (const std::string& phone : result.words) {
            if (phone == "SIL" || phone == "<s>" || phone == "</s>" || phone == "<unk>") continue;
            if (hypothesis.empty() || hypothesis.back() != phone) hypothesis.push_back(phone);
        }
        phone_errors += models::edit_distance(reference, hypothesis);

        // Frame-level accuracy of the Gaussians alone (no search), which is the
        // direct analogue of what the legacy vowel classifiers produce.
        std::vector<float> scores;
        for (int t = 0; t < features.num_frames && t < static_cast<int>(frame_labels.size()); ++t) {
            if (frame_labels[t] < 0) continue;
            acoustic.score_frame(features.frame(t), scores);
            int best_state = 0;
            for (int s = 1; s < models::num_states(); ++s) {
                if (scores[s] > scores[best_state]) best_state = s;
            }
            if (best_state / models::kNumStatesPerPhone == frame_labels[t]) ++frames_correct;
            ++frames_total;
        }
        ++utterances;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Held-out speakers: " << test_speakers.size() << "   utterances: " << utterances
              << "\n";
    std::cout << "  phone error rate : " << 100.0 * phone_errors.error_rate() << " %   (S "
              << phone_errors.substitutions << " / D " << phone_errors.deletions << " / I "
              << phone_errors.insertions << " over " << phone_errors.reference_length
              << " phones)\n";
    std::cout << "  frame accuracy   : " << 100.0 * frames_correct / std::max<int64_t>(frames_total, 1)
              << " %   (" << frames_correct << "/" << frames_total << " frames, 40-way)\n";
    std::cout << "  speed            : " << timer.times_real_time() << "x real time  (RTF "
              << timer.rtf() << ")\n";
    std::cout << "  search           : " << states_visited / std::max<int64_t>(utterances, 1)
              << " state visits per utterance\n";
}

// ---------------------------------------------------------------- Tier 3

void run_librispeech(const fs::path& data_root, const fs::path& models_dir, int limit, bool dev_split,
                     const models::DecoderConfig& config, bool restrict_lexicon,
                     bool force_monophone) {
    print_header("TIER 3  Continuous captioning  -  MFCC + monophone HMM + bigram Viterbi");

    models::AcousticModel acoustic;
    if (!acoustic.load((models_dir / "monophone.am").string())) {
        std::cout << "[SKIP] monophone.am not found; run train_models\n";
        return;
    }
    models::NgramLm lm;
    if (!lm.load((models_dir / "bigram.lm").string())) {
        std::cout << "[SKIP] bigram.lm not found; run train_models\n";
        return;
    }

    std::unordered_set<std::string> vocabulary;
    if (restrict_lexicon) {
        for (int w = 0; w < lm.vocabulary_size(); ++w) vocabulary.insert(lm.word(w));
    }

    models::Lexicon lexicon;
    const fs::path dict = models_dir / "cmudict.dict";
    if (!lexicon.load_cmudict(dict.string(), vocabulary)) {
        std::cout << "[SKIP] " << dict << " not found (CMU Pronouncing Dictionary)\n";
        return;
    }

    models::ViterbiDecoder decoder;
    models::RtfTimer build_timer;
    build_timer.start();
    // Use the tied-state triphone tree when one was trained; without it the
    // same code path is a context-independent monophone system.
    models::TriphoneTree tree;
    const bool has_tree = !force_monophone &&
                          tree.load((models_dir / "triphone.tree").string());
    decoder.build(lexicon, lm, acoustic, config, has_tree ? &tree : nullptr);
    const double build_seconds = build_timer.stop();

    std::cout << "Acoustic model: " << acoustic.units() << " units x " << acoustic.mixtures()
              << " mixtures" << (has_tree ? " (tied-state triphones)" : " (monophone)") << "\n";
    std::cout << "Search network: " << decoder.num_states() << " HMM states from "
              << lexicon.pronunciations().size() << " pronunciations / " << lm.vocabulary_size()
              << " LM words (built in " << std::fixed << std::setprecision(2) << build_seconds
              << " s)\n";
    std::cout << "Decoder: acoustic-scale " << config.acoustic_scale << ", beam " << config.beam
              << ", max-active " << config.max_active << ", word-beam " << config.word_beam
              << ", word-penalty " << config.word_insertion_penalty << "\n";

    const fs::path libri_dir = data_root / "audio" / "librispeech";
    std::vector<fs::path> wavs;
    if (fs::exists(libri_dir)) {
        for (const auto& entry : fs::directory_iterator(libri_dir)) {
            if (entry.path().extension() == ".wav") wavs.push_back(entry.path());
        }
    }
    std::sort(wavs.begin(), wavs.end());
    if (wavs.empty()) {
        std::cout << "[SKIP] no LibriSpeech audio in " << libri_dir << "\n";
        return;
    }

    // First 25 % is the development slice used for tuning; the rest is the
    // held-out evaluation set. Never report tuning numbers as test numbers.
    const size_t boundary = wavs.size() / 4;
    std::vector<fs::path> selected;
    if (dev_split) {
        selected.assign(wavs.begin(), wavs.begin() + boundary);
    } else {
        selected.assign(wavs.begin() + boundary, wavs.end());
    }
    if (limit > 0 && static_cast<int>(selected.size()) > limit) selected.resize(limit);

    models::MfccExtractor extractor;
    models::EditCounts word_errors;
    models::RtfTimer timer, feature_timer;
    int64_t oov = 0, reference_words = 0, states_visited = 0, word_entries = 0;
    int utterances = 0;
    std::vector<std::string> sample_reference, sample_hypothesis;

    for (const fs::path& wav : selected) {
        const std::string transcript = read_file(fs::path(wav).replace_extension(".txt"));
        const std::vector<std::string> reference = models::tokenize(transcript);
        if (reference.empty()) continue;

        std::vector<float> audio;
        if (!load_and_preprocess_audio(wav.string(), audio) || audio.empty()) continue;
        const double duration = static_cast<double>(audio.size()) / models::kSampleRate;

        feature_timer.start();
        const models::FeatureMatrix features = extractor.extract(audio);
        feature_timer.stop();
        feature_timer.add_audio(duration);

        timer.start();
        const models::DecodeResult result = decoder.decode(features);
        timer.stop();
        timer.add_audio(duration);

        std::vector<std::string> hypothesis;
        for (const std::string& word : result.words) {
            if (word == "<s>" || word == "</s>" || word == "<unk>") continue;
            hypothesis.push_back(word);
        }

        for (const std::string& word : reference) {
            if (lm.word_id(word) < 0) ++oov;
            ++reference_words;
        }
        word_errors += models::edit_distance(reference, hypothesis);
        states_visited += result.states_visited;
        word_entries += result.word_entries;
        if (utterances == 0) {
            sample_reference = reference;
            sample_hypothesis = hypothesis;
        }
        ++utterances;

        if (utterances % 20 == 0) {
            std::cout << "  " << utterances << "/" << selected.size() << " utterances, WER so far "
                      << 100.0 * word_errors.error_rate() << " %, " << timer.times_real_time()
                      << "x real time\n";
        }
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n" << (dev_split ? "DEV" : "TEST") << " slice: " << utterances << " utterances, "
              << timer.audio() / 60.0 << " min of audio\n";
    std::cout << "  word error rate : " << 100.0 * word_errors.error_rate() << " %   (S "
              << word_errors.substitutions << " / D " << word_errors.deletions << " / I "
              << word_errors.insertions << " over " << word_errors.reference_length
              << " words)\n";
    std::cout << "  OOV rate        : " << 100.0 * oov / std::max<int64_t>(reference_words, 1)
              << " %   (" << oov << " tokens outside the " << lm.vocabulary_size()
              << "-word vocabulary)\n";
    std::cout << "  decode speed    : " << timer.times_real_time() << "x real time  (RTF "
              << timer.rtf() << ")\n";
    std::cout << "  front-end speed : " << feature_timer.times_real_time() << "x real time\n";
    std::cout << "  search          : " << states_visited / std::max<int64_t>(utterances, 1)
              << " state visits and " << word_entries / std::max<int64_t>(utterances, 1)
              << " word entries per utterance\n";

    if (!sample_reference.empty()) {
        std::cout << "\n  sample reference : ";
        for (size_t i = 0; i < sample_reference.size() && i < 14; ++i) {
            std::cout << sample_reference[i] << ' ';
        }
        std::cout << "\n  sample hypothesis: ";
        for (size_t i = 0; i < sample_hypothesis.size() && i < 14; ++i) {
            std::cout << sample_hypothesis[i] << ' ';
        }
        std::cout << "\n";
    }

    std::cout << "\n  Legacy baselines on this tier: the LPC and Fourier programs emit a frame\n"
              << "  level vowel string (IY EH AA ...), not words. With no lexicon or LM they\n"
              << "  cannot produce a transcript at all, so their WER is 100 % by construction.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    const fs::path data_root = argument(argc, argv, "--data-root", "data");
    const fs::path models_dir = argument(argc, argv, "--models", "data/models");
    const int limit = std::stoi(argument(argc, argv, "--limit", "0"));
    const int templates_per_digit = std::stoi(argument(argc, argv, "--templates-per-digit", "20"));
    const int test_per_digit = std::stoi(argument(argc, argv, "--test-per-digit", "50"));
    const bool dev_split = has_flag(argc, argv, "--dev");
    const bool skip_baselines = has_flag(argc, argv, "--no-baselines");
    const int neighbours = std::stoi(argument(argc, argv, "--knn", "1"));
    const int dtw_radius = std::stoi(argument(argc, argv, "--dtw-radius", "12"));

    models::DecoderConfig config;
    config.acoustic_scale = std::stof(argument(argc, argv, "--acoustic-scale", "0.06"));
    config.beam = std::stof(argument(argc, argv, "--beam", "120"));
    config.word_beam = std::stof(argument(argc, argv, "--word-beam", "10"));
    config.max_active = std::stoi(argument(argc, argv, "--max-active", "4000"));
    config.max_word_ends = std::stoi(argument(argc, argv, "--max-word-ends", "24"));
    config.word_insertion_penalty = std::stof(argument(argc, argv, "--word-penalty", "0"));

    const std::vector<std::string> tiers =
        split_csv(argument(argc, argv, "--tiers", "digits,timit,librispeech"));

    std::cout << "======================================================================\n";
    std::cout << "NETWORKLESS CAPTIONING BENCHMARK\n";
    std::cout << "======================================================================\n";
    std::cout << "Data:   " << fs::absolute(data_root) << "\n";
    std::cout << "Models: " << fs::absolute(models_dir) << "\n";

    for (const std::string& tier : tiers) {
        if (tier == "digits") {
            run_digits(data_root, templates_per_digit, test_per_digit, !skip_baselines,
                       neighbours, dtw_radius, !has_flag(argc, argv, "--no-endpoint"));
        } else if (tier == "timit") {
            run_timit(data_root, models_dir, limit, config);
        } else if (tier == "librispeech") {
            run_librispeech(data_root, models_dir, limit, dev_split, config, true,
                            has_flag(argc, argv, "--monophone"));
        } else {
            std::cout << "[WARN] unknown tier '" << tier << "'\n";
        }
    }

    std::cout << "\n";
    return 0;
}
