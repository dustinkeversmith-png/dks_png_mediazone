// Trains the networkless captioning stack described in models/README.md:
//
//   * monophone acoustic model  <- data/audio/timit  (hand-aligned .phn labels)
//   * word bigram LM            <- data/models/lm_corpus.txt (LibriSpeech
//                                  train-clean-100 transcripts, disjoint from
//                                  the test-clean evaluation set)
//   * phone bigram LM           <- the same TIMIT alignments, for phone
//                                  recognition on Tier 2
//
// No gradient descent, no EM bootstrap: the TIMIT boundaries give a supervised
// state assignment directly, so training is a single streaming pass that
// accumulates sufficient statistics.
//
// Usage: train_models [--data-root data] [--out data/models] [--holdout 0.15]

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "audio_loadnorm.hpp"

#include <models/acoustic_model.hpp>
#include <models/mfcc.hpp>
#include <models/ngram_lm.hpp>
#include <models/phone_set.hpp>
#include <models/scoring.hpp>

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

// Speaker id lives in the .json sidecar; used to keep train/test disjoint.
std::string read_speaker(const fs::path& json_path) {
    std::ifstream file(json_path);
    if (!file) return "unknown";
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    const size_t key = content.find("\"speaker_id\"");
    if (key == std::string::npos) return "unknown";
    const size_t first = content.find('"', content.find(':', key));
    if (first == std::string::npos) return "unknown";
    const size_t last = content.find('"', first + 1);
    if (last == std::string::npos) return "unknown";
    return content.substr(first + 1, last - first - 1);
}

std::string argument(int argc, char** argv, const std::string& flag,
                     const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path data_root = argument(argc, argv, "--data-root", "data");
    const fs::path out_dir = argument(argc, argv, "--out", "data/models");
    const double holdout = std::stod(argument(argc, argv, "--holdout", "0.15"));

    fs::create_directories(out_dir);

    std::cout << "======================================================================\n";
    std::cout << "TRAINING NETWORKLESS CAPTIONING MODELS\n";
    std::cout << "======================================================================\n";

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
        speakers.push_back(read_speaker(fs::path(wav).replace_extension(".json")));
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

    std::cout << "\nTIMIT: " << timit_wavs.size() << " utterances, " << unique.size()
              << " speakers (" << test_speakers.size() << " held out for evaluation)\n";

    models::MfccExtractor extractor;
    models::AcousticModel acoustic;
    acoustic.begin_training();

    // Cached frames + their state assignment, so mixture EM does not have to
    // run the front-end a second time. 63 min of TIMIT is ~377k frames x 39
    // floats = 59 MB, cheaper than re-decoding the audio.
    const int mixtures = std::stoi(argument(argc, argv, "--mixtures", "8"));
    std::vector<float> cached_frames;
    std::vector<int> cached_states;
    if (mixtures > 1) cached_frames.reserve(400000ull * models::kFeatureDim);

    std::vector<std::vector<std::string>> phone_sentences;
    models::RtfTimer timer;
    int trained = 0;
    int64_t frames = 0;

    for (size_t i = 0; i < timit_wavs.size(); ++i) {
        if (test_speakers.count(speakers[i])) continue;

        std::vector<float> audio;
        if (!load_and_preprocess_audio(timit_wavs[i].string(), audio) || audio.empty()) continue;

        const std::vector<Segment> segments =
            read_alignment(fs::path(timit_wavs[i]).replace_extension(".phn"));
        if (segments.empty()) continue;

        timer.start();
        const models::FeatureMatrix features = extractor.extract(audio);
        timer.stop();
        timer.add_audio(static_cast<double>(audio.size()) / models::kSampleRate);
        if (features.empty()) continue;

        std::vector<std::string> phone_sequence;
        for (const Segment& segment : segments) {
            // Sample offsets -> frame indices (25 ms window, 10 ms hop).
            const int start = segment.start_sample / models::kHopSize;
            const int stop = std::min(segment.stop_sample / models::kHopSize,
                                      features.num_frames);
            if (stop <= start) continue;
            acoustic.accumulate_segment(features, start, stop, segment.phone);
            if (mixtures > 1) {
                const int length = stop - start;
                for (int t = start; t < stop; ++t) {
                    int sub = ((t - start) * models::kNumStatesPerPhone) / length;
                    if (sub >= models::kNumStatesPerPhone) sub = models::kNumStatesPerPhone - 1;
                    cached_states.push_back(models::state_index(segment.phone, sub));
                    cached_frames.insert(cached_frames.end(), features.frame(t),
                                         features.frame(t) + models::kFeatureDim);
                }
            }
            frames += stop - start;
            phone_sequence.push_back(models::phone_names()[segment.phone]);
        }
        if (!phone_sequence.empty()) phone_sentences.push_back(std::move(phone_sequence));
        ++trained;

        if (trained % 200 == 0) {
            std::cout << "  " << trained << " utterances, " << frames << " frames\n";
        }
    }

    acoustic.finish_training();

    if (mixtures > 1 && !cached_states.empty()) {
        models::RtfTimer em_timer;
        em_timer.start();
        acoustic.train_mixtures(cached_frames, cached_states, mixtures);
        const double seconds = em_timer.stop();
        std::cout << "  mixture EM: " << mixtures << " components/state over "
                  << cached_states.size() << " frames in " << seconds << " s\n";
    }

    const fs::path am_path = out_dir / "monophone.am";
    if (!acoustic.save(am_path.string())) {
        std::cerr << "[ERROR] Could not write " << am_path << "\n";
        return 1;
    }
    std::cout << "  trained on " << trained << " utterances / " << frames << " frames ("
              << frames / 100.0 / 60.0 << " min)\n";
    std::cout << "  front-end speed: " << timer.times_real_time() << "x real time\n";
    std::cout << "  wrote " << am_path << " (" << models::num_states() << " states x "
              << acoustic.mixtures() << " mixtures x " << models::kFeatureDim << " dims)\n";

    // Phone bigram over the same alignments, used by the Tier 2 phone loop.
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
            std::cout << "  wrote " << (out_dir / "phone.lm") << " ("
                      << phone_lm.vocabulary_size() << " units)\n";
        }
    }

    // ------------------------------------------------------------ word LM
    const fs::path corpus = out_dir / "lm_corpus.txt";
    if (!fs::exists(corpus)) {
        std::cerr << "\n[WARN] " << corpus << " not found; skipping word LM.\n";
        std::cerr << "       It holds LibriSpeech train-clean-100 transcripts.\n";
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
