// Caption one audio file with the networkless stack: MFCC -> monophone GMM-HMM
// -> CMUDict lexicon -> bigram Viterbi beam search.
//
// Usage:
//   caption <file.wav> [--models data/models] [--phones] [--acoustic-scale F]
//           [--word-penalty F] [--max-active N] [--beam F] [--word-beam F]
//
//   --phones  decode a phone string instead of words (uses the phone loop)

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "audio_loadnorm.hpp"

#include <models/acoustic_model.hpp>
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argv[1][0] == '-') {
        std::cerr << "Usage: caption <file.wav> [--models data/models] [--phones]\n";
        return 2;
    }

    const std::string input = argv[1];
    const fs::path models_dir = argument(argc, argv, "--models", "data/models");
    const bool phones_only = has_flag(argc, argv, "--phones");

    models::DecoderConfig config;
    config.acoustic_scale = std::stof(argument(argc, argv, "--acoustic-scale", "0.2"));
    config.word_insertion_penalty = std::stof(argument(argc, argv, "--word-penalty", phones_only ? "-6" : "-2"));
    config.beam = std::stof(argument(argc, argv, "--beam", "120"));
    config.word_beam = std::stof(argument(argc, argv, "--word-beam", phones_only ? "6" : "10"));
    config.max_active = std::stoi(argument(argc, argv, "--max-active", "12000"));

    models::AcousticModel acoustic;
    if (!acoustic.load((models_dir / "monophone.am").string())) {
        std::cerr << "[ERROR] cannot load " << (models_dir / "monophone.am")
                  << " - run train_models first\n";
        return 1;
    }

    models::NgramLm lm;
    models::Lexicon lexicon;
    if (phones_only) {
        if (!lm.load((models_dir / "phone.lm").string())) {
            std::cerr << "[ERROR] cannot load phone.lm\n";
            return 1;
        }
        for (int p = 0; p < models::num_phones(); ++p) {
            lexicon.add_entry(models::phone_names()[p], {p});
        }
        config.allow_silence = false;
    } else {
        if (!lm.load((models_dir / "bigram.lm").string())) {
            std::cerr << "[ERROR] cannot load bigram.lm\n";
            return 1;
        }
        std::unordered_set<std::string> vocabulary;
        for (int w = 0; w < lm.vocabulary_size(); ++w) vocabulary.insert(lm.word(w));
        if (!lexicon.load_cmudict((models_dir / "cmudict.dict").string(), vocabulary)) {
            std::cerr << "[ERROR] cannot load cmudict.dict\n";
            return 1;
        }
    }

    std::vector<float> audio;
    if (!load_and_preprocess_audio(input, audio) || audio.empty()) {
        std::cerr << "[ERROR] cannot read audio: " << input << "\n";
        return 1;
    }
    const double duration = static_cast<double>(audio.size()) / models::kSampleRate;

    // A triphone tree is used when one was trained; without it the same model
    // file is a plain monophone system.
    models::TriphoneTree tree;
    const bool has_tree = !phones_only && tree.load((models_dir / "triphone.tree").string());

    models::ViterbiDecoder decoder;
    decoder.build(lexicon, lm, acoustic, config, has_tree ? &tree : nullptr);

    models::MfccExtractor extractor;
    models::RtfTimer timer;
    timer.start();
    const models::FeatureMatrix features = extractor.extract(audio);
    const models::DecodeResult result = decoder.decode(features);
    const double seconds = timer.stop();

    for (const std::string& token : result.words) {
        if (token == "<s>" || token == "</s>" || token == "<unk>") continue;
        std::cout << token << ' ';
    }
    std::cout << "\n";

    std::cerr << std::fixed << std::setprecision(2);
    std::cerr << "[" << duration << " s audio, " << features.num_frames << " frames, decoded in "
              << seconds << " s = " << duration / seconds << "x real time, "
              << decoder.num_states() << " HMM states]\n";
    return 0;
}
