#include "forced_align.hpp"
#include "ngram_lm.hpp"
#include <filesystem>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    using namespace models;
    Lexicon lexicon;
    const int sil = phone_id("SIL"), k = phone_id("K"), ae = phone_id("AE");
    const int t = phone_id("T"), s = phone_id("S");
    lexicon.add_entry("CAT", {k, ae, t});
    lexicon.add_entry("SAT", {s, ae, t});
    PronunciationTable table;
    table.build(lexicon);
    std::vector<int> phones;
    std::vector<std::pair<int, int>> contexts;
    check(table.phones_for({"CAT", "SAT"}, phones, nullptr, false, &contexts), "pronunciation failed");
    check(phones == std::vector<int>({sil,k,ae,t,s,ae,t,sil}), "unexpected phones");
    check(contexts.size() == phones.size(), "context count");
    check(contexts[3] == std::make_pair(ae,sil), "word end leaked next word");
    check(contexts[4] == std::make_pair(sil,ae), "word start leaked previous word");
    check(contexts[2] == std::make_pair(k,t), "internal context lost");
    check(contexts.front() == std::make_pair(sil,sil), "silence context");
    check(table.phones_for({"CAT", "SAT"}, phones, nullptr, true, &contexts), "pause pronunciation failed");
    check(phones.size() == contexts.size() && phones[4] == sil, "pause contexts misaligned");
    check(contexts[4] == std::make_pair(sil,sil), "pause context");
    int guessed = 0;
    check(table.phones_for({"ZORB"}, phones, &guessed, false, &contexts), "G2P failed");
    check(guessed == 1 && phones.size() == contexts.size(), "G2P contexts misaligned");

    AcousticModel model;
    model.begin_training();
    model.finish_training();
    ForcedAligner aligner;
    FeatureMatrix features;
    features.num_frames = 6;
    features.data.assign(6 * kFeatureDim, 0.0f);
    auto segments = aligner.align(features, {k,k}, model);
    check(segments.size() == 2, "repeated phones collapsed");
    check(segments[0].stop_frame == 3, "invalid three-state path");
    features.num_frames = 5;
    check(aligner.align(features, {k,k}, model).empty(), "impossible path accepted");
    check(aligner.state_path().empty(), "failed alignment retained old path");
    bool rejected = false;
    try { aligner.align(features, {k,k}, model, nullptr, &contexts); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "bad context length accepted");
    const auto corpus = std::filesystem::temp_directory_path() /
        ("asr-lm-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
    { std::ofstream out(corpus); out << "CAT SAT CAT\nCAT SAT\nSAT CAT\n"; }
    NgramLm lm;
    const bool trained = lm.train_from_text(corpus.string(), 20);
    std::filesystem::remove(corpus);
    check(trained, "LM training failed");
    for (int h = 0; h < lm.vocabulary_size(); ++h) {
        double sum = 0.0;
        for (int w = 0; w < lm.vocabulary_size(); ++w) {
            const double probability = std::exp(lm.log_probability(h,w));
            sum += probability;
            check(probability + 1e-7 >= std::exp(lm.log_backoff(h) + lm.log_unigram(w)),
                  "explicit probability below decoder backoff");
        }
        check(std::abs(sum - 1.0) < 1e-5, "LM probabilities do not sum to one");
    }
    std::cout << "Model context and LM regression checks passed\n";
}
