#include "vocal/evaluator.hpp"
#include "vocal/acoustic_model.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vocal {
namespace {

std::vector<double> parse_line(const std::string& line) {
    std::string normalized = line;
    for (auto& c : normalized) if (c == ',') c = ' ';
    std::istringstream input(normalized);
    std::vector<double> values;
    for (double value; input >> value;) values.push_back(value);
    return values;
}

}  // namespace

metrics::Frames Evaluator::load_matrix(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open feature matrix: " + path.string());
    metrics::Frames frames;
    for (std::string line; std::getline(input, line);) {
        auto frame = parse_line(line);
        if (!frame.empty()) {
            if (!frames.empty() && frame.size() != frames.front().size()) {
                throw std::runtime_error("inconsistent feature width: " + path.string());
            }
            frames.push_back(std::move(frame));
        }
    }
    return frames;
}

std::vector<double> Evaluator::load_vector(const std::filesystem::path& path) {
    auto rows = load_matrix(path);
    std::vector<double> result;
    for (const auto& row : rows) result.insert(result.end(), row.begin(), row.end());
    return result;
}

EvaluationResult Evaluator::evaluate(const EvaluationCase& item) const {
    EvaluationResult result{item.utterance_id,
                            metrics::mcd_db(load_matrix(item.reference_mcep),
                                            load_matrix(item.synthesized_mcep)),
                            std::nullopt,
                            std::nullopt};
    if (item.reference_f0 && item.synthesized_f0) {
        result.pitch = metrics::pitch(load_vector(*item.reference_f0),
                                      load_vector(*item.synthesized_f0));
    }
    if (item.reference_text && item.hypothesis_text) {
        result.wer = metrics::word_error_rate(*item.reference_text, *item.hypothesis_text);
    }
    return result;
}

std::span<const float> MelSpectrogram::frame(std::size_t index) const {
    if (index >= frames || log_mel.size() != frames * bins) {
        throw std::out_of_range("invalid mel-spectrogram frame");
    }
    return {log_mel.data() + index * bins, bins};
}

}  // namespace vocal
