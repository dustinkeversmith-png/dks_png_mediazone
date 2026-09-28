#pragma once
// ONNX session wrapper and filterbank settings shared by the streaming and
// batch Zipformer engines, so both read the same graph contract.
#include "ort_session_options.hpp"
#include <kaldi-native-fbank/csrc/online-feature.h>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace captions::detail {
struct Graph {
    Ort::Session session;
    std::vector<std::string> inputs, outputs;
    std::vector<const char*> in_names, out_names;
    Graph(Ort::Env& env, const std::filesystem::path& path, int threads, bool spin = false)
        : session(env, path.c_str(), cpu_session_options(threads, spin)) {
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session.GetInputCount(); ++i)
            inputs.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        for (size_t i = 0; i < session.GetOutputCount(); ++i)
            outputs.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        for (const auto& s : inputs) in_names.push_back(s.c_str());
        for (const auto& s : outputs) out_names.push_back(s.c_str());
    }
    std::string metadata(const char* key) const {
        Ort::AllocatorWithDefaultOptions allocator;
        auto value = session.GetModelMetadata().LookupCustomMetadataMapAllocated(key, allocator);
        if (!value) throw std::runtime_error(std::string("Missing ONNX metadata: ") + key);
        return value.get();
    }
    std::vector<Ort::Value> run(std::span<const Ort::Value> in) {
        return session.Run(Ort::RunOptions{nullptr}, in_names.data(), in.data(), in.size(),
                           out_names.data(), out_names.size());
    }
};

inline knf::FbankOptions fbank_options() {
    knf::FbankOptions o;
    o.frame_opts.samp_freq = 16000;
    o.frame_opts.dither = 0;
    o.frame_opts.snip_edges = false;
    o.mel_opts.num_bins = 80;
    o.mel_opts.low_freq = 20;
    o.mel_opts.high_freq = -400;
    return o;
}

// k2 token table; ids beyond the model vocabulary are disambiguation symbols.
inline std::vector<std::string> read_tokens(const std::filesystem::path& path, int vocabulary, int& unknown) {
    std::vector<std::string> tokens(vocabulary);
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open ASR tokens");
    std::string line;
    unknown = -1;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto at = line.find_last_of(" \t");
        if (at == std::string::npos) continue;
        const int id = std::stoi(line.substr(at + 1));
        if (id < 0 || id >= vocabulary) continue;
        tokens[id] = line.substr(0, at);
        if (tokens[id] == "<unk>") unknown = id;
    }
    if (tokens[0] != "<blk>")
        throw std::runtime_error("Token vocabulary does not match model");
    for (const auto& t : tokens) if (t.empty()) throw std::runtime_error("Token vocabulary does not match model");
    return tokens;
}
} // namespace captions::detail
