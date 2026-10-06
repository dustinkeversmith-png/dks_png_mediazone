#pragma once
// ONNX session wrapper and filterbank settings shared by the streaming and
// batch Zipformer engines, so both read the same graph contract.
#include "ort_session_options.hpp"
#include <kaldi-native-fbank/csrc/online-feature.h>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace captions::detail {
// CPU identity for the optimized-model cache key: ORT's fully optimized
// graphs may use kernels/layouts chosen for the CPU they were built on.
inline std::string cpu_brand() {
    std::string brand;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4] = {};
    for (int leaf = 0x80000002; leaf <= 0x80000004; ++leaf) {
        __cpuid(regs, leaf);
        brand.append(reinterpret_cast<const char*>(regs), sizeof(regs));
    }
#endif
    std::string key;
    for (char c : brand) if (std::isalnum(static_cast<unsigned char>(c))) key += c;
    return key.empty() ? "cpu" : key;
}

// Session creation spends most of its time optimizing the graph. The first
// load saves the optimized graph to <model dir>/.ort-cache/ (keyed by ORT
// version, model size + mtime and CPU); later loads read it with optimization
// off. Any cache problem falls back to optimizing the original model.
inline Ort::Session load_session(Ort::Env& env, const std::filesystem::path& path, Ort::SessionOptions options) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    const auto time = fs::last_write_time(path, ec).time_since_epoch().count();
    if (ec) return Ort::Session(env, path.c_str(), options);
    // One loader at a time per process decides whether to read or write the
    // cache, so concurrent engines never race on the same file.
    static std::mutex cache_mutex;
    std::lock_guard lock(cache_mutex);
    const fs::path dir = path.parent_path() / ".ort-cache";
    const fs::path cached = dir / (path.stem().string() + "." + OrtGetApiBase()->GetVersionString() + "." +
                                   std::to_string(size) + "." + std::to_string(time) + "." + cpu_brand() + ".onnx");
    if (fs::exists(cached, ec)) {
        try {
            Ort::SessionOptions fast = options.Clone();
            fast.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
            return Ort::Session(env, cached.c_str(), fast);
        } catch (const Ort::Exception&) { fs::remove(cached, ec); }
    }
    fs::create_directories(dir, ec);
    if (ec) return Ort::Session(env, path.c_str(), options);
    // Write under a unique name, then rename, so concurrent loads never read a partial file.
    const fs::path partial = cached.string() + "." + std::to_string(std::random_device{}()) + ".tmp";
    Ort::SessionOptions saving = options.Clone();
    saving.SetOptimizedModelFilePath(partial.c_str());
    saving.SetLogSeverityLevel(3);  // the CPU-specific-graph warning is expected: the key includes the CPU
    Ort::Session session(env, path.c_str(), saving);
    fs::rename(partial, cached, ec);
    if (ec) fs::remove(partial, ec);
    return session;
}

struct Graph {
    Ort::Session session;
    std::vector<std::string> inputs, outputs;
    std::vector<const char*> in_names, out_names;
    Graph(Ort::Env& env, const std::filesystem::path& path, int threads, bool spin = false)
        : session(load_session(env, path, cpu_session_options(threads, spin))) {
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

namespace captions::detail {
// NeMo FastConformer front end: 80 Slaney mel bins of a Hann-windowed 512-point
// power spectrum, 0.97 pre-emphasis, no DC removal or dither. Offline models
// then normalize each mel bin over the utterance; streaming ones do not.
inline knf::FbankOptions nemo_fbank_options(int bins) {
    knf::FbankOptions o;
    o.frame_opts.samp_freq = 16000;
    o.frame_opts.dither = 0;
    o.frame_opts.snip_edges = false;
    o.frame_opts.remove_dc_offset = false;
    o.frame_opts.window_type = "hann";
    o.frame_opts.preemph_coeff = 0.97F;
    o.mel_opts.num_bins = bins;
    o.mel_opts.low_freq = 0;
    o.mel_opts.high_freq = 8000;
    o.mel_opts.is_librosa = true;
    return o;
}

// NeMo tokens.txt: vocab_size pieces, then <blk> as id vocab_size.
inline std::vector<std::string> read_nemo_tokens(const std::filesystem::path& path, int vocabulary) {
    std::vector<std::string> tokens(static_cast<size_t>(vocabulary) + 1);
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open ASR tokens");
    for (std::string line; std::getline(file, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto at = line.find_last_of(" \t");
        if (at == std::string::npos) continue;
        const int id = std::stoi(line.substr(at + 1));
        if (id >= 0 && id <= vocabulary) tokens[static_cast<size_t>(id)] = line.substr(0, at);
    }
    if (tokens.back() != "<blk>") throw std::runtime_error("NeMo tokens do not end with <blk>");
    return tokens;
}
} // namespace captions::detail
