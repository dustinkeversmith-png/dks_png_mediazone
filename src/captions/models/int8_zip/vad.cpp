#include <captions/vad.hpp>
#include "ort_graph.hpp"
#include <algorithm>
#include <array>

namespace captions {
namespace {
constexpr int kContext = 64;     // v5 prepends the previous 64 samples to each chunk
constexpr int kWarmup = 16;      // chunks (0.5 s) decoded before a stripe's first scored chunk
constexpr int kMinStripe = 62;   // ~2 s: short files still get several parallel stripes
constexpr int kMaxStripes = 256;
constexpr int kState = 128;
} // namespace

struct SileroVad::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "vad"};
    detail::Graph graph;
    Impl(const std::filesystem::path& model, int threads) : graph(env, model, threads) {
        if (graph.inputs.size() != 3 || graph.outputs.size() != 2)
            throw std::runtime_error("Expected Silero VAD v5 (input, state, sr)");
    }
};

SileroVad::SileroVad(const std::filesystem::path& model, int threads)
    : impl_(std::make_unique<Impl>(model, threads)) {}
SileroVad::~SileroVad() = default;

std::vector<float> SileroVad::probabilities(std::span<const float> audio) {
    const auto chunks = static_cast<int>((audio.size() + kChunk - 1) / kChunk);
    std::vector<float> prob(static_cast<size_t>(chunks), 0.F);
    if (chunks == 0) return prob;
    const int stripes = std::clamp(chunks / kMinStripe, 1, kMaxStripes);
    const int per = (chunks + stripes - 1) / stripes;  // scored chunks per stripe
    const int steps = per + kWarmup;
    const auto B = static_cast<std::int64_t>(stripes);
    constexpr int W = kContext + kChunk;

    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> x(static_cast<size_t>(B) * W), state(2 * static_cast<size_t>(B) * kState, 0.F);
    std::int64_t rate = 16000;
    const std::array<std::int64_t, 2> xs{B, W};
    const std::array<std::int64_t, 3> ss{2, B, kState};
    // Sample at absolute index i, zero outside the recording.
    auto sample = [&](std::int64_t i) {
        return i >= 0 && i < static_cast<std::int64_t>(audio.size()) ? audio[static_cast<size_t>(i)] : 0.F;
    };
    for (int step = 0; step < steps; ++step) {
        for (std::int64_t s = 0; s < B; ++s) {
            const std::int64_t chunk = s * per + step - kWarmup;  // may be negative during warm-up
            const std::int64_t first = chunk * kChunk - kContext;
            float* row = x.data() + s * W;
            for (int i = 0; i < W; ++i) row[i] = sample(first + i);
        }
        std::array<Ort::Value, 3> in{
            Ort::Value::CreateTensor<float>(memory, x.data(), x.size(), xs.data(), xs.size()),
            Ort::Value::CreateTensor<float>(memory, state.data(), state.size(), ss.data(), ss.size()),
            Ort::Value::CreateTensor<std::int64_t>(memory, &rate, 1, nullptr, 0)};
        auto out = impl_->graph.run(in);
        const float* p = out[0].GetTensorData<float>();
        std::copy_n(out[1].GetTensorData<float>(), state.size(), state.data());
        if (step < kWarmup) continue;
        for (std::int64_t s = 0; s < B; ++s) {
            const std::int64_t chunk = s * per + step - kWarmup;
            if (chunk < chunks) prob[static_cast<size_t>(chunk)] = p[s];
        }
    }
    return prob;
}
} // namespace captions
