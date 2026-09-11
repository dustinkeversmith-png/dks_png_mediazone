#include "vocal/streaming_asr.hpp"
#include "vocal/features.hpp"
#include <models/scoring.hpp>
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
namespace {
std::string arg(int n, char** v, const std::string& key, std::string fallback = {}) {
    for (int i = 1; i < n; ++i) if (v[i] == key) {
        if (i + 1 == n) throw std::invalid_argument("Missing value for " + key);
        return v[i + 1];
    }
    return fallback;
}
bool flag(int n, char** v, const std::string& key) {
    for (int i = 1; i < n; ++i) if (v[i] == key) return true;
    return false;
}
std::string read(const fs::path& p) {
    std::ifstream in(p);
    if (!in) throw std::runtime_error("Cannot open " + p.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
std::string quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 32) out += ' ';
        else out += c;
    }
    return out + '"';
}
fs::path graph(const fs::path& dir, const std::string& prefix) {
    fs::path result;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const auto name = entry.path().filename().string();
        if (name.starts_with(prefix) && name.ends_with(".int8.onnx")) {
            if (!result.empty()) throw std::runtime_error("Ambiguous " + prefix + " graph");
            result = entry.path();
        }
    }
    if (result.empty()) throw std::runtime_error("Missing INT8 " + prefix + " graph in " + dir.string());
    return result;
}
double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    return values[std::min(values.size()-1, static_cast<size_t>(p * (values.size()-1)))];
}
void decode(vocal::StreamingOnnxAsr& asr, std::span<const float> samples, size_t chunk,
            bool partials = false) {
    std::string previous;
    for (size_t i = 0; i < samples.size(); i += chunk) {
        asr.accept(samples.subspan(i, std::min(chunk, samples.size()-i)));
        if (partials) {
            auto text = asr.text();
            if (text != previous) {
                std::cout << "{\"audio_s\":" << std::min(i+chunk, samples.size()) / 16000.0
                          << ",\"partial\":" << quote(text) << "}\n";
                previous = std::move(text);
            }
        }
    }
    asr.finish();
}
struct Capture {
    static constexpr size_t capacity = 160000;
    std::array<float, capacity> ring{};
    std::atomic<size_t> written{0}, consumed{0}, dropped{0};
    static void callback(ma_device* device, void*, const void* input, ma_uint32 count) {
        auto& c = *static_cast<Capture*>(device->pUserData);
        const auto w = c.written.load(std::memory_order_relaxed);
        const auto r = c.consumed.load(std::memory_order_acquire);
        if (count > capacity - (w-r)) { c.dropped.fetch_add(count); return; }
        const auto* samples = static_cast<const float*>(input);
        for (size_t i = 0; i < count; ++i) c.ring[(w+i)%capacity] = samples ? samples[i] : 0.F;
        c.written.store(w+count, std::memory_order_release);
    }
};
volatile std::sig_atomic_t stopped = 0;
void stop(int) { stopped = 1; }
void microphone(vocal::StreamingOnnxAsr& asr, int seconds_limit) {
    Capture capture;
    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = 16000;
    config.periodSizeInMilliseconds = 20;
    config.dataCallback = Capture::callback;
    config.pUserData = &capture;
    ma_device device;
    if (ma_device_init(nullptr, &config, &device) != MA_SUCCESS)
        throw std::runtime_error("Microphone initialization failed");
    struct Guard { ma_device* d; ~Guard() { ma_device_uninit(d); } } guard{&device};
    if (ma_device_start(&device) != MA_SUCCESS) throw std::runtime_error("Microphone start failed");
    std::signal(SIGINT, stop);
    const auto started = Clock::now();
    std::array<float, 1600> buffer{};
    std::string previous;
    while (!stopped && (seconds_limit == 0 || std::chrono::duration<double>(Clock::now()-started).count() < seconds_limit)) {
        const auto r = capture.consumed.load(std::memory_order_relaxed);
        const auto w = capture.written.load(std::memory_order_acquire);
        const auto n = std::min(buffer.size(), w-r);
        if (!n) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
        for (size_t i = 0; i < n; ++i) buffer[i] = capture.ring[(r+i)%Capture::capacity];
        capture.consumed.store(r+n, std::memory_order_release);
        asr.accept(std::span<const float>(buffer.data(), n));
        const auto text = asr.text();
        if (text != previous) { std::cout << text << '\n'; previous = text; }
    }
    ma_device_stop(&device);
    auto r = capture.consumed.load();
    const auto w = capture.written.load();
    while (r < w) {
        const auto n = std::min(buffer.size(), w-r);
        for (size_t i = 0; i < n; ++i) buffer[i] = capture.ring[(r+i)%Capture::capacity];
        asr.accept(std::span<const float>(buffer.data(), n)); r += n;
    }
    asr.finish();
    std::cout << "FINAL: " << asr.text() << '\n';
    if (capture.dropped.load()) throw std::runtime_error("Microphone overflow: " + std::to_string(capture.dropped.load()) + " samples lost");
}
void self_test(vocal::StreamingOnnxAsr& asr, const fs::path& wav) {
    auto check = [](bool yes, const char* message) { if (!yes) throw std::runtime_error(message); };
    std::vector<float> silence(16000*5, 0.F);
    decode(asr, silence, 137);
    check(asr.text().empty(), "Silence hallucination");
    check(asr.stats().encoder_calls == 0, "Gate decoded silence");
    asr.finish();
    bool rejected = false;
    try { asr.accept(std::span<const float>(silence.data(), 1)); }
    catch (const std::logic_error&) { rejected = true; }
    check(rejected, "Accepted audio after finish");
    auto audio = vocal::load_wav_mono(wav,16000);
    asr.reset(); decode(asr, audio.samples, 1600); const auto expected = asr.text();
    check(!expected.empty(), "Speech produced no tokens");
    asr.reset(); decode(asr, audio.samples, 137);
    check(asr.text() == expected, "Transport chunk boundaries changed transcript");
    asr.reset(); decode(asr, audio.samples, 2560);
    check(asr.text() == expected, "Cache reset or larger packet changed transcript");
    asr.reset(); asr.finish(); check(asr.text().empty(), "Empty stream retained transcript");
    std::cout << "Streaming regression checks passed\n";
}
} // namespace

int main(int argc, char** argv) try {
    std::cout << std::unitbuf << std::fixed << std::setprecision(6);
    if (argc == 1 || flag(argc,argv,"--help")) {
        std::cout << "caption-streaming --file audio.wav | --mic | --benchmark <librispeech-directory>\n"
                     "  [--models artifacts/models/asr_streaming_int8/compact] [--threads 2] [--packet-ms 100]\n"
                     "  [--dev --limit 40] [--report artifacts/asr_streaming_test.json] [--no-gate]\n"
                     "  --self-test <wav> tests silence, reset, finish, and packet-boundary invariance.\n";
        return 0;
    }
    const fs::path dir = arg(argc,argv,"--models","artifacts/models/asr_streaming_int8/compact");
    vocal::StreamingAsrConfig c;
    c.encoder=graph(dir,"encoder"); c.decoder=graph(dir,"decoder"); c.joiner=graph(dir,"joiner"); c.tokens=dir/"tokens.txt";
    c.threads=std::stoi(arg(argc,argv,"--threads","2"));
    c.packet_ms=std::stoi(arg(argc,argv,"--packet-ms","100"));
    c.energy_gate=!flag(argc,argv,"--no-gate");
    c.gate_rms=std::stof(arg(argc,argv,"--gate-rms","0.0003"));
    auto load_start=Clock::now();
    vocal::StreamingOnnxAsr asr(c);
    const double load_seconds=std::chrono::duration<double>(Clock::now()-load_start).count();
    std::cerr << "Model: " << asr.model_type() << ", native shift " << asr.model_chunk_ms()
              << " ms, initial feature window " << asr.first_window_ms() << " ms, packet " << c.packet_ms << " ms\n";
    if (flag(argc,argv,"--mic")) { microphone(asr,std::stoi(arg(argc,argv,"--seconds","0"))); return 0; }
    if (auto p=arg(argc,argv,"--self-test"); !p.empty()) { self_test(asr,p); return 0; }
    if (auto p=arg(argc,argv,"--file"); !p.empty()) {
        auto audio=vocal::load_wav_mono(p,16000);
        decode(asr,audio.samples,static_cast<size_t>(c.packet_ms)*16,true);
        std::cout << "FINAL: " << asr.text() << '\n';
        std::cerr << "RTF " << asr.stats().compute_seconds/(audio.samples.size()/16000.0) << '\n';
        return 0;
    }
    const auto data=arg(argc,argv,"--benchmark");
    if (data.empty()) throw std::invalid_argument("Select --file, --mic, --self-test, or --benchmark");
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(data)) if (e.path().extension()==".wav") files.push_back(e.path());
    std::sort(files.begin(),files.end());
    // Use exactly the legacy harness's split and scoring, fail rather than silently skip files.
    if (files.size()!=600) throw std::runtime_error("Expected the frozen 600-WAV evaluation corpus");
    const bool dev=flag(argc,argv,"--dev");
    if (dev) files.resize(files.size()/4); else files.erase(files.begin(),files.begin()+files.size()/4);
    const int limit=std::stoi(arg(argc,argv,"--limit","0"));
    if (limit<0) throw std::invalid_argument("Negative limit");
    if (limit>0 && static_cast<size_t>(limit)<files.size()) files.resize(limit);
    fs::path report=arg(argc,argv,"--report","artifacts/asr_streaming_test.json");
    if (!report.parent_path().empty()) fs::create_directories(report.parent_path());
    std::ofstream utterances(report.string()+".jsonl");
    if (!utterances) throw std::runtime_error("Cannot write utterance report");
    utterances << std::setprecision(10);
    models::EditCounts errors;
    double audio_seconds=0,compute=0,pipeline=0;
    std::vector<double> encoder_times,packet_times;
    std::uint64_t gated=0,calls=0;
    for (size_t i=0;i<files.size();++i) {
        auto refpath=files[i]; refpath.replace_extension(".txt");
        const std::string reference=read(refpath);
        if (models::tokenize(reference).empty()) throw std::runtime_error("Empty reference");
        const auto started=Clock::now();
        auto wav=vocal::load_wav_mono(files[i],16000);
        if (wav.samples.empty()) throw std::runtime_error("Empty audio");
        asr.reset();
        decode(asr,wav.samples,static_cast<size_t>(c.packet_ms)*16);
        const auto hyp=asr.text();
        pipeline+=std::chrono::duration<double>(Clock::now()-started).count();
        const auto counts=models::edit_distance(models::tokenize(reference),models::tokenize(hyp));
        errors+=counts;
        const auto& s=asr.stats();
        audio_seconds+=wav.samples.size()/16000.0; compute+=s.compute_seconds;
        gated+=s.gated_samples; calls+=s.encoder_calls;
        encoder_times.insert(encoder_times.end(),s.encoder_ms.begin(),s.encoder_ms.end());
        packet_times.insert(packet_times.end(),s.packet_ms.begin(),s.packet_ms.end());
        utterances << "{\"file\":" << quote(files[i].filename().string()) << ",\"reference\":" << quote(reference)
                   << ",\"hypothesis\":" << quote(hyp) << ",\"S\":" << counts.substitutions
                   << ",\"D\":" << counts.deletions << ",\"I\":" << counts.insertions
                   << ",\"N\":" << counts.reference_length << ",\"audio_seconds\":" << wav.samples.size()/16000.0
                   << ",\"compute_seconds\":" << s.compute_seconds << "}\n";
        if ((i+1)%20==0 || i+1==files.size())
            std::cout << i+1 << '/' << files.size() << " WER " << errors.error_rate()*100 << "% RTF " << compute/audio_seconds << '\n';
    }
    std::ofstream out(report);
    if (!out) throw std::runtime_error("Cannot write summary report");
    out << std::setprecision(10)
        << "{\n\"split\":" << quote(dev?"dev":"test") << ",\n\"utterances\":" << files.size()
        << ",\n\"reference_words\":" << errors.reference_length << ",\n\"substitutions\":" << errors.substitutions
        << ",\n\"deletions\":" << errors.deletions << ",\n\"insertions\":" << errors.insertions
        << ",\n\"wer\":" << errors.error_rate() << ",\n\"rtf\":" << compute/audio_seconds
        << ",\n\"pipeline_rtf\":" << pipeline/audio_seconds << ",\n\"audio_seconds\":" << audio_seconds
        << ",\n\"compute_seconds\":" << compute << ",\n\"model_load_seconds\":" << load_seconds
        << ",\n\"encoder_calls\":" << calls << ",\n\"gated_samples\":" << gated
        << ",\n\"encoder_p95_ms\":" << percentile(encoder_times,.95)
        << ",\n\"encoder_p99_ms\":" << percentile(encoder_times,.99)
        << ",\n\"packet_p95_ms\":" << percentile(packet_times,.95)
        << ",\n\"packet_p99_ms\":" << percentile(packet_times,.99)
        << ",\n\"packet_max_ms\":" << percentile(packet_times,1)
        << ",\n\"model_chunk_ms\":" << asr.model_chunk_ms()
        << ",\n\"initial_feature_window_ms\":" << asr.first_window_ms()
        << ",\n\"packet_ms\":" << c.packet_ms << ",\n\"threads\":" << c.threads
        << ",\n\"energy_gate\":" << (c.energy_gate?"true":"false") << ",\n\"gate_rms\":" << c.gate_rms
        << ",\n\"models\":" << quote(fs::absolute(dir).string())
        << ",\n\"wer_target_met\":" << (errors.error_rate()<.07?"true":"false")
        << ",\n\"rtf_target_met\":" << (pipeline/audio_seconds<.06?"true":"false")
        << ",\n\"latency_target_met\":false\n}\n";
    std::cout << "Saved " << report << '\n';
    return 0;
} catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
