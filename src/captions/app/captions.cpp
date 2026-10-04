// captions: one command line for the Zipformer2 caption engine.
//   batch (default)  files and video, many times real time, beam-4 search
//   live             microphone or file through the streaming engine
// Plus --benchmark and --self-test for the frozen LibriSpeech split.
#include <captions/caption_engine.hpp>
#include <captions/streaming_asr.hpp>
#include <input/wav_reader.hpp>
#include <benchmarks/scoring.hpp>
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
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {
constexpr const char* kUsage = R"(captions - Zipformer2 speech-to-text captions (CPU, offline)

USAGE
  captions --input <media> [--format srt] [--out <path>]       caption a file (batch)
  captions --mode live --mic [--seconds N]                     live microphone captions
  captions --mode live --input <media> [--realtime]            live engine on a file
  captions --benchmark <librispeech-dir> [--mode batch|live] [--dev] [--limit N]
           [--longform [--repeat N]] [--report <file.json>]
  captions --self-test <librispeech-dir>                       regression checks

INPUT / OUTPUT
  --input <path>       any audio/video ffmpeg can read; PCM WAV is read directly
  --format <list>      srt | vtt | txt | json, comma-separated for several (default srt)
  --out <path>         output base path (default: input path without extension);
                       '-' writes to stdout (one format only)
  --upper              keep the model's raw uppercase instead of sentence case
  --line-chars N       caption line width, two lines per cue (default 42)
  --ffmpeg <path>      ffmpeg executable for non-WAV input

MODES
  --mode batch         (default) whole file at once: segment, batch, beam search
  --mode live          streaming engine, text appears as audio arrives (greedy search)
  --mic                capture from the default microphone (implies --mode live)
  --seconds N          stop microphone capture after N seconds (default: Ctrl+C)
  --realtime           live mode on a file: feed audio at real-time pace

TUNING
  --threads T          threads per worker in batch mode, encoder threads in live mode (default 2)
  --workers N          batch workers (default 6)
  --batch B            segments per encoder call (default 16)
  --beam K             beam width (default 4; 1 = greedy)
  --max-seg S          longest segment in seconds (default 20)
  --gate-rms R         speech energy gate (default 0.0003)
  --packet-ms N        live packet size, 10..160 (default 100)
  --no-gate            live mode: decode silence too
  --models <dir>       relocated Zipformer2 checkpoint directory
  --quiet              no progress/statistics on stderr
)";

// Flags and which take a value; anything else is rejected.
const std::map<std::string, bool> kFlags = {
    {"--input", true}, {"--format", true}, {"--out", true}, {"--upper", false}, {"--line-chars", true},
    {"--ffmpeg", true}, {"--mode", true}, {"--mic", false}, {"--seconds", true}, {"--realtime", false},
    {"--threads", true}, {"--workers", true}, {"--batch", true}, {"--beam", true}, {"--max-seg", true},
    {"--gate-rms", true}, {"--packet-ms", true}, {"--no-gate", false}, {"--models", true}, {"--quiet", false},
    {"--benchmark", true}, {"--dev", false}, {"--limit", true}, {"--longform", false}, {"--repeat", true},
    {"--report", true}, {"--self-test", true}, {"--help", false}, {"-h", false}};

struct Args {
    std::map<std::string, std::string> values;
    std::set<std::string> flags;
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            const auto known = kFlags.find(key);
            if (known == kFlags.end()) throw std::invalid_argument("Unknown option " + key + " (see --help)");
            if (!known->second) { flags.insert(key); continue; }
            if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
            values[key] = argv[++i];
        }
    }
    bool has(const std::string& k) const { return flags.count(k) || values.count(k); }
    std::string get(const std::string& k, std::string fallback = {}) const {
        const auto it = values.find(k);
        return it == values.end() ? fallback : it->second;
    }
    int integer(const std::string& k, int fallback) const {
        const auto v = get(k);
        if (v.empty()) return fallback;
        size_t used = 0;
        const int n = std::stoi(v, &used);
        if (used != v.size()) throw std::invalid_argument("Expected an integer for " + k);
        return n;
    }
    double number(const std::string& k, double fallback) const {
        const auto v = get(k);
        if (v.empty()) return fallback;
        size_t used = 0;
        const double n = std::stod(v, &used);
        if (used != v.size()) throw std::invalid_argument("Expected a number for " + k);
        return n;
    }
};

std::string read_text(const fs::path& p) {
    std::ifstream in(p);
    if (!in) throw std::runtime_error("Cannot open " + p.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
std::string quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') { out += '\\'; out += static_cast<char>(c); }
        else if (c < 32) out += ' ';
        else out += static_cast<char>(c);
    }
    return out + '"';
}
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::vector<captions::CaptionFormat> formats(const Args& a) {
    std::vector<captions::CaptionFormat> out;
    std::stringstream list(a.get("--format", "srt"));
    for (std::string name; std::getline(list, name, ',');)
        if (!name.empty()) out.push_back(captions::parse_format(name));
    if (out.empty()) throw std::invalid_argument("--format needs at least one of srt, vtt, txt, json");
    return out;
}

captions::CaptionEngineConfig engine_config(const Args& a) {
    captions::CaptionEngineConfig c;
    c.models = a.get("--models");
    c.workers = a.integer("--workers", c.workers);
    c.threads_per_worker = a.integer("--threads", c.threads_per_worker);
    c.batch = a.integer("--batch", c.batch);
    c.beam = a.integer("--beam", c.beam);
    c.segmenter.max_seconds = a.number("--max-seg", c.segmenter.max_seconds);
    c.segmenter.min_seconds = std::min(c.segmenter.min_seconds, c.segmenter.max_seconds * 0.4);
    c.segmenter.gate_rms = static_cast<float>(a.number("--gate-rms", c.segmenter.gate_rms));
    c.ffmpeg = a.get("--ffmpeg");
    return c;
}

captions::StreamingAsrConfig live_config(const Args& a) {
    const auto files = captions::model_files(a.has("--models") ? fs::path(a.get("--models")) : captions::default_model_dir());
    captions::StreamingAsrConfig c;
    c.encoder = files.encoder; c.decoder = files.decoder; c.joiner = files.joiner; c.tokens = files.tokens;
    c.threads = std::min(4, a.integer("--threads", 2));
    c.packet_ms = a.integer("--packet-ms", 100);
    c.energy_gate = !a.has("--no-gate");
    c.gate_rms = static_cast<float>(a.number("--gate-rms", c.gate_rms));
    return c;
}

void print_batch_stats(const captions::BatchAsrStats& s, double extra_seconds) {
    std::cerr << std::fixed << std::setprecision(1) << "Captioned " << s.audio_seconds / 60 << " min of audio in "
              << s.wall_seconds + extra_seconds << " s (" << std::setprecision(0)
              << s.audio_seconds / (s.wall_seconds + extra_seconds) << "x real time; " << s.segments
              << " segments)\n";
}

// ---- batch ----------------------------------------------------------------
int run_batch(const Args& a) {
    const fs::path input = a.get("--input");
    const auto fmts = formats(a);
    const std::string out_arg = a.get("--out");
    const bool to_stdout = out_arg == "-";
    if (to_stdout && fmts.size() != 1) throw std::invalid_argument("--out - writes one format; pass a single --format");
    const bool quiet = a.has("--quiet");

    const auto t0 = Clock::now();
    captions::CaptionEngine engine(engine_config(a));
    const auto audio = captions::load_audio(input, engine.config().ffmpeg);
    const double setup = since(t0);
    if (!quiet) std::cerr << "Decoded " << std::fixed << std::setprecision(1) << audio.size() / 16000.0
                          << " s of audio\n";
    const auto segments = engine.transcribe_pcm(audio);
    captions::CueOptions cue;
    cue.line_chars = static_cast<size_t>(a.integer("--line-chars", 42));
    cue.sentence_case = !a.has("--upper");
    const auto cues = captions::build_cues(segments, cue);

    for (const auto f : fmts) {
        const auto text = captions::render(f, segments, cues);
        if (to_stdout) { std::cout << text; continue; }
        // Default: next to the input. --out is a base path; a caption extension
        // on it (talk.srt) is swapped per format rather than doubled.
        fs::path path = out_arg.empty() ? input : fs::path(out_arg);
        const auto ext = path.extension().string();
        const bool caption_ext = ext == ".srt" || ext == ".vtt" || ext == ".txt" || ext == ".json";
        if (out_arg.empty() || caption_ext) path.replace_extension(captions::extension(f));
        else path += captions::extension(f);
        if (path.has_parent_path()) fs::create_directories(path.parent_path());
        std::ofstream file(path, std::ios::binary);
        if (!file || !(file << text)) throw std::runtime_error("Cannot write " + path.string());
        if (!quiet) std::cerr << "Wrote " << path.string() << '\n';
    }
    if (!quiet) print_batch_stats(engine.stats(), setup);
    return 0;
}

// ---- live -----------------------------------------------------------------
// Live text only ever grows by appending, so the terminal shows it as a
// running transcript: print each new suffix as it is decoded.
struct LivePrinter {
    std::string shown;
    void update(const std::string& text) {
        if (text.size() > shown.size() && text.compare(0, shown.size(), shown) == 0) {
            std::cout << text.substr(shown.size()) << std::flush;
        } else if (text != shown) {
            std::cout << '\n' << text << std::flush;
        }
        shown = text;
    }
};

struct Capture {
    static constexpr size_t capacity = 160000;
    std::array<float, capacity> ring{};
    std::atomic<size_t> written{0}, consumed{0}, dropped{0};
    static void callback(ma_device* device, void*, const void* input, ma_uint32 count) {
        auto& c = *static_cast<Capture*>(device->pUserData);
        const auto w = c.written.load(std::memory_order_relaxed);
        const auto r = c.consumed.load(std::memory_order_acquire);
        if (count > capacity - (w - r)) { c.dropped.fetch_add(count); return; }
        const auto* samples = static_cast<const float*>(input);
        for (size_t i = 0; i < count; ++i) c.ring[(w + i) % capacity] = samples ? samples[i] : 0.F;
        c.written.store(w + count, std::memory_order_release);
    }
};
volatile std::sig_atomic_t stopped = 0;
void on_interrupt(int) { stopped = 1; }

void microphone(captions::StreamingOnnxAsr& asr, int seconds_limit, LivePrinter& printer) {
    // The callback only fills a lock-free ring; ONNX never runs on the audio thread.
    auto capture = std::make_unique<Capture>();
    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = 16000;
    config.periodSizeInMilliseconds = 20;
    config.dataCallback = Capture::callback;
    config.pUserData = capture.get();
    ma_device device;
    if (ma_device_init(nullptr, &config, &device) != MA_SUCCESS) throw std::runtime_error("Microphone initialization failed");
    struct Guard { ma_device* d; ~Guard() { ma_device_uninit(d); } } guard{&device};
    if (ma_device_start(&device) != MA_SUCCESS) throw std::runtime_error("Microphone start failed");
    std::signal(SIGINT, on_interrupt);
    const auto started = Clock::now();
    std::array<float, 1600> buffer{};
    auto drain = [&] {
        const auto r = capture->consumed.load(std::memory_order_relaxed);
        const auto w = capture->written.load(std::memory_order_acquire);
        const auto n = std::min(buffer.size(), w - r);
        for (size_t i = 0; i < n; ++i) buffer[i] = capture->ring[(r + i) % Capture::capacity];
        capture->consumed.store(r + n, std::memory_order_release);
        if (n) asr.accept(std::span<const float>(buffer.data(), n));
        return n;
    };
    while (!stopped && (seconds_limit == 0 || since(started) < seconds_limit)) {
        if (!drain()) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
        printer.update(asr.text());
    }
    ma_device_stop(&device);
    while (drain()) {}
    if (capture->dropped.load())
        throw std::runtime_error("Microphone overflow: " + std::to_string(capture->dropped.load()) + " samples lost");
}

int run_live(const Args& a) {
    const bool quiet = a.has("--quiet");
    captions::StreamingOnnxAsr asr(live_config(a));
    if (!quiet) std::cerr << "Live captions: " << asr.model_chunk_ms() << " ms model chunks, "
                          << asr.first_window_ms() << " ms before the first words"
                          << (a.has("--mic") ? "; Ctrl+C to stop" : "") << '\n';
    LivePrinter printer;
    double audio_seconds = 0;
    if (a.has("--mic")) {
        microphone(asr, a.integer("--seconds", 0), printer);
    } else {
        const auto audio = captions::load_audio(a.get("--input"), a.get("--ffmpeg"));
        audio_seconds = audio.size() / 16000.0;
        const size_t packet = static_cast<size_t>(a.integer("--packet-ms", 100)) * 16;
        const auto started = Clock::now();
        for (size_t i = 0; i < audio.size(); i += packet) {
            asr.accept(std::span<const float>(audio).subspan(i, std::min(packet, audio.size() - i)));
            printer.update(asr.text());
            if (a.has("--realtime"))
                std::this_thread::sleep_until(started + std::chrono::microseconds(
                    static_cast<long long>((i + packet) / 16.0 * 1000)));
        }
    }
    asr.finish();
    printer.update(asr.text());
    std::cout << '\n';
    if (const auto out = a.get("--out"); !out.empty() && out != "-") {
        fs::path path = out;
        if (path.extension() != ".txt") path += ".txt";
        std::ofstream file(path, std::ios::binary);
        if (!file || !(file << asr.text() << '\n')) throw std::runtime_error("Cannot write " + path.string());
        if (!quiet) std::cerr << "Wrote " << path.string() << '\n';
    }
    if (!quiet && audio_seconds > 0)
        std::cerr << std::fixed << std::setprecision(0) << "Decode speed " << audio_seconds / asr.stats().compute_seconds
                  << "x real time\n";
    return 0;
}

// ---- benchmark --------------------------------------------------------------
std::vector<fs::path> frozen_split(const fs::path& data, bool dev, int limit) {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(data)) if (e.path().extension() == ".wav") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    // The frozen split every published number uses: first quarter dev, rest test.
    if (files.size() != 600) throw std::runtime_error("Expected the frozen 600-WAV evaluation corpus");
    if (dev) files.resize(files.size() / 4); else files.erase(files.begin(), files.begin() + files.size() / 4);
    if (limit < 0) throw std::invalid_argument("Negative --limit");
    if (limit > 0 && static_cast<size_t>(limit) < files.size()) files.resize(limit);
    return files;
}
std::string reference_for(fs::path wav) { wav.replace_extension(".txt"); return read_text(wav); }

int run_benchmark(const Args& a) {
    const fs::path data = a.get("--benchmark");
    const bool live = a.get("--mode", "batch") == "live";
    const bool dev = a.has("--dev");
    const auto files = frozen_split(data, dev, a.integer("--limit", 0));
    std::vector<std::vector<float>> audio;
    for (const auto& f : files) audio.push_back(captions::read_wav_mono(f, 16000).samples);
    double audio_seconds = 0;
    for (const auto& x : audio) audio_seconds += x.size() / 16000.0;

    std::vector<std::string> hyp(files.size());
    double compute = 0;
    std::string details;
    if (a.has("--longform")) {
        if (live) throw std::invalid_argument("--longform measures the batch engine");
        // The split back to back with 0.5 s gaps (x --repeat): one long
        // recording, segmented automatically and scored as one word sequence.
        const int repeat = a.integer("--repeat", 1);
        if (repeat < 1 || repeat > 100) throw std::invalid_argument("--repeat must be 1..100");
        std::vector<float> joined;
        std::string reference;
        for (int r = 0; r < repeat; ++r)
            for (size_t i = 0; i < files.size(); ++i) {
                joined.insert(joined.end(), audio[i].begin(), audio[i].end());
                joined.insert(joined.end(), 8000, 0.F);
                reference += reference_for(files[i]) + ' ';
            }
        captions::CaptionEngine engine(engine_config(a));
        std::string hypothesis;
        for (const auto& s : engine.transcribe_pcm(joined)) hypothesis += s.text + ' ';
        const auto e = models::edit_distance(models::tokenize(reference), models::tokenize(hypothesis));
        const auto& s = engine.stats();
        std::cout << std::fixed << std::setprecision(2) << "longform " << s.audio_seconds / 3600 << " h: WER "
                  << e.error_rate() * 100 << "% (S " << e.substitutions << " D " << e.deletions << " I "
                  << e.insertions << " / " << e.reference_length << ") in " << std::setprecision(1) << s.wall_seconds
                  << " s = " << std::setprecision(0) << s.audio_seconds / s.wall_seconds << "x real time\n";
        return 0;
    }
    if (live) {
        captions::StreamingOnnxAsr asr(live_config(a));
        const size_t packet = static_cast<size_t>(a.integer("--packet-ms", 100)) * 16;
        for (size_t i = 0; i < files.size(); ++i) {
            asr.reset();
            for (size_t p = 0; p < audio[i].size(); p += packet)
                asr.accept(std::span<const float>(audio[i]).subspan(p, std::min(packet, audio[i].size() - p)));
            asr.finish();
            hyp[i] = asr.text();
            compute += asr.stats().compute_seconds;
        }
        details = ",\n\"threads\":" + std::to_string(std::min(4, a.integer("--threads", 2))) +
                  ",\n\"packet_ms\":" + std::to_string(a.integer("--packet-ms", 100));
    } else {
        // Every file goes through the long-form segmenter; all segments of all
        // files are then decoded in one batched pass (the production path).
        captions::CaptionEngine engine(engine_config(a));
        std::vector<std::span<const float>> clips;
        std::vector<size_t> owner;
        for (size_t i = 0; i < audio.size(); ++i)
            for (const auto& s : captions::segment_audio(audio[i], engine.config().segmenter)) {
                clips.push_back(std::span<const float>(audio[i]).subspan(s.begin, s.end - s.begin));
                owner.push_back(i);
            }
        const auto results = engine.transcribe_clips(clips);
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].text.empty()) continue;
            if (!hyp[owner[i]].empty()) hyp[owner[i]] += ' ';
            hyp[owner[i]] += results[i].text;
        }
        compute = engine.stats().wall_seconds;
        const auto& c = engine.config();
        details = ",\n\"workers\":" + std::to_string(c.workers) + ",\n\"threads_per_worker\":" +
                  std::to_string(c.threads_per_worker) + ",\n\"batch\":" + std::to_string(c.batch) +
                  ",\n\"beam\":" + std::to_string(c.beam);
    }

    const fs::path report = a.get("--report");
    std::ofstream jsonl;
    if (!report.empty()) {
        if (report.has_parent_path()) fs::create_directories(report.parent_path());
        jsonl.open(report.string() + ".jsonl");
        if (!jsonl) throw std::runtime_error("Cannot write " + report.string() + ".jsonl");
    }
    models::EditCounts errors;
    for (size_t i = 0; i < files.size(); ++i) {
        const auto ref = reference_for(files[i]);
        const auto e = models::edit_distance(models::tokenize(ref), models::tokenize(hyp[i]));
        errors += e;
        if (jsonl) jsonl << "{\"file\":" << quote(files[i].filename().string()) << ",\"reference\":" << quote(ref)
                         << ",\"hypothesis\":" << quote(hyp[i]) << ",\"S\":" << e.substitutions << ",\"D\":"
                         << e.deletions << ",\"I\":" << e.insertions << ",\"N\":" << e.reference_length << "}\n";
    }
    std::cout << std::fixed << std::setprecision(2) << (live ? "live" : "batch") << ' ' << (dev ? "dev" : "test")
              << ", " << files.size() << " files: WER " << errors.error_rate() * 100 << "% (S " << errors.substitutions
              << " D " << errors.deletions << " I " << errors.insertions << " / " << errors.reference_length << "), "
              << std::setprecision(0) << audio_seconds / compute << "x real time\n";
    if (!report.empty()) {
        std::ofstream out(report);
        out << std::setprecision(10) << "{\n\"mode\":" << quote(live ? "live" : "batch")
            << ",\n\"split\":" << quote(dev ? "dev" : "test") << ",\n\"utterances\":" << files.size()
            << ",\n\"reference_words\":" << errors.reference_length << ",\n\"substitutions\":" << errors.substitutions
            << ",\n\"deletions\":" << errors.deletions << ",\n\"insertions\":" << errors.insertions
            << ",\n\"wer\":" << errors.error_rate() << ",\n\"audio_seconds\":" << audio_seconds
            << ",\n\"compute_seconds\":" << compute << ",\n\"times_real_time\":" << audio_seconds / compute << details
            << ",\n\"hardware_threads\":" << std::thread::hardware_concurrency() << "\n}\n";
        std::cout << "Saved " << report.string() << '\n';
    }
    return 0;
}

// ---- self-test -------------------------------------------------------------
int run_self_test(const Args& a) {
    auto check = [](bool ok, const std::string& what) { if (!ok) throw std::runtime_error("Self-test failed: " + what); };
    const fs::path data = a.get("--self-test");
    const auto files = frozen_split(data, true, 10);

    // Live engine contracts.
    captions::StreamingOnnxAsr live(live_config(a));
    std::vector<float> silence(16000 * 5, 0.F);
    live.accept(silence); live.finish();
    check(live.text().empty() && live.stats().encoder_calls == 0, "silence is gated, not decoded");
    bool rejected = false;
    try { live.accept(std::span<const float>(silence.data(), 1)); } catch (const std::logic_error&) { rejected = true; }
    check(rejected, "accept() after finish() is rejected");
    const auto first = captions::read_wav_mono(files.front(), 16000).samples;
    std::string expected;
    for (size_t chunk : {size_t{1600}, size_t{137}, size_t{2560}}) {
        live.reset();
        for (size_t i = 0; i < first.size(); i += chunk)
            live.accept(std::span<const float>(first).subspan(i, std::min(chunk, first.size() - i)));
        live.finish();
        if (expected.empty()) { expected = live.text(); check(!expected.empty(), "speech produces text"); }
        check(live.text() == expected, "packet boundaries do not change the transcript");
    }
    live.reset(); live.finish();
    check(live.text().empty(), "reset clears the transcript");

    // Batch == live: greedy, batch 1, ungated, clip by clip.
    auto cfg = live_config(a);
    cfg.energy_gate = false;
    captions::StreamingOnnxAsr ungated(cfg);
    auto ec = engine_config(a);
    ec.batch = 1; ec.beam = 1; ec.workers = 2;
    captions::CaptionEngine greedy(ec);
    std::vector<std::vector<float>> audio;
    for (const auto& f : files) audio.push_back(captions::read_wav_mono(f, 16000).samples);
    std::vector<std::span<const float>> clips(audio.begin(), audio.end());
    const auto results = greedy.transcribe_clips(clips);
    for (size_t i = 0; i < files.size(); ++i) {
        ungated.reset(); ungated.accept(audio[i]); ungated.finish();
        check(ungated.text() == results[i].text, "batch matches live on " + files[i].filename().string());
        check(!results[i].words.empty() && results[i].words.front().start >= 0 &&
              results[i].words.back().end <= clips[i].size() / 16000.0 + 1.0, "word timestamps inside the clip");
    }

    // Default engine (beam 4) end to end through the caption formats.
    captions::CaptionEngine engine(engine_config(a));
    const auto segments = engine.transcribe_pcm(first);
    const auto cues = captions::build_cues(segments);
    check(!cues.empty(), "cues from speech");
    for (size_t i = 0; i + 1 < cues.size(); ++i) check(cues[i].end <= cues[i + 1].start, "cues do not overlap");
    const auto srt = captions::render(captions::CaptionFormat::srt, segments, cues);
    check(srt.rfind("1\n00:00:", 0) == 0 && srt.find(" --> ") != std::string::npos, "SRT layout");
    check(captions::render(captions::CaptionFormat::vtt, segments, cues).rfind("WEBVTT\n\n", 0) == 0, "VTT header");
    check(engine.transcribe_pcm(silence).empty(), "silence produces no segments");
    std::cout << "Self-test passed: live contracts, batch/live parity on " << files.size()
              << " clips, beam-4 captions\n";
    return 0;
}
} // namespace

int main(int argc, char** argv) try {
    const Args a(argc, argv);
    if (argc == 1 || a.has("--help") || a.has("-h")) { std::cout << kUsage; return 0; }
    if (a.has("--self-test")) return run_self_test(a);
    if (a.has("--benchmark")) return run_benchmark(a);
    const auto mode = a.has("--mic") ? std::string("live") : a.get("--mode", "batch");
    if (mode != "batch" && mode != "live") throw std::invalid_argument("--mode must be batch or live");
    if (!a.has("--input") && !a.has("--mic")) throw std::invalid_argument("Give --input <media> or --mic (see --help)");
    if (a.has("--input") && a.has("--mic")) throw std::invalid_argument("Use either --input or --mic");
    return mode == "live" ? run_live(a) : run_batch(a);
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}
