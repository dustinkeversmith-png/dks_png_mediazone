// captions: one command line for the caption engines.
//   batch (default)  files and video: Parakeet TDT + Silero VAD, many times real time
//   live             microphone or file through the streaming Zipformer2
// Plus --benchmark/--eval (WER, optional noise mixing) and --self-test.
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
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {
constexpr const char* kUsage = R"(captions - speech-to-text captions on the CPU, offline

USAGE
  captions --input <media> [--format srt] [--out <path>]       caption a file (batch)
  captions --mode live --mic [--seconds N]                     live microphone captions
  captions --mode live --input <media> [--realtime]            live engine on a file
  captions --benchmark <librispeech-dir> [--mode batch|live] [--dev] [--limit N]
           [--longform [--repeat N]] [--report <file.json>]
  captions --eval <dir of NNN.wav + NNN.txt> [--noise <wav|dir> --snr <dB>] [--mode live --streams N]
                       (with --hotwords it also reports how many terms were recognized)
  captions --self-test <librispeech-dir>                       regression checks

MODELS
  batch   Parakeet TDT 110M (multi-domain English, punctuation + casing) with
          Silero VAD segmentation and hallucination filtering
  live    NeMo cache-aware streaming FastConformer (multi-domain English, greedy
          RNN-T, 560 ms chunks); text appears as audio arrives
  Fetch with src/captions/models/int8_zip/scripts/fetch_models.ps1

INPUT / OUTPUT
  --input <path>       any audio/video ffmpeg can read; PCM WAV is read directly
  --format <list>      srt | vtt | txt | json, comma-separated for several (default srt)
  --out <path>         output base path (default: input path without extension);
                       '-' writes to stdout (one format only)
  --upper              keep a model's raw uppercase instead of sentence case
  --line-chars N       caption line width, two lines per cue (default 42)
  --ffmpeg <path>      ffmpeg executable for non-WAV input

MODES
  --mode batch         (default) whole file at once
  --mode live          streaming engine
  --mic                capture from the default microphone (implies --mode live)
  --seconds N          stop microphone capture after N seconds (default: Ctrl+C)
  --live-model M       480ms (default, first words ~0.6 s) or 1040ms (~1.1 s, more
                       accurate on far-field/podcast audio; fetch_models.ps1 -Live1040)
  --no-agc             live: disable automatic gain control (on by default; boosts
                       audio quieter than --agc-quiet, default -30 dBFS, up to
                       --agc-target, default -23; louder audio is left untouched)
  --realtime           live mode on a file: feed audio at real-time pace

DOMAIN TERMS (batch and live)
  --hotwords <file>    names/jargon to favour, one per line ("term" or "term :boost";
                       # comments); spelled with the model's own subword pieces
  --hotword "a,b"      the same, inline
  --hotword-boost B    logit bonus per matching token (default 2)
  --hotword-start F    fraction of the bonus for a term's first token (default 0.25)

ROBUSTNESS (batch)
  --vad-threshold P    speech probability that opens a segment (default 0.02)
  --no-vad             segment with the RMS energy gate instead of the VAD
  --min-confidence C   hallucination filter: drop segments whose mean word
  --speech-floor P       confidence < C AND mean speech probability < P
                       (defaults 0.6 and 0.3; --min-confidence 0 disables)

TUNING
  --workers N          batch workers (default: one per hardware thread)
  --threads T          threads per batch worker (default 1); live encoder threads (default 2)
  --batch B            segments per encoder call (default: Parakeet 1, Zipformer2 16)
  --beam K             Zipformer2 beam width (default 4; 1 = greedy)
  --max-seg S          longest segment in seconds (default 20)
  --private-sessions   give each worker its own model copy (more memory, slower load)
  --packet-ms N        live packet size, 10..160 (default 100)
  --no-gate            live mode: decode silence too
  --models <dir>       another model directory (Parakeet/NeMo or Zipformer2 export)
  --verbose            time breakdown after batch captioning
  --quiet              no progress/statistics on stderr
)";

// Flags and which take a value; anything else is rejected.
const std::map<std::string, bool> kFlags = {
    {"--input", true}, {"--format", true}, {"--out", true}, {"--upper", false}, {"--line-chars", true},
    {"--ffmpeg", true}, {"--mode", true}, {"--mic", false}, {"--seconds", true}, {"--realtime", false},
    {"--threads", true}, {"--workers", true}, {"--batch", true}, {"--beam", true}, {"--max-seg", true},
    {"--gate-rms", true}, {"--packet-ms", true}, {"--no-gate", false}, {"--models", true}, {"--quiet", false},
    {"--benchmark", true}, {"--dev", false}, {"--limit", true}, {"--longform", false}, {"--repeat", true},
    {"--report", true}, {"--self-test", true}, {"--help", false}, {"-h", false},
    {"--private-sessions", false}, {"--fixed-seg", false}, {"--verbose", false},
    {"--eval", true}, {"--noise", true}, {"--snr", true}, {"--no-vad", false}, {"--vad-threshold", true},
    {"--dump-vad", false}, {"--min-confidence", true}, {"--speech-floor", true},
    {"--hotwords", true}, {"--hotword", true}, {"--hotword-boost", true},
    {"--hotword-start", true}, {"--streams", true}, {"--no-agc", false}, {"--agc-target", true},
    {"--live-model", true}, {"--agc-quiet", true}};

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

void report_hotwords(const captions::HotwordBiaser& h, bool quiet) {
    for (const auto& s : h.skipped())
        std::cerr << "warning: hotword \"" << s << "\" cannot be spelled with this model's vocabulary\n";
    if (!quiet && !h.empty()) std::cerr << "Biasing toward " << h.phrase_count() << " hotwords\n";
}

// Domain terms from --hotwords <file> and --hotword "a,b".
std::vector<captions::HotwordPhrase> hotword_list(const Args& a) {
    std::vector<captions::HotwordPhrase> terms;
    if (a.has("--hotwords")) terms = captions::read_hotwords(a.get("--hotwords"));
    if (a.has("--hotword")) for (auto& p : captions::parse_hotwords(a.get("--hotword"))) terms.push_back(p);
    return terms;
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
    c.share_sessions = !a.has("--private-sessions");
    c.adaptive_segments = !a.has("--fixed-seg");
    c.use_vad = !a.has("--no-vad");
    c.min_confidence = static_cast<float>(a.number("--min-confidence", c.min_confidence));
    c.speech_floor = static_cast<float>(a.number("--speech-floor", c.speech_floor));
    c.segmenter.vad_threshold = static_cast<float>(a.number("--vad-threshold", c.segmenter.vad_threshold));
    c.hotwords = hotword_list(a);
    c.hotword_boost = static_cast<float>(a.number("--hotword-boost", c.hotword_boost));
    c.hotword_start = static_cast<float>(a.number("--hotword-start", c.hotword_start));
    return c;
}

captions::StreamingAsrConfig live_config(const Args& a) {
    // --live-model picks the lookahead variant: 480ms (default, ~0.6 s to first
    // words) or 1040ms (~1.1 s, more accurate on far-field and podcast audio).
    fs::path dir = captions::live_model_dir();
    if (a.has("--live-model")) {
        const auto v = a.get("--live-model");
        if (v != "480ms" && v != "1040ms") throw std::invalid_argument("--live-model must be 480ms or 1040ms");
        dir = captions::model_root() / ("nemo-streaming-" + v);
    }
    const auto files = captions::model_files(a.has("--models") ? fs::path(a.get("--models")) : dir);
    captions::StreamingAsrConfig c;
    c.encoder = files.encoder; c.decoder = files.decoder; c.joiner = files.joiner; c.tokens = files.tokens;
    c.threads = std::min(4, a.integer("--threads", 2));
    c.packet_ms = a.integer("--packet-ms", 100);
    c.energy_gate = !a.has("--no-gate");
    c.gate_rms = static_cast<float>(a.number("--gate-rms", c.gate_rms));
    c.agc = !a.has("--no-agc");
    c.agc_target_db = static_cast<float>(a.number("--agc-target", c.agc_target_db));
    c.agc_quiet_db = static_cast<float>(a.number("--agc-quiet", c.agc_quiet_db));
    c.hotwords = hotword_list(a);
    c.hotword_boost = static_cast<float>(a.number("--hotword-boost", c.hotword_boost));
    c.hotword_start = static_cast<float>(a.number("--hotword-start", c.hotword_start));
    return c;
}

void print_batch_stats(const captions::BatchAsrStats& s, double extra_seconds) {
    std::cerr << std::fixed << std::setprecision(1) << "Captioned " << s.audio_seconds / 60 << " min of audio in "
              << s.wall_seconds + extra_seconds << " s (" << std::setprecision(0)
              << s.audio_seconds / (s.wall_seconds + extra_seconds) << "x real time; " << s.segments
              << " segments)\n";
}
void print_breakdown(const captions::BatchAsrStats& s, double setup_seconds) {
    std::cerr << std::fixed << std::setprecision(2) << "  setup (load + decode input) " << setup_seconds
              << " s, transcribe wall " << s.wall_seconds << " s; summed worker time: features "
              << s.fbank_seconds << " s, encoder " << s.encoder_seconds << " s, search " << s.search_seconds << " s\n";
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
    report_hotwords(engine.hotwords(), quiet);
    const auto audio = captions::load_audio(input, engine.config().ffmpeg);
    const double setup = since(t0);
    if (!quiet) std::cerr << "Decoded " << std::fixed << std::setprecision(1) << audio.size() / 16000.0
                          << " s of audio\n";
    if (a.has("--dump-vad")) {
        // Debug: per-second mean and max speech probability.
        captions::SileroVad vad(engine.config().vad);
        const auto p = vad.probabilities(audio);
        const size_t per_second = 16000 / captions::SileroVad::kChunk;
        for (size_t s = 0; s * per_second < p.size(); ++s) {
            const auto b = p.begin() + static_cast<std::ptrdiff_t>(s * per_second);
            const auto e = p.begin() + static_cast<std::ptrdiff_t>(std::min(p.size(), (s + 1) * per_second));
            double sum = 0;
            for (auto it = b; it != e; ++it) sum += *it;
            std::cout << s << ' ' << std::setprecision(2) << sum / static_cast<double>(e - b) << ' ' << *std::max_element(b, e) << '\n';
        }
        return 0;
    }
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
    if (a.has("--verbose")) print_breakdown(engine.stats(), setup);
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
    report_hotwords(asr.hotwords(), quiet);
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

// Any directory of NNN.wav + NNN.txt pairs (data/audio/eval/<set>), sorted.
std::vector<fs::path> eval_files(const fs::path& data, int limit) {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(data)) {
        if (e.path().extension() != ".wav") continue;
        auto txt = e.path(); txt.replace_extension(".txt");
        if (fs::exists(txt)) files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error("No wav+txt pairs in " + data.string());
    if (limit > 0 && static_cast<size_t>(limit) < files.size()) files.resize(limit);
    return files;
}

// Mixes real recorded noise into speech at a fixed SNR (speech power over the
// whole clip vs noise power). Noise files are cycled and the start offset is
// derived from the clip index, so every run mixes identically.
void add_noise(std::vector<float>& speech, size_t index, const std::vector<std::vector<float>>& noises, double snr_db) {
    const auto& noise = noises[index % noises.size()];
    if (noise.size() < 16000) throw std::runtime_error("Noise recording shorter than 1 s");
    size_t offset = (index * 7919 * 16000ULL) % noise.size();
    double ps = 0, pn = 0;
    for (size_t i = 0; i < speech.size(); ++i) {
        const double n = noise[(offset + i) % noise.size()];
        ps += speech[i] * static_cast<double>(speech[i]);
        pn += n * n;
    }
    if (pn <= 0 || ps <= 0) return;
    const double gain = std::sqrt(ps / (pn * std::pow(10.0, snr_db / 10.0)));
    for (size_t i = 0; i < speech.size(); ++i)
        speech[i] = static_cast<float>(std::clamp(speech[i] + gain * noise[(offset + i) % noise.size()], -1.0, 1.0));
}

int run_benchmark(const Args& a) {
    const bool generic = a.has("--eval");
    const fs::path data = generic ? a.get("--eval") : a.get("--benchmark");
    const bool live = a.get("--mode", "batch") == "live";
    const bool dev = a.has("--dev");
    const auto files = generic ? eval_files(data, a.integer("--limit", 0))
                               : frozen_split(data, dev, a.integer("--limit", 0));
    std::vector<std::vector<float>> audio;
    const std::string split_label = (generic ? data.filename().string() : std::string(dev ? "dev" : "test")) +
        (a.has("--noise") ? " +noise " + a.get("--snr", "10") + " dB" : std::string());
    for (const auto& f : files) audio.push_back(captions::read_wav_mono(f, 16000).samples);
    if (a.has("--noise")) {
        std::vector<std::vector<float>> noises;
        const fs::path source = a.get("--noise");
        if (fs::is_directory(source)) {
            std::vector<fs::path> paths;
            for (const auto& e : fs::directory_iterator(source)) if (e.path().extension() == ".wav") paths.push_back(e.path());
            std::sort(paths.begin(), paths.end());
            for (const auto& p : paths) noises.push_back(captions::read_wav_mono(p, 16000).samples);
        } else noises.push_back(captions::read_wav_mono(source, 16000).samples);
        if (noises.empty()) throw std::runtime_error("No noise recordings in " + source.string());
        const double snr = a.number("--snr", 10);
        for (size_t i = 0; i < audio.size(); ++i) add_noise(audio[i], i, noises, snr);
    }
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
        // Independent live streams in parallel (one engine each) to finish the
        // benchmark sooner; speed is still reported per stream.
        const size_t packet = static_cast<size_t>(a.integer("--packet-ms", 100)) * 16;
        const int streams = std::max(1, a.integer("--streams", 6));
        std::vector<double> stream_compute(static_cast<size_t>(streams), 0.0);
        std::atomic<size_t> next{0};
        std::exception_ptr failure;
        std::mutex failure_mutex;
        std::vector<std::thread> threads;
        for (int s = 0; s < streams; ++s)
            threads.emplace_back([&, s] {
                try {
                    captions::StreamingOnnxAsr asr(live_config(a));
                    for (size_t i; (i = next.fetch_add(1)) < files.size();) {
                        asr.reset();
                        for (size_t p = 0; p < audio[i].size(); p += packet)
                            asr.accept(std::span<const float>(audio[i]).subspan(p, std::min(packet, audio[i].size() - p)));
                        asr.finish();
                        hyp[i] = asr.text();
                        stream_compute[static_cast<size_t>(s)] += asr.stats().compute_seconds;
                    }
                } catch (...) { std::lock_guard lock(failure_mutex); failure = std::current_exception(); next = files.size(); }
            });
        for (auto& t : threads) t.join();
        if (failure) std::rethrow_exception(failure);
        for (double c : stream_compute) compute += c;
        details = ",\n\"threads\":" + std::to_string(std::min(4, a.integer("--threads", 2))) +
                  ",\n\"packet_ms\":" + std::to_string(a.integer("--packet-ms", 100));
    } else {
        // The production path: every file is VAD-segmented, all segments of
        // all files are decoded in one batched pass, then filtered.
        captions::CaptionEngine engine(engine_config(a));
        const auto results = engine.transcribe_batch(std::vector<std::span<const float>>(audio.begin(), audio.end()));
        for (size_t i = 0; i < results.size(); ++i)
            for (const auto& s : results[i]) {
                if (!hyp[i].empty()) hyp[i] += ' ';
                hyp[i] += s.text;
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
    // Hotword recall: of the term occurrences in the references, how many the
    // hypothesis also contains (normalized token sequences, per utterance).
    std::vector<std::vector<std::string>> terms;
    for (const auto& p : hotword_list(a)) if (auto t = models::tokenize(p.text); !t.empty()) terms.push_back(t);
    auto occurrences = [](const std::vector<std::string>& words, const std::vector<std::string>& term) {
        size_t n = 0;
        for (size_t i = 0; i + term.size() <= words.size(); ++i)
            if (std::equal(term.begin(), term.end(), words.begin() + static_cast<std::ptrdiff_t>(i))) ++n;
        return n;
    };
    size_t term_total = 0, term_found = 0, term_extra = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        const auto ref = reference_for(files[i]);
        const auto ref_words = models::tokenize(ref), hyp_words = models::tokenize(hyp[i]);
        const auto e = models::edit_distance(ref_words, hyp_words);
        for (const auto& term : terms) {
            const size_t r = occurrences(ref_words, term), h = occurrences(hyp_words, term);
            term_total += r; term_found += std::min(r, h); term_extra += h > r ? h - r : 0;
        }
        errors += e;
        if (jsonl) jsonl << "{\"file\":" << quote(files[i].filename().string()) << ",\"reference\":" << quote(ref)
                         << ",\"hypothesis\":" << quote(hyp[i]) << ",\"S\":" << e.substitutions << ",\"D\":"
                         << e.deletions << ",\"I\":" << e.insertions << ",\"N\":" << e.reference_length << "}\n";
    }
    std::cout << std::fixed << std::setprecision(2) << (live ? "live" : "batch") << ' ' << split_label
              << ", " << files.size() << " files: WER " << errors.error_rate() * 100 << "% (S " << errors.substitutions
              << " D " << errors.deletions << " I " << errors.insertions << " / " << errors.reference_length << "), "
              << std::setprecision(0) << audio_seconds / compute << "x real time\n";
    if (!terms.empty())
        std::cout << "  hotword recall " << term_found << " / " << term_total << " (" << std::setprecision(1)
                  << (term_total ? 100.0 * static_cast<double>(term_found) / static_cast<double>(term_total) : 0.0)
                  << "%), " << term_extra << " extra occurrences\n";
    if (!report.empty()) {
        std::ofstream out(report);
        out << std::setprecision(10) << "{\n\"mode\":" << quote(live ? "live" : "batch")
            << ",\n\"split\":" << quote(split_label) << ",\n\"utterances\":" << files.size()
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

    // WER normalizer: spelling conventions must not count as errors.
    auto same = [&](const std::string& x, const std::string& y) {
        check(models::tokenize(x) == models::tokenize(y), "normalizer: '" + x + "' == '" + y + "'");
    };
    same("in nineteen seventy five", "in 1975");
    same("one thousand nine hundred and seventy five", "1975");
    same("two thousand and seventeen", "2017");
    same("twenty twenty", "2020");
    same("nineteen o five", "1905");
    same("twenty-one years", "twenty one years");
    same("Mr. Smith's car, uh, cost $5!", "MISTER SMITH'S CAR COST FIVE DOLLARS");
    same("about 20% of 1,500", "about twenty percent of fifteen hundred");
    same("the 3rd and 21st", "the third and twenty first");
    same("three point five", "3.5");
    same("<unk> one of them", "1 of them");
    check(models::tokenize("seven hundred") != models::tokenize("seven"), "normalizer keeps distinct numbers apart");

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

    // Batch == live for the streaming Zipformer2 (both engines run the same
    // chunked algorithm): greedy, batch 1, ungated, clip by clip. NeMo
    // streaming exports have no batch counterpart, so this uses models/librispeech.
    const fs::path zipformer = captions::model_root() / "librispeech";
    size_t parity = 0;
    if (fs::exists(zipformer / "tokens.txt")) {
        const auto zf = captions::model_files(zipformer);
        auto cfg = live_config(a);
        cfg.encoder = zf.encoder; cfg.decoder = zf.decoder; cfg.joiner = zf.joiner; cfg.tokens = zf.tokens;
        cfg.energy_gate = false;
        cfg.agc = false;  // the batch engine has no gain control: compare raw audio
        cfg.hotwords.clear();
        captions::StreamingOnnxAsr ungated(cfg);
        auto ec = engine_config(a);
        ec.models = zipformer;
        ec.batch = 1; ec.beam = 1; ec.workers = 2; ec.use_vad = false; ec.min_confidence = 0; ec.hotwords.clear();
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
        parity = files.size();
    } else {
        std::cout << "Skipping batch/live parity: " << zipformer.string() << " not present\n";
    }

    // Hotwords: a term spelled with the model's own pieces must be tracked
    // through the prefix tree, and an empty list must change nothing.
    {
        const std::vector<std::string> pieces = {"<blk>", "\xE2\x96\x81" "ku", "ber", "net", "es", "\xE2\x96\x81" "the"};
        captions::HotwordBiaser h(captions::parse_hotwords("kubernetes"), pieces, 2.0F);
        check(h.phrase_count() == 1 && h.skipped().empty(), "hotword spelled from vocabulary pieces");
        std::vector<float> logits(pieces.size(), 0.F);
        h.bias(0, logits.data(), logits.size());
        check(logits[1] == 0.5F && logits[2] == 0.F, "hotword start token boosted (scaled) from the root");
        int s = h.advance(0, 1);
        std::fill(logits.begin(), logits.end(), 0.F);
        h.bias(s, logits.data(), logits.size());
        check(logits[2] == 2.0F, "hotword continuation boosted");
        check(h.advance(s, 5) == 0, "a non-matching token resets the match");
        check(captions::HotwordBiaser(captions::parse_hotwords("zzz \xC3\xA9"), pieces, 2.0F).skipped().size() == 1,
              "unspellable hotword reported");
    }

    // Default batch engine end to end through the caption formats.
    captions::CaptionEngine engine(engine_config(a));
    const auto segments = engine.transcribe_pcm(first);
    const auto cues = captions::build_cues(segments);
    check(!cues.empty(), "cues from speech");
    for (size_t i = 0; i + 1 < cues.size(); ++i) check(cues[i].end <= cues[i + 1].start, "cues do not overlap");
    const auto srt = captions::render(captions::CaptionFormat::srt, segments, cues);
    check(srt.rfind("1\n00:00:", 0) == 0 && srt.find(" --> ") != std::string::npos, "SRT layout");
    check(captions::render(captions::CaptionFormat::vtt, segments, cues).rfind("WEBVTT\n\n", 0) == 0, "VTT header");
    check(engine.transcribe_pcm(silence).empty(), "silence produces no segments");
    std::cout << "Self-test passed: normalizer, live contracts, hotwords, batch/live parity on " << parity
              << " clips, beam-4 captions\n";
    return 0;
}
} // namespace

int main(int argc, char** argv) try {
    const Args a(argc, argv);
    if (argc == 1 || a.has("--help") || a.has("-h")) { std::cout << kUsage; return 0; }
    if (a.has("--self-test")) return run_self_test(a);
    if (a.has("--benchmark") || a.has("--eval")) return run_benchmark(a);
    const auto mode = a.has("--mic") ? std::string("live") : a.get("--mode", "batch");
    if (mode != "batch" && mode != "live") throw std::invalid_argument("--mode must be batch or live");
    if (!a.has("--input") && !a.has("--mic")) throw std::invalid_argument("Give --input <media> or --mic (see --help)");
    if (a.has("--input") && a.has("--mic")) throw std::invalid_argument("Use either --input or --mic");
    return mode == "live" ? run_live(a) : run_batch(a);
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}
