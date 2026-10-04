// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\app\caption_batch.cpp ===
// Offline, many-times-real-time captioning of recorded audio/video with the
// INT8 Zipformer. See models/int8_zip/BATCH_ASR.md.
#include <captions/batch_asr.hpp>
#include <captions/streaming_asr.hpp>
#include <input/wav_reader.hpp>
#include <benchmarks/scoring.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
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
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
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

// Any container/codec ffmpeg understands, decoded straight to 16 kHz mono
// float through a pipe. WAV files skip ffmpeg entirely.
fs::path find_ffmpeg(const std::string& requested) {
    if (!requested.empty()) return requested;
    for (fs::path dir = fs::current_path(); !dir.empty(); dir = dir.parent_path()) {
        const auto candidate = dir / "dependencies/ffmpeg/bin/ffmpeg.exe";
        if (fs::exists(candidate)) return candidate;
        if (dir == dir.parent_path()) break;
    }
    return "ffmpeg";
}
std::vector<float> load_audio(const fs::path& path, const std::string& ffmpeg_arg) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".wav") {
        try { return captions::read_wav_mono(path, 16000).samples; }
        catch (const std::exception&) { /* compressed/extensible WAV: fall through to ffmpeg */ }
    }
    const auto ffmpeg = find_ffmpeg(ffmpeg_arg);
    const std::string command = "\"\"" + ffmpeg.string() + "\" -nostdin -v error -i \"" + path.string() +
                                "\" -vn -ac 1 -ar 16000 -f f32le -\"";
#ifdef _WIN32
    FILE* pipe = _popen(command.c_str(), "rb");
#else
    FILE* pipe = popen(command.c_str(), "r");
#endif
    if (!pipe) throw std::runtime_error("Cannot start ffmpeg");
    std::vector<float> samples;
    std::vector<float> buffer(1 << 16);
    size_t n;
    while ((n = std::fread(buffer.data(), sizeof(float), buffer.size(), pipe)) > 0)
        samples.insert(samples.end(), buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(n));
#ifdef _WIN32
    const int status = _pclose(pipe);
#else
    const int status = pclose(pipe);
#endif
    if (status != 0 || samples.empty())
        throw std::runtime_error("ffmpeg could not decode " + path.string() + " (use --ffmpeg <path>)");
    return samples;
}

// The model emits uppercase without punctuation; captions read better in
// sentence case with the pronoun "I" restored.
std::string display_word(std::string w, bool capitalize) {
    std::transform(w.begin(), w.end(), w.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (w == "i" || w.rfind("i'", 0) == 0 || capitalize) if (!w.empty()) w[0] = static_cast<char>(std::toupper(w[0]));
    return w;
}
struct Cue { double start, end; std::string text; };
std::vector<Cue> build_cues(const std::vector<captions::SegmentResult>& segments, size_t line_chars,
                            double max_seconds, bool upper) {
    std::vector<Cue> cues;
    const size_t cue_chars = line_chars * 2;
    for (const auto& seg : segments) {
        bool sentence_start = true;
        std::vector<const captions::Word*> current;
        size_t chars = 0;
        auto flush = [&] {
            if (current.empty()) return;
            std::vector<std::string> words;
            for (const auto* w : current) {
                words.push_back(upper ? w->text : display_word(w->text, sentence_start));
                sentence_start = false;
            }
            // Two balanced lines when the cue does not fit on one.
            std::string text;
            size_t total = 0;
            for (const auto& w : words) total += w.size() + 1;
            size_t line = 0;
            bool broke = total <= line_chars + 1;
            for (const auto& w : words) {
                if (!text.empty()) {
                    if (!broke && line + w.size() + 1 > total / 2 + 1) { text += '\n'; line = 0; broke = true; }
                    else { text += ' '; ++line; }
                }
                text += w;
                line += w.size();
            }
            cues.push_back({current.front()->start, current.back()->end, text});
            current.clear();
            chars = 0;
        };
        for (const auto& w : seg.words) {
            const bool gap = !current.empty() && w.start - current.back()->end > 0.8;
            const bool full = !current.empty() && (chars + w.text.size() + 1 > cue_chars ||
                                                  w.end - current.front()->start > max_seconds);
            if (gap || full) { flush(); if (gap) sentence_start = true; }
            current.push_back(&w);
            chars += w.text.size() + 1;
        }
        flush();
    }
    // Hold each cue on screen a little past its last word, never into the next.
    for (size_t i = 0; i < cues.size(); ++i) {
        double end = std::max(cues[i].end + 0.4, cues[i].start + 0.7);
        if (i + 1 < cues.size()) end = std::min(end, cues[i + 1].start);
        cues[i].end = std::max(end, cues[i].start + 0.001);
    }
    return cues;
}
std::string stamp(double t, char separator) {
    const auto ms = static_cast<long long>(std::llround(std::max(0.0, t) * 1000));
    std::ostringstream s;
    s << std::setfill('0') << std::setw(2) << ms / 3600000 << ':' << std::setw(2) << ms / 60000 % 60 << ':'
      << std::setw(2) << ms / 1000 % 60 << separator << std::setw(3) << ms % 1000;
    return s.str();
}
void write_outputs(const std::vector<captions::SegmentResult>& segments, const std::vector<Cue>& cues,
                   const fs::path& base, const std::string& formats) {
    auto open = [&](const char* ext) {
        fs::path p = base; p += ext;
        std::ofstream out(p, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write " + p.string());
        std::cerr << "Wrote " << p.string() << '\n';
        return out;
    };
    if (formats.find("srt") != std::string::npos) {
        auto out = open(".srt");
        for (size_t i = 0; i < cues.size(); ++i)
            out << i + 1 << '\n' << stamp(cues[i].start, ',') << " --> " << stamp(cues[i].end, ',') << '\n'
                << cues[i].text << "\n\n";
    }
    if (formats.find("vtt") != std::string::npos) {
        auto out = open(".vtt");
        out << "WEBVTT\n\n";
        for (const auto& c : cues) out << stamp(c.start, '.') << " --> " << stamp(c.end, '.') << '\n' << c.text << "\n\n";
    }
    if (formats.find("txt") != std::string::npos) {
        auto out = open(".txt");
        for (const auto& s : segments) if (!s.text.empty()) out << s.text << '\n';
    }
    if (formats.find("json") != std::string::npos) {
        auto out = open(".json");
        out << std::fixed << std::setprecision(3) << "[\n";
        for (size_t i = 0; i < segments.size(); ++i) {
            const auto& s = segments[i];
            out << "{\"start\":" << s.start << ",\"end\":" << s.end << ",\"text\":" << quote(s.text) << ",\"words\":[";
            for (size_t j = 0; j < s.words.size(); ++j)
                out << (j ? "," : "") << "[" << quote(s.words[j].text) << ',' << s.words[j].start << ',' << s.words[j].end << ']';
            out << "]}" << (i + 1 < segments.size() ? ",\n" : "\n");
        }
        out << "]\n";
    }
}

std::vector<fs::path> frozen_split(const fs::path& data, bool dev, int limit) {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(data)) if (e.path().extension() == ".wav") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    // Same frozen split as caption-streaming: first quarter dev, rest test.
    if (files.size() != 600) throw std::runtime_error("Expected the frozen 600-WAV evaluation corpus");
    if (dev) files.resize(files.size() / 4); else files.erase(files.begin(), files.begin() + files.size() / 4);
    if (limit < 0) throw std::invalid_argument("Negative limit");
    if (limit > 0 && static_cast<size_t>(limit) < files.size()) files.resize(limit);
    return files;
}
std::string reference_for(const fs::path& wav) { auto p = wav; p.replace_extension(".txt"); return read(p); }

void print_stats(const captions::BatchOnnxAsr& asr, double extra_seconds = 0) {
    const auto& s = asr.stats();
    std::cerr << std::setprecision(3) << "audio " << s.audio_seconds << " s, decoded " << s.decoded_seconds
              << " s in " << s.segments << " segments / " << s.batches << " batches; wall " << s.wall_seconds
              << " s = " << s.audio_seconds / s.wall_seconds << "x real time";
    if (extra_seconds > 0) std::cerr << " (" << s.audio_seconds / (s.wall_seconds + extra_seconds) << "x incl. load/decode)";
    std::cerr << "\nworker time " << s.worker_seconds << " s: fbank " << s.fbank_seconds << ", encoder " << s.encoder_seconds
              << ", search " << s.search_seconds;
    std::cerr << '\n';
}

// Batch clip transcripts must equal the streaming engine's for the same audio
// when each clip is one ungated segment decoded alone (batch 1).
void parity_test(const captions::BatchAsrConfig& base, const fs::path& data) {
    auto files = frozen_split(data, true, 10);
    captions::StreamingAsrConfig sc;
    sc.encoder = base.encoder; sc.decoder = base.decoder; sc.joiner = base.joiner; sc.tokens = base.tokens;
    sc.energy_gate = false;
    captions::StreamingOnnxAsr streaming(sc);
    auto one = base; one.batch = 1; one.workers = 2; one.beam = 1;
    captions::BatchOnnxAsr batch(one);
    std::vector<std::vector<float>> audio;
    for (const auto& f : files) audio.push_back(captions::read_wav_mono(f, 16000).samples);
    std::vector<std::span<const float>> clips(audio.begin(), audio.end());
    const auto results = batch.transcribe(clips);
    for (size_t i = 0; i < files.size(); ++i) {
        streaming.reset();
        streaming.accept(audio[i]);
        streaming.finish();
        if (streaming.text() != results[i].text)
            throw std::runtime_error("Batch/streaming mismatch on " + files[i].filename().string() +
                                     "\n  streaming: " + streaming.text() + "\n  batch:     " + results[i].text);
        if (results[i].words.empty() || results[i].words.front().start < 0 ||
            results[i].words.back().end > clips[i].size() / 16000.0 + 1.0)
            throw std::runtime_error("Word timestamps outside the clip");
    }
    std::cout << "Batch/streaming parity checks passed (" << files.size() << " clips)\n";
}
} // namespace

int main(int argc, char** argv) try {
    std::cout << std::unitbuf;
    if (argc == 1 || flag(argc, argv, "--help")) {
        std::cout <<
            "caption-batch --input <media> [--out base] [--formats srt,vtt,txt,json]\n"
            "caption-batch --benchmark <librispeech-dir> [--dev] [--limit N] [--report file.json]\n"
            "caption-batch --longform <librispeech-dir> [--repeat N]   (test split as one recording)\n"
            "caption-batch --parity <librispeech-dir>                  (batch == streaming check)\n"
            "  --models <dir> (default src/captions/models/int8_zip/models/compact)\n"
            "  --workers N (0=auto) --threads T (per worker, default 2) --batch B (default 16)\n"
            "  --beam K (default 4; 1 = greedy, bit-identical to caption-streaming) --spin\n"
            "  --max-seg S (20) --min-seg S (8) --gate-rms R (0.0003) --line-chars 42 --upper\n"
            "  --ffmpeg <path to ffmpeg for non-WAV input>\n";
        return 0;
    }
    const fs::path dir = arg(argc, argv, "--models", "src/captions/models/int8_zip/models/compact");
    captions::BatchAsrConfig c;
    c.encoder = graph(dir, "encoder"); c.decoder = graph(dir, "decoder"); c.joiner = graph(dir, "joiner");
    c.tokens = dir / "tokens.txt";
    c.workers = std::stoi(arg(argc, argv, "--workers", "0"));
    c.threads_per_worker = std::stoi(arg(argc, argv, "--threads", "2"));
    c.batch = std::stoi(arg(argc, argv, "--batch", "16"));
    c.spin = flag(argc, argv, "--spin");
    c.beam = std::stoi(arg(argc, argv, "--beam", "4"));
    captions::SegmenterConfig seg;
    seg.max_seconds = std::stod(arg(argc, argv, "--max-seg", "20"));
    seg.min_seconds = std::stod(arg(argc, argv, "--min-seg", "8"));
    seg.gate_rms = std::stof(arg(argc, argv, "--gate-rms", "0.0003"));

    if (auto p = arg(argc, argv, "--parity"); !p.empty()) { parity_test(c, p); return 0; }

    const auto load_start = Clock::now();
    captions::BatchOnnxAsr asr(c);
    const double load_seconds = since(load_start);
    std::cerr << "Model " << asr.model_type() << ": " << asr.workers() << " workers x " << c.threads_per_worker
              << " threads, batch " << c.batch << ", loaded in " << std::setprecision(3) << load_seconds << " s\n";

    if (auto input = arg(argc, argv, "--input"); !input.empty()) {
        const auto t0 = Clock::now();
        const auto audio = load_audio(input, arg(argc, argv, "--ffmpeg"));
        const double decode_seconds = since(t0);
        std::cerr << "Decoded " << audio.size() / 16000.0 << " s of audio in " << decode_seconds << " s\n";
        const auto segments = asr.transcribe_long(audio, seg);
        const auto cues = build_cues(segments, std::stoul(arg(argc, argv, "--line-chars", "42")), 6.0,
                                     flag(argc, argv, "--upper"));
        fs::path base = arg(argc, argv, "--out");
        if (base.empty()) { base = input; base.replace_extension(); }
        write_outputs(segments, cues, base, arg(argc, argv, "--formats", "srt,txt"));
        print_stats(asr, load_seconds + decode_seconds);
        return 0;
    }

    if (auto data = arg(argc, argv, "--longform"); !data.empty()) {
        // The 450 test utterances back to back with 0.5 s gaps: one ~1.1 h
        // recording (x --repeat), scored as a single word sequence.
        const auto files = frozen_split(data, false, 0);
        const int repeat = std::stoi(arg(argc, argv, "--repeat", "1"));
        if (repeat < 1 || repeat > 100) throw std::invalid_argument("--repeat must be 1..100");
        std::vector<float> audio;
        std::string reference;
        for (int r = 0; r < repeat; ++r)
            for (const auto& f : files) {
                const auto wav = captions::read_wav_mono(f, 16000);
                audio.insert(audio.end(), wav.samples.begin(), wav.samples.end());
                audio.insert(audio.end(), 8000, 0.F);
                reference += reference_for(f) + ' ';
            }
        const auto segments = asr.transcribe_long(audio, seg);
        std::string hypothesis;
        for (const auto& s : segments) hypothesis += s.text + ' ';
        const auto e = models::edit_distance(models::tokenize(reference), models::tokenize(hypothesis));
        std::cout << std::setprecision(4) << "longform WER " << e.error_rate() * 100 << "% (S " << e.substitutions
                  << " D " << e.deletions << " I " << e.insertions << " / " << e.reference_length << " words, "
                  << segments.size() << " segments)\n";
        print_stats(asr);
        return 0;
    }

    const auto data = arg(argc, argv, "--benchmark");
    if (data.empty()) throw std::invalid_argument("Select --input, --benchmark, --longform, or --parity");
    const auto files = frozen_split(data, flag(argc, argv, "--dev"), std::stoi(arg(argc, argv, "--limit", "0")));
    // Every file goes through the long-form segmenter, then all segments of
    // all files are decoded in one pass (the production path).
    std::vector<std::vector<float>> audio;
    std::vector<std::span<const float>> clips;
    std::vector<size_t> owner;
    for (const auto& f : files) audio.push_back(captions::read_wav_mono(f, 16000).samples);
    for (size_t i = 0; i < audio.size(); ++i)
        for (const auto& s : captions::segment_audio(audio[i], seg)) {
            clips.push_back(std::span<const float>(audio[i]).subspan(s.begin, s.end - s.begin));
            owner.push_back(i);
        }
    const auto results = asr.transcribe(clips);
    std::vector<std::string> hyp(files.size());
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].text.empty()) continue;
        if (!hyp[owner[i]].empty()) hyp[owner[i]] += ' ';
        hyp[owner[i]] += results[i].text;
    }
    models::EditCounts errors;
    double audio_seconds = 0;
    fs::path report = arg(argc, argv, "--report");
    std::ofstream jsonl;
    if (!report.empty()) {
        if (report.has_parent_path()) fs::create_directories(report.parent_path());
        jsonl.open(report.string() + ".jsonl");
    }
    for (size_t i = 0; i < files.size(); ++i) {
        const auto ref = reference_for(files[i]);
        const auto e = models::edit_distance(models::tokenize(ref), models::tokenize(hyp[i]));
        errors += e;
        audio_seconds += audio[i].size() / 16000.0;
        if (jsonl) jsonl << "{\"file\":" << quote(files[i].filename().string()) << ",\"reference\":" << quote(ref)
                         << ",\"hypothesis\":" << quote(hyp[i]) << ",\"S\":" << e.substitutions << ",\"D\":"
                         << e.deletions << ",\"I\":" << e.insertions << ",\"N\":" << e.reference_length << "}\n";
    }
    const auto& s = asr.stats();
    std::cout << std::setprecision(4) << files.size() << " files, WER " << errors.error_rate() * 100 << "% (S "
              << errors.substitutions << " D " << errors.deletions << " I " << errors.insertions << " / "
              << errors.reference_length << ")\n";
    print_stats(asr);
    if (!report.empty()) {
        std::ofstream out(report);
        out << std::setprecision(10) << "{\n\"split\":" << quote(flag(argc, argv, "--dev") ? "dev" : "test")
            << ",\n\"utterances\":" << files.size() << ",\n\"reference_words\":" << errors.reference_length
            << ",\n\"substitutions\":" << errors.substitutions << ",\n\"deletions\":" << errors.deletions
            << ",\n\"insertions\":" << errors.insertions << ",\n\"wer\":" << errors.error_rate()
            << ",\n\"audio_seconds\":" << audio_seconds << ",\n\"wall_seconds\":" << s.wall_seconds
            << ",\n\"times_real_time\":" << audio_seconds / s.wall_seconds
            << ",\n\"worker_seconds\":" << s.worker_seconds << ",\n\"segments\":" << s.segments
            << ",\n\"batches\":" << s.batches << ",\n\"encoder_calls\":" << s.encoder_calls
            << ",\n\"model_load_seconds\":" << load_seconds << ",\n\"workers\":" << asr.workers()
            << ",\n\"threads_per_worker\":" << c.threads_per_worker << ",\n\"batch\":" << c.batch
            << ",\n\"hardware_threads\":" << std::thread::hardware_concurrency()
            << ",\n\"models\":" << quote(fs::absolute(dir).string()) << "\n}\n";
        std::cout << "Saved " << report.string() << '\n';
    }
    return 0;
} catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\app\caption_streaming.cpp ===
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
void decode(captions::StreamingOnnxAsr& asr, std::span<const float> samples, size_t chunk,
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
void microphone(captions::StreamingOnnxAsr& asr, int seconds_limit) {
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
void self_test(captions::StreamingOnnxAsr& asr, const fs::path& wav) {
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
    auto audio = captions::read_wav_mono(wav,16000);
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
    const fs::path dir = arg(argc,argv,"--models","src/captions/models/int8_zip/models/compact");
    captions::StreamingAsrConfig c;
    c.encoder=graph(dir,"encoder"); c.decoder=graph(dir,"decoder"); c.joiner=graph(dir,"joiner"); c.tokens=dir/"tokens.txt";
    c.threads=std::stoi(arg(argc,argv,"--threads","2"));
    c.packet_ms=std::stoi(arg(argc,argv,"--packet-ms","100"));
    c.energy_gate=!flag(argc,argv,"--no-gate");
    c.gate_rms=std::stof(arg(argc,argv,"--gate-rms","0.0003"));
    auto load_start=Clock::now();
    captions::StreamingOnnxAsr asr(c);
    const double load_seconds=std::chrono::duration<double>(Clock::now()-load_start).count();
    std::cerr << "Model: " << asr.model_type() << ", native shift " << asr.model_chunk_ms()
              << " ms, initial feature window " << asr.first_window_ms() << " ms, packet " << c.packet_ms << " ms\n";
    if (flag(argc,argv,"--mic")) { microphone(asr,std::stoi(arg(argc,argv,"--seconds","0"))); return 0; }
    if (auto p=arg(argc,argv,"--self-test"); !p.empty()) { self_test(asr,p); return 0; }
    if (auto p=arg(argc,argv,"--file"); !p.empty()) {
        auto audio=captions::read_wav_mono(p,16000);
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
        auto wav=captions::read_wav_mono(files[i],16000);
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

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\benchmarks\benchmark_models.cpp ===
// End-to-end evaluation of the networkless captioning stack against the three
// downloaded tiers, alongside the two legacy frame-level baselines.
//
//   Tier 1  data/audio/digits       isolated digits   -> accuracy
//   Tier 2  data/audio/timit        phone recognition -> phone error rate
//   Tier 3  data/audio/librispeech  captioning        -> word error rate
//
// Every tier reports speed as a real-time factor (RTF = compute seconds per
// audio second) measured on one CPU thread, so accuracy and cost are always
// read together.
//
// Usage:
//   benchmark_models [--tiers digits,timit,librispeech] [--models data/models]
//                    [--limit N] [--templates-per-digit K] [--dev]
//                    [--acoustic-scale F] [--beam F] [--max-active N]
//                    [--word-penalty F] [--vocab-lexicon 1]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "audio_loadnorm.hpp"

#include <toy_pruned_hmm/acoustic_model.hpp>
#include <naive/baseline_frontends.hpp>
#include <toy_pruned_hmm/dtw/dtw.hpp>
#include <toy_pruned_hmm/phonemes/lexicon.hpp>
#include <filter/mfcc.hpp>
#include <toy_pruned_hmm/ngram/ngram_lm.hpp>
#include <toy_pruned_hmm/phonemes/phone_set.hpp>
#include <benchmarks/scoring.hpp>
#include <toy_pruned_hmm/triphone/triphone.hpp>
#include <toy_pruned_hmm/viterbi_decoder.hpp>

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

std::vector<std::string> split_csv(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) parts.push_back(item);
    }
    return parts;
}

std::string read_file(const fs::path& path) {
    std::ifstream file(path);
    if (!file) return {};
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::string json_string_field(const std::string& content, const std::string& key) {
    const size_t at = content.find("\"" + key + "\"");
    if (at == std::string::npos) return {};
    const size_t colon = content.find(':', at);
    if (colon == std::string::npos) return {};
    const size_t first = content.find('"', colon);
    if (first == std::string::npos) return {};
    const size_t last = content.find('"', first + 1);
    if (last == std::string::npos) return {};
    return content.substr(first + 1, last - first - 1);
}

void print_header(const std::string& title) {
    std::cout << "\n======================================================================\n";
    std::cout << title << "\n";
    std::cout << "======================================================================\n";
}

// ---------------------------------------------------------------- Tier 1

struct DigitClip {
    fs::path path;
    std::string label;
};

void run_digits(const fs::path& data_root, int templates_per_digit, int test_per_digit,
                bool run_baselines, int neighbours, int radius, bool use_endpoint) {
    print_header("TIER 1  Isolated digits  -  MFCC + DTW template matching");

    const fs::path digits_dir = data_root / "audio" / "digits";
    if (!fs::exists(digits_dir)) {
        std::cout << "[SKIP] " << digits_dir << " not found\n";
        return;
    }

    std::vector<DigitClip> train, test;
    for (int digit = 0; digit < 10; ++digit) {
        const fs::path class_dir = digits_dir / std::to_string(digit);
        if (!fs::exists(class_dir)) continue;
        std::vector<fs::path> clips;
        for (const auto& entry : fs::directory_iterator(class_dir)) {
            if (entry.path().extension() == ".wav") clips.push_back(entry.path());
        }
        std::sort(clips.begin(), clips.end());
        // Deterministic shuffle so the split does not track file order.
        std::mt19937 rng(1234 + digit);
        std::shuffle(clips.begin(), clips.end(), rng);

        const std::string label = std::to_string(digit);
        for (size_t i = 0; i < clips.size(); ++i) {
            if (static_cast<int>(i) < templates_per_digit) {
                train.push_back({clips[i], label});
            } else if (static_cast<int>(i) < templates_per_digit + test_per_digit) {
                test.push_back({clips[i], label});
            }
        }
    }

    if (train.empty() || test.empty()) {
        std::cout << "[SKIP] not enough digit clips (train=" << train.size()
                  << ", test=" << test.size() << ")\n";
        return;
    }

    std::cout << "Templates: " << train.size() << "   Test clips: " << test.size() << "\n";
    std::cout << "Note: this corpus carries no speaker ids, so the split is clip-disjoint\n"
              << "      but not speaker-disjoint.\n\n";

    models::MfccExtractor extractor;
    models::DtwRecognizer recognizer;
    models::RtfTimer enroll_timer;

    for (const DigitClip& clip : train) {
        std::vector<float> audio;
        if (!load_and_preprocess_audio(clip.path.string(), audio) || audio.empty()) continue;
        enroll_timer.start();
        models::FeatureMatrix features = extractor.extract(audio);
        if (use_endpoint) features = models::endpoint(features);
        enroll_timer.stop();
        enroll_timer.add_audio(static_cast<double>(audio.size()) / models::kSampleRate);
        if (!features.empty()) recognizer.add_template(clip.label, std::move(features));
    }

    struct Outcome {
        int correct = 0;
        int total = 0;
        models::RtfTimer timer;
    };
    Outcome dtw;
    int64_t scored = 0, pruned = 0, abandoned = 0;
    std::map<std::string, std::map<std::string, int>> confusion;

    for (const DigitClip& clip : test) {
        std::vector<float> audio;
        if (!load_and_preprocess_audio(clip.path.string(), audio) || audio.empty()) continue;
        const double duration = static_cast<double>(audio.size()) / models::kSampleRate;

        dtw.timer.start();
        models::FeatureMatrix features = extractor.extract(audio);
        if (use_endpoint) features = models::endpoint(features);
        const models::DtwRecognizer::Result result =
            recognizer.classify(features, neighbours, radius);
        dtw.timer.stop();
        dtw.timer.add_audio(duration);

        scored += result.templates_scored;
        pruned += result.templates_pruned;
        abandoned += result.templates_abandoned;
        ++dtw.total;
        if (result.label == clip.label) ++dtw.correct;
        confusion[clip.label][result.label]++;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "MFCC + banded DTW (" << neighbours << "-NN, "
              << (use_endpoint ? "endpointed" : "no endpointing") << ", band radius " << radius
              << ")\n";
    std::cout << "  accuracy      : " << 100.0 * dtw.correct / std::max(dtw.total, 1) << " %  ("
              << dtw.correct << "/" << dtw.total << ")\n";
    std::cout << "  speed         : " << dtw.timer.times_real_time() << "x real time  (RTF "
              << dtw.timer.rtf() << ")\n";
    std::cout << "  per clip      : " << 1000.0 * dtw.timer.elapsed() / std::max(dtw.total, 1)
              << " ms against " << recognizer.size() << " templates\n";
    std::cout << "  search savings: lower bound skipped "
              << 100.0 * pruned / std::max<int64_t>(scored + pruned, 1)
              << " % of templates, early abandon dropped "
              << 100.0 * abandoned / std::max<int64_t>(scored + pruned, 1)
              << " % mid-alignment\n";
    std::cout << "  enrolment     : " << enroll_timer.times_real_time() << "x real time\n";

    if (!run_baselines) return;

    // Legacy frame-level pipelines on the same split: their collapsed vowel
    // strings are classified by nearest-neighbour edit distance, which is the
    // most favourable classifier those features admit.
    struct BaselineRun {
        std::string name;
        std::vector<std::vector<std::string>> templates;
        std::vector<std::string> labels;
        int correct = 0;
        int total = 0;
        models::RtfTimer timer;
    };

    models::baseline::LpcFrontend lpc;
    models::baseline::FourierFrontend fourier;
    BaselineRun runs[2] = {{"LPC formants + Bark lookup"}, {"FFT envelope + Bark lookup"}};

    for (const DigitClip& clip : train) {
        std::vector<float> audio;
        if (!load_and_preprocess_audio(clip.path.string(), audio) || audio.empty()) continue;
        runs[0].templates.push_back(models::baseline::collapse(lpc.vowel_string(audio)));
        runs[0].labels.push_back(clip.label);
        runs[1].templates.push_back(models::baseline::collapse(fourier.vowel_string(audio)));
        runs[1].labels.push_back(clip.label);
    }

    for (const DigitClip& clip : test) {
        std::vector<float> audio;
        if (!load_and_preprocess_audio(clip.path.string(), audio) || audio.empty()) continue;
        const double duration = static_cast<double>(audio.size()) / models::kSampleRate;

        for (int which = 0; which < 2; ++which) {
            BaselineRun& run = runs[which];
            run.timer.start();
            const std::vector<std::string> hypothesis =
                models::baseline::collapse(which == 0 ? lpc.vowel_string(audio)
                                                      : fourier.vowel_string(audio));
            int best = 1 << 30;
            std::string best_label;
            for (size_t i = 0; i < run.templates.size(); ++i) {
                const models::EditCounts counts =
                    models::edit_distance(run.templates[i], hypothesis);
                if (static_cast<int>(counts.errors()) < best) {
                    best = static_cast<int>(counts.errors());
                    best_label = run.labels[i];
                }
            }
            run.timer.stop();
            run.timer.add_audio(duration);
            ++run.total;
            if (best_label == clip.label) ++run.correct;
        }
    }

    for (const BaselineRun& run : runs) {
        std::cout << "\n" << run.name << " (legacy frame classifier, 1-NN edit distance)\n";
        std::cout << "  accuracy      : " << 100.0 * run.correct / std::max(run.total, 1) << " %  ("
                  << run.correct << "/" << run.total << ")\n";
        std::cout << "  speed         : " << run.timer.times_real_time() << "x real time  (RTF "
                  << run.timer.rtf() << ")\n";
    }

    std::cout << "\nConfusions (reference -> most frequent hypothesis):\n";
    for (const auto& row : confusion) {
        const auto best = std::max_element(
            row.second.begin(), row.second.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        int total = 0;
        for (const auto& cell : row.second) total += cell.second;
        std::cout << "  " << row.first << " -> " << best->first << "  (" << best->second << "/"
                  << total << ")\n";
    }
}

// ---------------------------------------------------------------- Tier 2

void run_timit(const fs::path& data_root, const fs::path& models_dir, int limit,
               const models::DecoderConfig& base_config) {
    print_header("TIER 2  Phonetic recognition  -  monophone HMM + phone bigram Viterbi");

    models::AcousticModel acoustic;
    if (!acoustic.load((models_dir / "monophone.am").string())) {
        std::cout << "[SKIP] " << (models_dir / "monophone.am") << " not found; run train_models\n";
        return;
    }
    models::NgramLm phone_lm;
    if (!phone_lm.load((models_dir / "phone.lm").string())) {
        std::cout << "[SKIP] " << (models_dir / "phone.lm") << " not found; run train_models\n";
        return;
    }

    // Phone loop: a lexicon whose words are single phones.
    models::Lexicon phone_lexicon;
    for (int p = 0; p < models::num_phones(); ++p) {
        phone_lexicon.add_entry(models::phone_names()[p], {p});
    }

    models::DecoderConfig config = base_config;
    config.allow_silence = false;  // SIL is already a unit in the loop
    models::ViterbiDecoder decoder;
    decoder.build(phone_lexicon, phone_lm, acoustic, config);

    std::unordered_set<std::string> test_speakers;
    {
        std::ifstream file(models_dir / "timit_test_speakers.txt");
        std::string speaker;
        while (file >> speaker) test_speakers.insert(speaker);
    }
    if (test_speakers.empty()) {
        std::cout << "[SKIP] no held-out speaker list; run train_models first\n";
        return;
    }

    const fs::path timit_dir = data_root / "audio" / "timit";
    std::vector<fs::path> wavs;
    for (const auto& entry : fs::directory_iterator(timit_dir)) {
        if (entry.path().extension() == ".wav") wavs.push_back(entry.path());
    }
    std::sort(wavs.begin(), wavs.end());

    models::MfccExtractor extractor;
    models::EditCounts phone_errors;
    models::RtfTimer timer;
    int64_t frames_correct = 0, frames_total = 0;
    int utterances = 0;
    int64_t states_visited = 0;

    for (const fs::path& wav : wavs) {
        const std::string meta = read_file(fs::path(wav).replace_extension(".json"));
        if (!test_speakers.count(json_string_field(meta, "speaker_id"))) continue;
        if (limit > 0 && utterances >= limit) break;

        std::vector<float> audio;
        if (!load_and_preprocess_audio(wav.string(), audio) || audio.empty()) continue;

        // Reference phone sequence, folded to the 39-phone inventory.
        std::vector<std::string> reference;
        std::vector<int> frame_labels;
        {
            std::ifstream alignment(fs::path(wav).replace_extension(".phn"));
            std::string line;
            while (std::getline(alignment, line)) {
                std::istringstream stream(line);
                int start = 0, stop = 0;
                std::string label;
                if (!(stream >> start >> stop >> label)) continue;
                const int phone = models::timit_phone_id(label);
                if (phone < 0) continue;
                const std::string& name = models::phone_names()[phone];
                if (name != "SIL" && (reference.empty() || reference.back() != name)) {
                    reference.push_back(name);
                }
                for (int f = start / models::kHopSize; f < stop / models::kHopSize; ++f) {
                    if (f >= static_cast<int>(frame_labels.size())) frame_labels.resize(f + 1, -1);
                    frame_labels[f] = phone;
                }
            }
        }
        if (reference.empty()) continue;

        timer.start();
        const models::FeatureMatrix features = extractor.extract(audio);
        const models::DecodeResult result = decoder.decode(features);
        timer.stop();
        timer.add_audio(static_cast<double>(audio.size()) / models::kSampleRate);
        states_visited += result.states_visited;

        std::vector<std::string> hypothesis;
        for (const std::string& phone : result.words) {
            if (phone == "SIL" || phone == "<s>" || phone == "</s>" || phone == "<unk>") continue;
            if (hypothesis.empty() || hypothesis.back() != phone) hypothesis.push_back(phone);
        }
        phone_errors += models::edit_distance(reference, hypothesis);

        // Frame-level accuracy of the Gaussians alone (no search), which is the
        // direct analogue of what the legacy vowel classifiers produce.
        std::vector<float> scores;
        for (int t = 0; t < features.num_frames && t < static_cast<int>(frame_labels.size()); ++t) {
            if (frame_labels[t] < 0) continue;
            acoustic.score_frame(features.frame(t), scores);
            int best_state = 0;
            for (int s = 1; s < models::num_states(); ++s) {
                if (scores[s] > scores[best_state]) best_state = s;
            }
            if (best_state / models::kNumStatesPerPhone == frame_labels[t]) ++frames_correct;
            ++frames_total;
        }
        ++utterances;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Held-out speakers: " << test_speakers.size() << "   utterances: " << utterances
              << "\n";
    std::cout << "  phone error rate : " << 100.0 * phone_errors.error_rate() << " %   (S "
              << phone_errors.substitutions << " / D " << phone_errors.deletions << " / I "
              << phone_errors.insertions << " over " << phone_errors.reference_length
              << " phones)\n";
    std::cout << "  frame accuracy   : " << 100.0 * frames_correct / std::max<int64_t>(frames_total, 1)
              << " %   (" << frames_correct << "/" << frames_total << " frames, 40-way)\n";
    std::cout << "  speed            : " << timer.times_real_time() << "x real time  (RTF "
              << timer.rtf() << ")\n";
    std::cout << "  search           : " << states_visited / std::max<int64_t>(utterances, 1)
              << " state visits per utterance\n";
}

// ---------------------------------------------------------------- Tier 3

void run_librispeech(const fs::path& data_root, const fs::path& models_dir, int limit, bool dev_split,
                     const models::DecoderConfig& config, bool restrict_lexicon,
                     bool force_monophone) {
    print_header("TIER 3  Continuous captioning  -  MFCC + monophone HMM + bigram Viterbi");

    models::AcousticModel acoustic;
    if (!acoustic.load((models_dir / "monophone.am").string())) {
        std::cout << "[SKIP] monophone.am not found; run train_models\n";
        return;
    }
    models::NgramLm lm;
    if (!lm.load((models_dir / "bigram.lm").string())) {
        std::cout << "[SKIP] bigram.lm not found; run train_models\n";
        return;
    }

    std::unordered_set<std::string> vocabulary;
    if (restrict_lexicon) {
        for (int w = 0; w < lm.vocabulary_size(); ++w) vocabulary.insert(lm.word(w));
    }

    models::Lexicon lexicon;
    const fs::path dict = models_dir / "cmudict.dict";
    if (!lexicon.load_cmudict(dict.string(), vocabulary)) {
        std::cout << "[SKIP] " << dict << " not found (CMU Pronouncing Dictionary)\n";
        return;
    }

    models::ViterbiDecoder decoder;
    models::RtfTimer build_timer;
    build_timer.start();
    // Use the tied-state triphone tree when one was trained; without it the
    // same code path is a context-independent monophone system.
    models::TriphoneTree tree;
    const bool has_tree = !force_monophone &&
                          tree.load((models_dir / "triphone.tree").string());
    decoder.build(lexicon, lm, acoustic, config, has_tree ? &tree : nullptr);
    const double build_seconds = build_timer.stop();

    std::cout << "Acoustic model: " << acoustic.units() << " units x " << acoustic.mixtures()
              << " mixtures" << (has_tree ? " (tied-state triphones)" : " (monophone)") << "\n";
    std::cout << "Search network: " << decoder.num_states() << " HMM states from "
              << lexicon.pronunciations().size() << " pronunciations / " << lm.vocabulary_size()
              << " LM words (built in " << std::fixed << std::setprecision(2) << build_seconds
              << " s)\n";
    std::cout << "Decoder: acoustic-scale " << config.acoustic_scale << ", beam " << config.beam
              << ", max-active " << config.max_active << ", word-beam " << config.word_beam
              << ", word-penalty " << config.word_insertion_penalty << "\n";

    const fs::path libri_dir = data_root / "audio" / "librispeech";
    std::vector<fs::path> wavs;
    if (fs::exists(libri_dir)) {
        for (const auto& entry : fs::directory_iterator(libri_dir)) {
            if (entry.path().extension() == ".wav") wavs.push_back(entry.path());
        }
    }
    std::sort(wavs.begin(), wavs.end());
    if (wavs.empty()) {
        std::cout << "[SKIP] no LibriSpeech audio in " << libri_dir << "\n";
        return;
    }

    // First 25 % is the development slice used for tuning; the rest is the
    // held-out evaluation set. Never report tuning numbers as test numbers.
    const size_t boundary = wavs.size() / 4;
    std::vector<fs::path> selected;
    if (dev_split) {
        selected.assign(wavs.begin(), wavs.begin() + boundary);
    } else {
        selected.assign(wavs.begin() + boundary, wavs.end());
    }
    if (limit > 0 && static_cast<int>(selected.size()) > limit) selected.resize(limit);

    models::MfccExtractor extractor;
    models::EditCounts word_errors;
    models::RtfTimer timer, feature_timer;
    int64_t oov = 0, reference_words = 0, states_visited = 0, word_entries = 0;
    int utterances = 0;
    std::vector<std::string> sample_reference, sample_hypothesis;

    for (const fs::path& wav : selected) {
        const std::string transcript = read_file(fs::path(wav).replace_extension(".txt"));
        const std::vector<std::string> reference = models::tokenize(transcript);
        if (reference.empty()) continue;

        std::vector<float> audio;
        if (!load_and_preprocess_audio(wav.string(), audio) || audio.empty()) continue;
        const double duration = static_cast<double>(audio.size()) / models::kSampleRate;

        feature_timer.start();
        const models::FeatureMatrix features = extractor.extract(audio);
        feature_timer.stop();
        feature_timer.add_audio(duration);

        timer.start();
        const models::DecodeResult result = decoder.decode(features);
        timer.stop();
        timer.add_audio(duration);

        std::vector<std::string> hypothesis;
        for (const std::string& word : result.words) {
            if (word == "<s>" || word == "</s>" || word == "<unk>") continue;
            hypothesis.push_back(word);
        }

        for (const std::string& word : reference) {
            if (lm.word_id(word) < 0) ++oov;
            ++reference_words;
        }
        word_errors += models::edit_distance(reference, hypothesis);
        states_visited += result.states_visited;
        word_entries += result.word_entries;
        if (utterances == 0) {
            sample_reference = reference;
            sample_hypothesis = hypothesis;
        }
        ++utterances;

        if (utterances % 20 == 0) {
            std::cout << "  " << utterances << "/" << selected.size() << " utterances, WER so far "
                      << 100.0 * word_errors.error_rate() << " %, " << timer.times_real_time()
                      << "x real time\n";
        }
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n" << (dev_split ? "DEV" : "TEST") << " slice: " << utterances << " utterances, "
              << timer.audio() / 60.0 << " min of audio\n";
    std::cout << "  word error rate : " << 100.0 * word_errors.error_rate() << " %   (S "
              << word_errors.substitutions << " / D " << word_errors.deletions << " / I "
              << word_errors.insertions << " over " << word_errors.reference_length
              << " words)\n";
    std::cout << "  OOV rate        : " << 100.0 * oov / std::max<int64_t>(reference_words, 1)
              << " %   (" << oov << " tokens outside the " << lm.vocabulary_size()
              << "-word vocabulary)\n";
    std::cout << "  decode speed    : " << timer.times_real_time() << "x real time  (RTF "
              << timer.rtf() << ")\n";
    std::cout << "  front-end speed : " << feature_timer.times_real_time() << "x real time\n";
    std::cout << "  search          : " << states_visited / std::max<int64_t>(utterances, 1)
              << " state visits and " << word_entries / std::max<int64_t>(utterances, 1)
              << " word entries per utterance\n";

    if (!sample_reference.empty()) {
        std::cout << "\n  sample reference : ";
        for (size_t i = 0; i < sample_reference.size() && i < 14; ++i) {
            std::cout << sample_reference[i] << ' ';
        }
        std::cout << "\n  sample hypothesis: ";
        for (size_t i = 0; i < sample_hypothesis.size() && i < 14; ++i) {
            std::cout << sample_hypothesis[i] << ' ';
        }
        std::cout << "\n";
    }

    std::cout << "\n  Legacy baselines on this tier: the LPC and Fourier programs emit a frame\n"
              << "  level vowel string (IY EH AA ...), not words. With no lexicon or LM they\n"
              << "  cannot produce a transcript at all, so their WER is 100 % by construction.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    const fs::path data_root = argument(argc, argv, "--data-root", "data");
    const fs::path models_dir = argument(argc, argv, "--models", "data/models");
    const int limit = std::stoi(argument(argc, argv, "--limit", "0"));
    const int templates_per_digit = std::stoi(argument(argc, argv, "--templates-per-digit", "20"));
    const int test_per_digit = std::stoi(argument(argc, argv, "--test-per-digit", "50"));
    const bool dev_split = has_flag(argc, argv, "--dev");
    const bool skip_baselines = has_flag(argc, argv, "--no-baselines");
    const int neighbours = std::stoi(argument(argc, argv, "--knn", "1"));
    const int dtw_radius = std::stoi(argument(argc, argv, "--dtw-radius", "12"));

    models::DecoderConfig config;
    config.acoustic_scale = std::stof(argument(argc, argv, "--acoustic-scale", "0.06"));
    config.beam = std::stof(argument(argc, argv, "--beam", "120"));
    config.word_beam = std::stof(argument(argc, argv, "--word-beam", "10"));
    config.max_active = std::stoi(argument(argc, argv, "--max-active", "4000"));
    config.max_word_ends = std::stoi(argument(argc, argv, "--max-word-ends", "24"));
    config.word_insertion_penalty = std::stof(argument(argc, argv, "--word-penalty", "0"));

    const std::vector<std::string> tiers =
        split_csv(argument(argc, argv, "--tiers", "digits,timit,librispeech"));

    std::cout << "======================================================================\n";
    std::cout << "NETWORKLESS CAPTIONING BENCHMARK\n";
    std::cout << "======================================================================\n";
    std::cout << "Data:   " << fs::absolute(data_root) << "\n";
    std::cout << "Models: " << fs::absolute(models_dir) << "\n";

    for (const std::string& tier : tiers) {
        if (tier == "digits") {
            run_digits(data_root, templates_per_digit, test_per_digit, !skip_baselines,
                       neighbours, dtw_radius, !has_flag(argc, argv, "--no-endpoint"));
        } else if (tier == "timit") {
            run_timit(data_root, models_dir, limit, config);
        } else if (tier == "librispeech") {
            run_librispeech(data_root, models_dir, limit, dev_split, config, true,
                            has_flag(argc, argv, "--monophone"));
        } else {
            std::cout << "[WARN] unknown tier '" << tier << "'\n";
        }
    }

    std::cout << "\n";
    return 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\cli\caption.cpp ===
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

#include <toy_pruned_hmm/acoustic_model.hpp>
#include <toy_pruned_hmm/phonemes/lexicon.hpp>
#include <filter/mfcc.hpp>
#include <toy_pruned_hmm/ngram/ngram_lm.hpp>
#include <toy_pruned_hmm/phonemes/phone_set.hpp>
#include <benchmarks/scoring.hpp>
#include <toy_pruned_hmm/triphone/triphone.hpp>
#include <toy_pruned_hmm/viterbi_decoder.hpp>

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

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\test_model_contexts.cpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\train_models.cpp ===
// Trains the networkless captioning stack described in models/README.md.
//
// Two acoustic training routes:
//
//   --train-source timit        bootstrap from TIMIT's hand-marked .phn
//                               boundaries (uniform split inside each phone)
//   --train-source librispeech  matched-domain training on train-clean-100:
//                               no boundaries exist, so a TIMIT-bootstrapped
//                               model forced-aligns the audio against the known
//                               word sequence, the GMMs are re-estimated from
//                               that alignment, and the loop repeats
//
// The second route exists because training on TIMIT and testing on LibriSpeech
// pays a channel-mismatch penalty (1986 close-mic studio vs modern audiobook
// recordings) that no decoder tuning can recover.
//
// Also trains: a word bigram (LibriSpeech train-clean-100 transcripts, disjoint
// from test-clean) and a phone bigram (from the TIMIT alignments).
//
// Usage:
//   train_models [--data-root data] [--out data/models] [--mixtures 8]
//                [--train-source timit|librispeech] [--align-iters 4]
//                [--holdout 0.15] [--vocab 20000]

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "audio_loadnorm.hpp"

#include <toy_pruned_hmm/acoustic_model.hpp>
#include <toy_pruned_hmm/forced_align.hpp>
#include <toy_pruned_hmm/phonemes/lexicon.hpp>
#include <filter/mfcc.hpp>
#include <toy_pruned_hmm/ngram/ngram_lm.hpp>
#include <toy_pruned_hmm/phonemes/phone_set.hpp>
#include <benchmarks/scoring.hpp>
#include <toy_pruned_hmm/triphone/triphone.hpp>

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

std::string read_file(const fs::path& path) {
    std::ifstream file(path);
    if (!file) return {};
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::string json_string_field(const std::string& content, const std::string& key) {
    const size_t at = content.find("\"" + key + "\"");
    if (at == std::string::npos) return {};
    const size_t colon = content.find(':', at);
    if (colon == std::string::npos) return {};
    const size_t first = content.find('"', colon);
    if (first == std::string::npos) return {};
    const size_t last = content.find('"', first + 1);
    if (last == std::string::npos) return {};
    return content.substr(first + 1, last - first - 1);
}

std::string argument(int argc, char** argv, const std::string& flag,
                     const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return fallback;
}

// A corpus held in memory: all frames in one flat array, plus per-utterance
// offsets and the phone sequence each utterance should align to.
struct FeatureStore {
    std::vector<float> frames;          // num_frames x kFeatureDim, row-major
    std::vector<int> offsets;           // utterance i occupies [offsets[i], offsets[i+1])
    std::vector<std::vector<std::pair<int, int>>> contexts;
    std::vector<std::vector<int>> phones;  // expected phone sequence per utterance

    int utterances() const { return static_cast<int>(offsets.size()) - 1; }
    int total_frames() const { return offsets.empty() ? 0 : offsets.back(); }

    models::FeatureMatrix view(int index) const {
        models::FeatureMatrix matrix;
        const int start = offsets[index];
        const int stop = offsets[index + 1];
        matrix.num_frames = stop - start;
        matrix.dim = models::kFeatureDim;
        matrix.data.assign(frames.begin() + static_cast<size_t>(start) * models::kFeatureDim,
                           frames.begin() + static_cast<size_t>(stop) * models::kFeatureDim);
        return matrix;
    }

    void append(const models::FeatureMatrix& features, std::vector<int> phone_sequence,
                std::vector<std::pair<int, int>> phone_contexts) {
        if (offsets.empty()) offsets.push_back(0);
        frames.insert(frames.end(), features.data.begin(), features.data.end());
        offsets.push_back(offsets.back() + features.num_frames);
        phones.push_back(std::move(phone_sequence));
        contexts.push_back(std::move(phone_contexts));
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    const fs::path data_root = argument(argc, argv, "--data-root", "data");
    const fs::path out_dir = argument(argc, argv, "--out", "data/models");
    const double holdout = std::stod(argument(argc, argv, "--holdout", "0.15"));
    const int mixtures = std::stoi(argument(argc, argv, "--mixtures", "8"));
    const int align_iterations = std::stoi(argument(argc, argv, "--align-iters", "4"));
    const std::string train_source = argument(argc, argv, "--train-source", "auto");
    // --lm-only retrains just the language model (e.g. at a different
    // vocabulary size) without touching the acoustic model.
    const bool lm_only = !!std::count(argv, argv + argc, std::string("--lm-only"));
    // Triphone state tying (Path 3): off with --monophone, so the monophone
    // baseline stays reproducible for comparison.
    const bool triphones = !std::count(argv, argv + argc, std::string("--monophone"));
    const int max_senones = std::stoi(argument(argc, argv, "--senones", "2200"));
    const double min_gain = std::stod(argument(argc, argv, "--split-gain", "1200"));
    const double min_occupancy = std::stod(argument(argc, argv, "--min-occupancy", "600"));
    const int triphone_align_iters = std::stoi(argument(argc, argv, "--triphone-iters", "2"));

    fs::create_directories(out_dir);

    std::cout << "======================================================================\n";
    std::cout << "TRAINING NETWORKLESS CAPTIONING MODELS\n";
    std::cout << "======================================================================\n";

    if (lm_only) {
        models::NgramLm lm;
        const int limit = std::stoi(argument(argc, argv, "--vocab", "20000"));
        if (!lm.train_from_text((out_dir / "lm_corpus.txt").string(), limit)) {
            std::cerr << "[ERROR] LM training failed\n";
            return 1;
        }
        lm.save((out_dir / "bigram.lm").string());
        std::cout << "\nWord bigram LM only: " << lm.vocabulary_size() << " words -> "
                  << (out_dir / "bigram.lm") << "\n";
        return 0;
    }

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
        speakers.push_back(json_string_field(read_file(fs::path(wav).replace_extension(".json")),
                                             "speaker_id"));
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

    std::cout << "\nTIMIT bootstrap: " << timit_wavs.size() << " utterances, " << unique.size()
              << " speakers (" << test_speakers.size() << " held out)\n";

    models::MfccExtractor extractor;
    models::AcousticModel acoustic;
    acoustic.begin_training();

    std::vector<std::vector<std::string>> phone_sentences;
    std::vector<float> timit_frames;
    std::vector<int> timit_states;
    models::RtfTimer front_end_timer;
    int trained = 0;
    int64_t frames = 0;

    for (size_t i = 0; i < timit_wavs.size(); ++i) {
        if (test_speakers.count(speakers[i])) continue;

        std::vector<float> audio;
        if (!load_and_preprocess_audio(timit_wavs[i].string(), audio) || audio.empty()) continue;

        const std::vector<Segment> segments =
            read_alignment(fs::path(timit_wavs[i]).replace_extension(".phn"));
        if (segments.empty()) continue;

        front_end_timer.start();
        const models::FeatureMatrix features = extractor.extract(audio);
        front_end_timer.stop();
        front_end_timer.add_audio(static_cast<double>(audio.size()) / models::kSampleRate);
        if (features.empty()) continue;

        std::vector<std::string> phone_sequence;
        for (const Segment& segment : segments) {
            const int start = segment.start_sample / models::kHopSize;
            const int stop = std::min(segment.stop_sample / models::kHopSize, features.num_frames);
            if (stop <= start) continue;
            acoustic.accumulate_segment(features, start, stop, segment.phone);
            const int length = stop - start;
            for (int t = start; t < stop; ++t) {
                int sub = ((t - start) * models::kNumStatesPerPhone) / length;
                if (sub >= models::kNumStatesPerPhone) sub = models::kNumStatesPerPhone - 1;
                timit_states.push_back(models::state_index(segment.phone, sub));
                timit_frames.insert(timit_frames.end(), features.frame(t),
                                    features.frame(t) + models::kFeatureDim);
            }
            frames += length;
            phone_sequence.push_back(models::phone_names()[segment.phone]);
        }
        if (!phone_sequence.empty()) phone_sentences.push_back(std::move(phone_sequence));
        ++trained;
    }

    acoustic.finish_training();
    std::cout << "  bootstrap model: " << trained << " utterances, " << frames << " frames ("
              << frames / 6000.0 << " min)\n";

    // Phone bigram over the TIMIT alignments, used by the Tier 2 phone loop.
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
            std::cout << "  wrote " << (out_dir / "phone.lm") << " (" << phone_lm.vocabulary_size()
                      << " units)\n";
        }
    }

    // ------------------------------------------- matched-domain re-training
    const fs::path libri_dir = data_root / "audio" / "librispeech_train";
    const bool want_librispeech =
        train_source == "librispeech" || (train_source == "auto" && fs::exists(libri_dir));

    std::vector<float>* mixture_frames = &timit_frames;
    std::vector<int>* mixture_states = &timit_states;
    std::vector<int> aligned_states;
    FeatureStore store;
    models::TriphoneTree tree;

    if (want_librispeech && fs::exists(libri_dir)) {
        // Lexicon for alignment: every word in the training transcripts.
        models::Lexicon lexicon;
        if (!lexicon.load_cmudict((out_dir / "cmudict.dict").string())) {
            std::cerr << "[ERROR] need " << (out_dir / "cmudict.dict") << " to align\n";
            return 1;
        }
        models::PronunciationTable pronunciations;
        pronunciations.build(lexicon);

        std::vector<fs::path> wavs;
        for (const auto& entry : fs::directory_iterator(libri_dir)) {
            if (entry.path().extension() == ".wav") wavs.push_back(entry.path());
        }
        std::sort(wavs.begin(), wavs.end());

        std::cout << "\nMatched-domain training: " << wavs.size() << " LibriSpeech utterances\n";
        std::cout << "  lexicon: " << pronunciations.size() << " words\n";

        int skipped_oov = 0;
        int guessed_words = 0;
        for (const fs::path& wav : wavs) {
            const std::vector<std::string> words =
                models::tokenize(read_file(fs::path(wav).replace_extension(".txt")));
            if (words.empty()) continue;

            std::vector<int> phone_sequence;
            std::vector<std::pair<int, int>> phone_contexts;
            if (!pronunciations.phones_for(words, phone_sequence, &guessed_words, false, &phone_contexts)) {
                ++skipped_oov;  // no pronunciation at all, even by rule
                continue;
            }

            std::vector<float> audio;
            if (!load_and_preprocess_audio(wav.string(), audio) || audio.empty()) continue;

            front_end_timer.start();
            const models::FeatureMatrix features = extractor.extract(audio);
            front_end_timer.stop();
            front_end_timer.add_audio(static_cast<double>(audio.size()) / models::kSampleRate);
            if (features.empty()) continue;

            store.append(features, std::move(phone_sequence), std::move(phone_contexts));
        }

        std::cout << "  loaded " << store.utterances() << " utterances, " << store.total_frames()
                  << " frames (" << store.total_frames() / 6000.0 << " min), " << skipped_oov
                  << " skipped, " << guessed_words
                  << " words pronounced by letter-to-sound fallback\n";
        std::cout << "  feature cache: " << store.frames.size() * sizeof(float) / (1024 * 1024)
                  << " MB\n";

        // Forced-alignment loop: align with the current model, re-estimate the
        // model from that alignment, repeat. The TIMIT model is only a
        // bootstrap - after the first iteration every parameter has been
        // re-estimated on LibriSpeech audio.
        models::ForcedAligner aligner;
        for (int iteration = 1; iteration <= align_iterations; ++iteration) {
            models::RtfTimer timer;
            timer.start();

            models::AcousticModel next;
            next.begin_training();
            aligned_states.clear();
            aligned_states.reserve(store.total_frames());

            int aligned = 0;
            int64_t aligned_frames = 0;
            for (int u = 0; u < store.utterances(); ++u) {
                const models::FeatureMatrix features = store.view(u);
                const std::vector<models::AlignedSegment> segments =
                    aligner.align(features, store.phones[u], acoustic);
                if (segments.empty()) {
                    // No complete path: keep the frames out of this iteration.
                    aligned_states.resize(aligned_states.size() + features.num_frames, -1);
                    continue;
                }
                const std::vector<int>& path = aligner.state_path();
                for (int t = 0; t < features.num_frames; ++t) {
                    next.accumulate_frame(features.frame(t), path[t]);
                    aligned_states.push_back(path[t]);
                }
                for (const models::AlignedSegment& segment : segments) {
                    next.accumulate_duration(segment.phone,
                                             segment.stop_frame - segment.start_frame);
                }
                ++aligned;
                aligned_frames += features.num_frames;
            }

            next.finish_training();
            acoustic = next;
            const double seconds = timer.stop();
            std::cout << "  align pass " << iteration << "/" << align_iterations << ": " << aligned
                      << "/" << store.utterances() << " utterances, " << aligned_frames
                      << " frames in " << seconds << " s\n";
        }

        mixture_frames = &store.frames;
        mixture_states = &aligned_states;

        // ------------------------------------------ tied-state triphones
        if (triphones) {
            models::RtfTimer timer;
            timer.start();

            // Collect per-context statistics from the final monophone
            // alignment: one entry per (centre, state) x (left, right).
            std::vector<std::vector<models::ContextStats>> contexts(
                models::TriphoneTree::num_tree_keys());
            std::vector<std::unordered_map<int, int>> lookup(contexts.size());
            const int silence = models::phone_id("SIL");

            std::vector<int> senone_of_frame(store.total_frames(), -1);
            std::vector<std::array<int, 4>> frame_context(store.total_frames(),
                                                          {{-1, -1, -1, -1}});
            // (phone, frames) for every aligned segment, replayed into the
            // triphone model so transitions do not have to be re-derived.
            std::vector<std::pair<int, int>> durations;

            for (int u = 0; u < store.utterances(); ++u) {
                const models::FeatureMatrix features = store.view(u);
                const std::vector<models::AlignedSegment> segments =
                    aligner.align(features, store.phones[u], acoustic);
                if (segments.empty()) continue;
                const std::vector<int>& path = aligner.state_path();
                const int base = store.offsets[u];

                for (size_t s = 0; s < segments.size(); ++s) {
                    const int centre = segments[s].phone;
                    const int left = store.contexts[u][s].first;
                    const int right = store.contexts[u][s].second;

                    for (int t = segments[s].start_frame; t < segments[s].stop_frame; ++t) {
                        const int sub = path[t] % models::kNumStatesPerPhone;
                        const int key = models::TriphoneTree::tree_key(centre, sub);
                        const int context_id = left * 64 + right;

                        auto it = lookup[key].find(context_id);
                        if (it == lookup[key].end()) {
                            it = lookup[key].emplace(context_id,
                                                     static_cast<int>(contexts[key].size())).first;
                            models::ContextStats stats;
                            stats.left = left;
                            stats.right = right;
                            contexts[key].push_back(std::move(stats));
                        }
                        contexts[key][it->second].add(features.frame(t));
                        frame_context[base + t] = {{left, centre, sub, right}};
                    }
                    durations.emplace_back(centre,
                                           segments[s].stop_frame - segments[s].start_frame);
                }
            }

            int distinct = 0;
            for (const auto& list : contexts) distinct += static_cast<int>(list.size());

            tree.build(contexts, min_gain, min_occupancy, max_senones);
            std::cout << "\n  triphone tying: " << distinct
                      << " observed contexts -> " << tree.num_senones() << " tied states in "
                      << timer.stop() << " s\n";

            // Re-label every frame with its senone and re-estimate from scratch.
            for (int i = 0; i < store.total_frames(); ++i) {
                const std::array<int, 4>& context = frame_context[i];
                if (context[1] < 0) continue;
                senone_of_frame[i] = tree.senone(context[0], context[1], context[2], context[3]);
            }

            models::AcousticModel triphone_model;
            triphone_model.begin_training(tree.num_senones());
            for (int i = 0; i < store.total_frames(); ++i) {
                if (senone_of_frame[i] < 0) continue;
                triphone_model.accumulate_frame(
                    store.frames.data() + static_cast<size_t>(i) * models::kFeatureDim,
                    senone_of_frame[i]);
            }
            for (const std::pair<int, int>& duration : durations) {
                triphone_model.accumulate_duration(duration.first, duration.second);
            }
            triphone_model.finish_training();
            acoustic = triphone_model;

            // One more alignment pass, now scored with the triphone model.
            for (int iteration = 1; iteration <= triphone_align_iters; ++iteration) {
                models::RtfTimer pass;
                pass.start();
                models::AcousticModel next;
                next.begin_training(tree.num_senones());
                std::fill(senone_of_frame.begin(), senone_of_frame.end(), -1);
                int aligned = 0;

                for (int u = 0; u < store.utterances(); ++u) {
                    const models::FeatureMatrix features = store.view(u);
                    const std::vector<models::AlignedSegment> segments =
                        aligner.align(features, store.phones[u], acoustic, &tree, &store.contexts[u]);
                    if (segments.empty()) continue;
                    const int base = store.offsets[u];
                    const std::vector<int>& path = aligner.state_path();

                    for (size_t s = 0; s < segments.size(); ++s) {
                        const int centre = segments[s].phone;
                        const int left = store.contexts[u][s].first;
                        const int right = store.contexts[u][s].second;
                        for (int t = segments[s].start_frame; t < segments[s].stop_frame; ++t) {
                            const int sub = path[t] % models::kNumStatesPerPhone;
                            const int senone = tree.senone(left, centre, sub, right);
                            senone_of_frame[base + t] = senone;
                            next.accumulate_frame(features.frame(t), senone);
                        }
                        next.accumulate_duration(centre,
                                                 segments[s].stop_frame - segments[s].start_frame);
                    }
                    ++aligned;
                }
                next.finish_training();
                acoustic = next;
                std::cout << "  triphone align pass " << iteration << "/" << triphone_align_iters
                          << ": " << aligned << " utterances in " << pass.stop() << " s\n";
            }

            aligned_states = senone_of_frame;
            tree.save((out_dir / "triphone.tree").string());
            std::cout << "  wrote " << (out_dir / "triphone.tree") << "\n";
        }
    }

    // ------------------------------------------------------------ mixtures
    if (mixtures > 1 && !mixture_states->empty()) {
        models::RtfTimer em_timer;
        em_timer.start();
        acoustic.train_mixtures(*mixture_frames, *mixture_states, mixtures);
        const double seconds = em_timer.stop();
        std::cout << "\n  mixture EM: " << mixtures << " components/state over "
                  << mixture_states->size() << " frames in " << seconds << " s\n";
    }

    const fs::path am_path = out_dir / "monophone.am";
    if (!acoustic.save(am_path.string())) {
        std::cerr << "[ERROR] Could not write " << am_path << "\n";
        return 1;
    }
    std::cout << "  wrote " << am_path << " (" << acoustic.units() << " units x "
              << acoustic.mixtures() << " mixtures x " << models::kFeatureDim << " dims)\n";
    std::cout << "  front-end speed: " << front_end_timer.times_real_time() << "x real time\n";

    // ------------------------------------------------------------ word LM
    const fs::path corpus = out_dir / "lm_corpus.txt";
    if (!fs::exists(corpus)) {
        std::cerr << "\n[WARN] " << corpus << " not found; skipping word LM.\n";
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

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\int8_zip\batch_asr.cpp ===
#include <captions/batch_asr.hpp>
#include "ort_graph.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace captions {
namespace {
using Clock = std::chrono::steady_clock;
using detail::Graph;
constexpr int kMel = 80;
constexpr size_t kFrame = 160;  // 10 ms at 16 kHz
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
} // namespace

std::vector<Span> segment_audio(std::span<const float> audio, const SegmenterConfig& c) {
    if (!(c.max_seconds > 0) || c.min_seconds < 0 || c.min_seconds >= c.max_seconds ||
        c.hangover_ms < 0 || c.pad_ms < 0 || !std::isfinite(c.gate_rms) || c.gate_rms < 0)
        throw std::invalid_argument("Invalid segmenter configuration");
    const size_t frames = (audio.size() + kFrame - 1) / kFrame;
    if (frames == 0) return {};
    std::vector<float> energy(frames);
    for (size_t f = 0; f < frames; ++f) {
        const size_t b = f * kFrame, e = std::min(audio.size(), b + kFrame);
        double power = 0;
        for (size_t i = b; i < e; ++i) power += static_cast<double>(audio[i]) * audio[i];
        energy[f] = static_cast<float>(power / static_cast<double>(e - b));
    }
    const float threshold = c.gate_rms * c.gate_rms;
    const size_t hang = static_cast<size_t>(c.hangover_ms) / 10, pad = static_cast<size_t>(c.pad_ms) / 10;

    // Voiced regions, bridging pauses shorter than the hangover.
    std::vector<std::pair<size_t, size_t>> regions;  // frame [begin, end)
    for (size_t f = 0; f < frames; ++f) {
        if (energy[f] <= threshold) continue;
        if (!regions.empty() && f - regions.back().second <= hang) regions.back().second = f + 1;
        else regions.push_back({f, f + 1});
    }
    std::vector<std::pair<size_t, size_t>> padded;
    for (auto [b, e] : regions) {
        b = b > pad ? b - pad : 0;
        e = std::min(frames, e + pad);
        if (!padded.empty() && b <= padded.back().second) padded.back().second = e;
        else padded.push_back({b, e});
    }

    // Cut long regions at the quietest point (200 ms moving average) in the
    // allowed window, so a cut falls in a pause rather than a stop closure.
    std::vector<double> prefix(frames + 1, 0.0);
    for (size_t f = 0; f < frames; ++f) prefix[f + 1] = prefix[f] + energy[f];
    auto smoothed = [&](size_t f) {
        const size_t b = f > 10 ? f - 10 : 0, e = std::min(frames, f + 10);
        return (prefix[e] - prefix[b]) / static_cast<double>(e - b);
    };
    const size_t max_frames = static_cast<size_t>(c.max_seconds * 100);
    const size_t min_frames = static_cast<size_t>(c.min_seconds * 100);
    std::vector<Span> spans;
    for (auto [b, e] : padded) {
        while (e - b > max_frames) {
            size_t cut = b + max_frames;
            double best = smoothed(cut);
            for (size_t f = b + std::max<size_t>(min_frames, 1); f < b + max_frames; ++f) {
                const double v = smoothed(f);
                if (v < best) { best = v; cut = f; }
            }
            spans.push_back({b * kFrame, cut * kFrame});
            b = cut;
        }
        spans.push_back({b * kFrame, std::min(audio.size(), e * kFrame)});
    }
    return spans;
}

namespace {
struct Job {
    std::span<const float> samples;
    double offset = 0;  // seconds of samples[0] in the source
    SegmentResult* out = nullptr;
};

// One worker's private sessions. ONNX Runtime sessions are thread-safe, but
// sharing one would also share its intra-op pool; private sessions keep every
// worker's matrix kernels on their own threads.
struct Worker {
    Graph encoder, decoder, joiner;
    Worker(Ort::Env& env, const BatchAsrConfig& c)
        : encoder(env, c.encoder, c.threads_per_worker, c.spin), decoder(env, c.decoder, 1),
          joiner(env, c.joiner, 1) {}
};
} // namespace

struct BatchOnnxAsr::Impl {
    BatchAsrConfig config;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "batch-asr"};
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<std::string> tokens;
    std::string type;
    int window = 0, shift = 0, context = 0, vocabulary = 0, unknown = -1, joiner_dim = 0;
    std::vector<int> batch_axis;  // per encoder cache input
    BatchAsrStats stats;
    std::mutex stats_mutex;

    explicit Impl(BatchAsrConfig c) : config(std::move(c)) {
        if (config.threads_per_worker < 1 || config.threads_per_worker > 16 ||
            config.batch < 1 || config.batch > 256 || config.beam < 1 || config.beam > 64 || config.workers < 0 || config.workers > 64)
            throw std::invalid_argument("Invalid batch ASR configuration");
        if (config.workers == 0) {
            const int hw = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
            config.workers = std::max(1, hw / config.threads_per_worker);
        }
        workers.resize(config.workers);
        std::vector<std::thread> loaders;
        std::exception_ptr failure;
        std::mutex failure_mutex;
        for (int i = 0; i < config.workers; ++i)
            loaders.emplace_back([&, i] {
                try { workers[i] = std::make_unique<Worker>(env, config); }
                catch (...) { std::lock_guard lock(failure_mutex); failure = std::current_exception(); }
            });
        for (auto& t : loaders) t.join();
        if (failure) std::rethrow_exception(failure);

        auto& w = *workers.front();
        type = w.encoder.metadata("model_type");
        if (type != "zipformer" && type != "zipformer2")
            throw std::runtime_error("Expected streaming Zipformer transducer graphs");
        window = std::stoi(w.encoder.metadata("T"));
        shift = std::stoi(w.encoder.metadata("decode_chunk_len"));
        context = std::stoi(w.decoder.metadata("context_size"));
        vocabulary = std::stoi(w.decoder.metadata("vocab_size"));
        if (window < 1 || window > 1000 || shift < 1 || shift > window || context < 1 || context > 32 ||
            vocabulary < 3 || w.encoder.inputs.size() != w.encoder.outputs.size() ||
            w.decoder.inputs.size() != 1 || w.joiner.inputs.size() != 2)
            throw std::runtime_error("Unsupported streaming graph contract");
        joiner_dim = static_cast<int>(w.joiner.session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().back());
        tokens = detail::read_tokens(config.tokens, vocabulary, unknown);
        // Every cache input has exactly one dynamic dimension: the batch axis.
        for (size_t i = 1; i < w.encoder.inputs.size(); ++i) {
            const auto shape = w.encoder.session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            int axis = -1;
            for (size_t d = 0; d < shape.size(); ++d) {
                if (shape[d] == -1) {
                    if (axis != -1) throw std::runtime_error("Cache tensor with two dynamic axes");
                    axis = static_cast<int>(d);
                } else if (shape[d] <= 0) throw std::runtime_error("Unsupported cache shape");
            }
            if (axis == -1) throw std::runtime_error("Cache tensor without a batch axis");
            batch_axis.push_back(axis);
        }
    }

    // Features for one segment, identical to the streaming engine's single
    // gated segment: samples, then right-context zeros, then InputFinished.
    std::vector<float> features(std::span<const float> samples, int& frames) const {
        knf::OnlineFbank fbank(detail::fbank_options());
        if (!samples.empty()) fbank.AcceptWaveform(16000, samples.data(), static_cast<int>(samples.size()));
        std::vector<float> tail(static_cast<size_t>(window) * 160 + 4800, 0.F);
        fbank.AcceptWaveform(16000, tail.data(), static_cast<int>(tail.size()));
        fbank.InputFinished();
        frames = fbank.NumFramesReady();
        std::vector<float> out(static_cast<size_t>(frames) * kMel);
        for (int t = 0; t < frames; ++t) std::copy_n(fbank.GetFrame(t), kMel, out.data() + static_cast<size_t>(t) * kMel);
        return out;
    }

    std::vector<Ort::Value> zero_states(Worker& w, std::int64_t batch) const {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<Ort::Value> states;
        states.reserve(batch_axis.size());
        for (size_t i = 0; i < batch_axis.size(); ++i) {
            auto info = w.encoder.session.GetInputTypeInfo(i + 1).GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            shape[batch_axis[i]] = batch;
            if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                auto v = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
                std::fill_n(v.GetTensorMutableData<float>(), v.GetTensorTypeAndShapeInfo().GetElementCount(), 0.F);
                states.push_back(std::move(v));
            } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                auto v = Ort::Value::CreateTensor<std::int64_t>(allocator, shape.data(), shape.size());
                std::fill_n(v.GetTensorMutableData<std::int64_t>(), v.GetTensorTypeAndShapeInfo().GetElementCount(), 0);
                states.push_back(std::move(v));
            } else throw std::runtime_error("Unsupported cache element type");
        }
        return states;
    }

    // Lockstep decode of one batch. Rows whose segment has ended keep running
    // on zero features (their outputs are discarded); jobs are length-sorted,
    // so that padding is small.
    struct Timing { double fbank = 0, encoder = 0, search = 0; };
    std::uint64_t decode_batch(Worker& w, std::span<Job> jobs, Timing& timing) const {
        auto t0 = Clock::now();
        const auto B = static_cast<std::int64_t>(jobs.size());
        const size_t D = static_cast<size_t>(joiner_dim);
        std::vector<std::vector<float>> feats(jobs.size());
        std::vector<int> chunks(jobs.size());
        int max_chunks = 0;
        for (size_t b = 0; b < jobs.size(); ++b) {
            int frames = 0;
            feats[b] = features(jobs[b].samples, frames);
            chunks[b] = frames >= window ? (frames - window) / shift + 1 : 0;
            max_chunks = std::max(max_chunks, chunks[b]);
        }
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        timing.fbank += seconds(t0);
        auto states = zero_states(w, B);

        // Decoder context per row; the original Zipformer export embeds y with
        // Gather, so it pads with blank (0); Zipformer2 masks -1 padding.
        std::vector<std::vector<std::int64_t>> history(jobs.size());
        std::vector<std::vector<std::pair<int, int>>> emitted(jobs.size());  // (token, output frame)
        for (auto& h : history) { h.assign(context, type == "zipformer" ? 0 : -1); h.back() = 0; }
        std::vector<float> dec_out(static_cast<size_t>(B) * D);
        std::vector<std::int64_t> dec_in;
        std::vector<size_t> rows;
        auto run_decoder = [&](const std::vector<size_t>& which) {
            dec_in.clear();
            for (size_t b : which) dec_in.insert(dec_in.end(), history[b].end() - context, history[b].end());
            const std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(which.size()), context};
            auto in = Ort::Value::CreateTensor<std::int64_t>(memory, dec_in.data(), dec_in.size(), shape.data(), shape.size());
            auto out = w.decoder.run(std::span<const Ort::Value>(&in, 1));
            const float* p = out[0].GetTensorData<float>();
            for (size_t i = 0; i < which.size(); ++i) std::copy_n(p + i * D, D, dec_out.data() + which[i] * D);
        };
        rows.resize(jobs.size());
        std::iota(rows.begin(), rows.end(), size_t{0});
        run_decoder(rows);

        // Beam state (config.beam > 1). Each hypothesis keeps its own decoder
        // output; hypotheses of every active row share one joiner call.
        struct Hyp {
            std::vector<std::int64_t> ys;
            std::vector<std::pair<int, int>> timeline;
            double score = 0;
            std::shared_ptr<const std::vector<float>> dec;
        };
        std::vector<std::vector<Hyp>> beams;
        if (config.beam > 1)
            for (size_t b = 0; b < jobs.size(); ++b)
                beams.push_back({Hyp{history[b], {}, 0.0, std::make_shared<const std::vector<float>>(
                    dec_out.begin() + b * D, dec_out.begin() + (b + 1) * D)}});

        std::vector<float> x(static_cast<size_t>(B) * window * kMel);
        std::vector<float> enc_rows, dec_rows;
        std::vector<size_t> active, fired;
        int out_frames = 0;

        // Modified beam search (icefall): at most one symbol per frame; the
        // top `beam` (hypothesis, token) pairs survive, equal token sequences
        // merge by log-add.
        auto beam_step = [&](const float* encoded, int U, int t, int k) {
            const size_t K = static_cast<size_t>(config.beam);
            size_t R = 0;
            for (size_t b : active) R += beams[b].size();
            enc_rows.resize(R * D);
            dec_rows.resize(R * D);
            size_t r = 0;
            for (size_t b : active)
                for (const auto& h : beams[b]) {
                    std::copy_n(encoded + (b * U + t) * D, D, enc_rows.data() + r * D);
                    std::copy_n(h.dec->data(), D, dec_rows.data() + r * D);
                    ++r;
                }
            const std::array<std::int64_t, 2> join_shape{static_cast<std::int64_t>(R), joiner_dim};
            std::array<Ort::Value, 2> join_in{
                Ort::Value::CreateTensor<float>(memory, enc_rows.data(), enc_rows.size(), join_shape.data(), 2),
                Ort::Value::CreateTensor<float>(memory, dec_rows.data(), dec_rows.size(), join_shape.data(), 2)};
            auto logits = w.joiner.run(join_in);
            if (logits[0].GetTensorTypeAndShapeInfo().GetElementCount() != R * vocabulary)
                throw std::runtime_error("Joiner/token vocabulary mismatch");
            float* values = logits[0].GetTensorMutableData<float>();

            struct Candidate { double score; size_t hyp; int token; };
            std::vector<Candidate> candidates;
            std::vector<int> order(static_cast<size_t>(vocabulary));
            std::vector<std::pair<size_t, size_t>> pending;  // (row, hyp) needing a decoder run
            r = 0;
            for (size_t b : active) {
                auto& hyps = beams[b];
                candidates.clear();
                for (size_t i = 0; i < hyps.size(); ++i, ++r) {
                    float* row = values + r * vocabulary;
                    if (unknown >= 0) row[unknown] = -std::numeric_limits<float>::infinity();
                    const float m = *std::max_element(row, row + vocabulary);
                    double sum = 0;
                    for (int v = 0; v < vocabulary; ++v) sum += std::exp(static_cast<double>(row[v] - m));
                    const double norm = m + std::log(sum);
                    std::iota(order.begin(), order.end(), 0);
                    const size_t take = std::min(K, order.size());
                    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(take), order.end(),
                                      [&](int a, int c) { return row[a] > row[c]; });
                    for (size_t j = 0; j < take; ++j)
                        candidates.push_back({hyps[i].score + row[order[j]] - norm, i, order[j]});
                }
                const size_t keep = std::min(K, candidates.size());
                std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep),
                                  candidates.end(), [](const auto& a, const auto& c) { return a.score > c.score; });
                std::vector<Hyp> next;
                for (size_t j = 0; j < keep; ++j) {
                    const auto& c = candidates[j];
                    Hyp h = hyps[c.hyp];
                    h.score = c.score;
                    if (c.token != 0) {
                        h.ys.push_back(c.token);
                        h.timeline.push_back({c.token, k * U + t});
                        h.dec.reset();
                    }
                    auto same = std::find_if(next.begin(), next.end(), [&](const Hyp& o) { return o.ys == h.ys; });
                    if (same == next.end()) { next.push_back(std::move(h)); continue; }
                    const double hi = std::max(same->score, h.score), lo = std::min(same->score, h.score);
                    same->score = hi + std::log1p(std::exp(lo - hi));  // candidates arrive best-first
                }
                hyps = std::move(next);
                for (size_t i = 0; i < hyps.size(); ++i) if (!hyps[i].dec) pending.push_back({b, i});
            }
            if (pending.empty()) return;
            dec_in.clear();
            for (auto [b, i] : pending) {
                const auto& ys = beams[b][i].ys;
                dec_in.insert(dec_in.end(), ys.end() - context, ys.end());
            }
            const std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(pending.size()), context};
            auto in = Ort::Value::CreateTensor<std::int64_t>(memory, dec_in.data(), dec_in.size(), shape.data(), shape.size());
            auto out = w.decoder.run(std::span<const Ort::Value>(&in, 1));
            const float* p = out[0].GetTensorData<float>();
            for (size_t j = 0; j < pending.size(); ++j)
                beams[pending[j].first][pending[j].second].dec =
                    std::make_shared<const std::vector<float>>(p + j * D, p + (j + 1) * D);
        };
        for (int k = 0; k < max_chunks; ++k) {
            std::fill(x.begin(), x.end(), 0.F);
            active.clear();
            for (size_t b = 0; b < jobs.size(); ++b) {
                if (k >= chunks[b]) continue;
                active.push_back(b);
                std::copy_n(feats[b].data() + static_cast<size_t>(k) * shift * kMel,
                            static_cast<size_t>(window) * kMel, x.data() + b * window * kMel);
            }
            const std::array<std::int64_t, 3> shape{B, window, kMel};
            std::vector<Ort::Value> in;
            in.reserve(1 + states.size());
            in.push_back(Ort::Value::CreateTensor<float>(memory, x.data(), x.size(), shape.data(), shape.size()));
            for (auto& s : states) in.push_back(std::move(s));
            t0 = Clock::now();
            auto out = w.encoder.run(in);
            timing.encoder += seconds(t0);
            t0 = Clock::now();
            for (size_t i = 0; i < states.size(); ++i) states[i] = std::move(out[i + 1]);
            const auto dims = out[0].GetTensorTypeAndShapeInfo().GetShape();
            if (dims.size() != 3 || dims[0] != B || dims[2] != joiner_dim)
                throw std::runtime_error("Unexpected encoder output dimensions");
            const auto U = static_cast<int>(dims[1]);
            out_frames = U;
            const float* encoded = out[0].GetTensorData<float>();

            if (config.beam > 1) {
                for (int t = 0; t < U; ++t) beam_step(encoded, U, t, k);
                timing.search += seconds(t0);
                continue;
            }
            for (int t = 0; t < U; ++t) {
                const auto A = static_cast<std::int64_t>(active.size());
                enc_rows.resize(active.size() * D);
                dec_rows.resize(active.size() * D);
                for (size_t i = 0; i < active.size(); ++i) {
                    std::copy_n(encoded + (active[i] * U + t) * D, D, enc_rows.data() + i * D);
                    std::copy_n(dec_out.data() + active[i] * D, D, dec_rows.data() + i * D);
                }
                const std::array<std::int64_t, 2> join_shape{A, joiner_dim};
                std::array<Ort::Value, 2> join_in{
                    Ort::Value::CreateTensor<float>(memory, enc_rows.data(), enc_rows.size(), join_shape.data(), 2),
                    Ort::Value::CreateTensor<float>(memory, dec_rows.data(), dec_rows.size(), join_shape.data(), 2)};
                auto logits = w.joiner.run(join_in);
                if (logits[0].GetTensorTypeAndShapeInfo().GetElementCount() != active.size() * vocabulary)
                    throw std::runtime_error("Joiner/token vocabulary mismatch");
                const float* values = logits[0].GetTensorData<float>();
                fired.clear();
                for (size_t i = 0; i < active.size(); ++i) {
                    const float* row = values + i * vocabulary;
                    const auto id = static_cast<int>(std::max_element(row, row + vocabulary) - row);
                    // Icefall streaming transducers use at most one symbol/frame.
                    if (id == 0 || id == unknown) continue;
                    history[active[i]].push_back(id);
                    emitted[active[i]].push_back({id, k * U + t});
                    fired.push_back(active[i]);
                }
                if (!fired.empty()) run_decoder(fired);
            }
            timing.search += seconds(t0);
        }

        // Output frames tile each chunk's shift evenly (40 ms for these exports).
        const double frame_seconds = out_frames > 0 ? shift * 0.01 / out_frames : 0.04;
        const std::string marker = "\xE2\x96\x81";
        for (size_t b = 0; b < beams.size(); ++b)
            emitted[b] = std::max_element(beams[b].begin(), beams[b].end(), [](const Hyp& a, const Hyp& c) {
                return a.score < c.score;
            })->timeline;
        for (size_t b = 0; b < jobs.size(); ++b) {
            auto& r = *jobs[b].out;
            r.start = jobs[b].offset;
            r.end = jobs[b].offset + jobs[b].samples.size() / 16000.0;
            r.words.clear();
            for (auto [id, frame] : emitted[b]) {
                std::string piece = tokens[static_cast<size_t>(id)];
                const double at = r.start + frame * frame_seconds;
                const bool starts_word = piece.rfind(marker, 0) == 0;
                if (starts_word) piece.erase(0, marker.size());
                if (starts_word || r.words.empty()) r.words.push_back({piece, at, at + frame_seconds});
                else { r.words.back().text += piece; r.words.back().end = at + frame_seconds; }
            }
            std::erase_if(r.words, [](const Word& wd) { return wd.text.empty(); });
            r.text.clear();
            for (const auto& wd : r.words) { if (!r.text.empty()) r.text += ' '; r.text += wd.text; }
        }
        return static_cast<std::uint64_t>(max_chunks);
    }

    void run(std::vector<Job>& jobs) {
        const auto started = Clock::now();
        // Longest first: similar lengths share a batch, and the longest
        // batches start early so no worker is left with a straggler.
        std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
            return a.samples.size() > b.samples.size();
        });
        // Round the batch count up to a multiple of the worker count so the
        // last round does not leave workers idle; batches shrink to match.
        const size_t W = workers.size();
        size_t count = (jobs.size() + config.batch - 1) / config.batch;
        count = std::min(jobs.size(), (count + W - 1) / W * W);
        std::vector<std::span<Job>> batches;
        for (size_t i = 0, b = 0; b < count; ++b) {
            const size_t size = (jobs.size() - i) / (count - b) + ((jobs.size() - i) % (count - b) ? 1 : 0);
            batches.emplace_back(jobs.data() + i, size);
            i += size;
        }
        std::atomic<size_t> next{0};
        std::atomic<std::uint64_t> calls{0};
        std::vector<double> busy(workers.size(), 0.0);
        std::vector<Timing> timing(workers.size());
        std::exception_ptr failure;
        std::mutex failure_mutex;
        std::vector<std::thread> threads;
        const size_t used = std::min(workers.size(), std::max<size_t>(batches.size(), 1));
        for (size_t wi = 0; wi < used; ++wi)
            threads.emplace_back([&, wi] {
                try {
                    for (size_t i; (i = next.fetch_add(1)) < batches.size();) {
                        const auto t0 = Clock::now();
                        calls += decode_batch(*workers[wi], batches[i], timing[wi]);
                        busy[wi] += seconds(t0);
                    }
                } catch (...) {
                    std::lock_guard lock(failure_mutex);
                    failure = std::current_exception();
                    next = batches.size();
                }
            });
        for (auto& t : threads) t.join();
        if (failure) std::rethrow_exception(failure);
        std::lock_guard lock(stats_mutex);
        stats.wall_seconds += seconds(started);
        stats.worker_seconds += std::accumulate(busy.begin(), busy.end(), 0.0);
        stats.encoder_calls += calls;
        for (const auto& t : timing) { stats.fbank_seconds += t.fbank; stats.encoder_seconds += t.encoder; stats.search_seconds += t.search; }
        stats.batches += batches.size();
        stats.segments += jobs.size();
        for (const auto& j : jobs) stats.decoded_seconds += j.samples.size() / 16000.0;
    }
};

BatchOnnxAsr::BatchOnnxAsr(BatchAsrConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
BatchOnnxAsr::~BatchOnnxAsr() = default;

std::vector<SegmentResult> BatchOnnxAsr::transcribe(const std::vector<std::span<const float>>& clips) {
    std::vector<SegmentResult> results(clips.size());
    std::vector<Job> jobs;
    for (size_t i = 0; i < clips.size(); ++i) {
        for (float v : clips[i]) if (!std::isfinite(v)) throw std::invalid_argument("Non-finite audio sample");
        jobs.push_back({clips[i], 0.0, &results[i]});
        impl_->stats.audio_seconds += clips[i].size() / 16000.0;
    }
    impl_->run(jobs);
    return results;
}

std::vector<SegmentResult> BatchOnnxAsr::transcribe_long(std::span<const float> audio,
                                                         const SegmenterConfig& segmenter) {
    for (float v : audio) if (!std::isfinite(v)) throw std::invalid_argument("Non-finite audio sample");
    const auto started = Clock::now();
    const auto spans = segment_audio(audio, segmenter);
    std::vector<SegmentResult> results(spans.size());
    std::vector<Job> jobs;
    for (size_t i = 0; i < spans.size(); ++i)
        jobs.push_back({audio.subspan(spans[i].begin, spans[i].end - spans[i].begin),
                        spans[i].begin / 16000.0, &results[i]});
    impl_->stats.audio_seconds += audio.size() / 16000.0;
    const double segment_seconds = seconds(started);
    impl_->run(jobs);
    impl_->stats.wall_seconds += segment_seconds;
    return results;
}

const BatchAsrStats& BatchOnnxAsr::stats() const { return impl_->stats; }
int BatchOnnxAsr::workers() const { return static_cast<int>(impl_->workers.size()); }
std::string BatchOnnxAsr::model_type() const { return impl_->type; }
} // namespace captions

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\int8_zip\streaming_asr.cpp ===
#include <captions/streaming_asr.hpp>
#include "ort_graph.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace captions {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
using detail::Graph;
using detail::fbank_options;
void append_text(std::string& destination, const std::string& source) {
    if (source.empty()) return;
    if (!destination.empty()) destination += ' ';
    destination += source;
}
} // namespace

struct StreamingOnnxAsr::Impl {
    StreamingAsrConfig config;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "streaming-asr"};
    Graph encoder, decoder, joiner;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> states;
    Ort::Value decoder_out{nullptr};
    std::unique_ptr<knf::OnlineFbank> fbank;
    std::vector<float> features, pending, pre_roll;
    std::vector<std::int64_t> history;
    std::vector<std::string> tokens;
    std::string committed;
    StreamingAsrStats stats;
    int window = 0, shift = 0, context = 0, vocabulary = 0, frame_offset = 0, unknown = -1;
    int quiet_samples = 0;
    bool active = false, finished = false;
    std::string type;

    explicit Impl(StreamingAsrConfig c)
        : config(std::move(c)), encoder(env, config.encoder, checked_threads(config.threads)),
          decoder(env, config.decoder, 1), joiner(env, config.joiner, 1) {
        if (config.packet_ms < 10 || config.packet_ms > 160 ||
            config.pre_roll_ms < 0 || config.pre_roll_ms > 1000 ||
            config.gate_hangover_ms < 200 || config.gate_hangover_ms > 10000 ||
            !std::isfinite(config.gate_rms) || config.gate_rms < 0)
            throw std::invalid_argument("Invalid streaming packet/gate configuration");
        type = encoder.metadata("model_type");
        if (type != "zipformer" && type != "zipformer2")
            throw std::runtime_error("Expected streaming Zipformer transducer graphs");
        window = std::stoi(encoder.metadata("T"));
        shift = std::stoi(encoder.metadata("decode_chunk_len"));
        context = std::stoi(decoder.metadata("context_size"));
        vocabulary = std::stoi(decoder.metadata("vocab_size"));
        if (window < 1 || window > 1000 || shift < 1 || shift > window || context < 1 || context > 32 ||
            vocabulary < 3 || encoder.inputs.size() != encoder.outputs.size() ||
            decoder.inputs.size() != 1 || joiner.inputs.size() != 2)
            throw std::runtime_error("Unsupported streaming graph contract");
        tokens = detail::read_tokens(config.tokens, vocabulary, unknown);
        features.resize(static_cast<size_t>(window) * 80);
        pending.reserve(static_cast<size_t>(config.packet_ms) * 16);
        pre_roll.reserve(static_cast<size_t>(config.pre_roll_ms + config.packet_ms) * 16);
        new_segment();
    }
    static int checked_threads(int n) {
        if (n < 1 || n > 4) throw std::invalid_argument("ASR threads must be 1..4");
        return n;
    }
    void new_segment() {
        fbank = std::make_unique<knf::OnlineFbank>(fbank_options());
        states.clear();
        states.reserve(encoder.inputs.size() - 1);
        for (size_t i = 1; i < encoder.inputs.size(); ++i) {
            auto type_info = encoder.session.GetInputTypeInfo(i);
            auto info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            for (auto& d : shape) { if (d == -1) d = 1; }
            if (std::any_of(shape.begin(), shape.end(), [](auto d) { return d <= 0; }))
                throw std::runtime_error("Unsupported dynamic cache shape");
            if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                auto v = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
                std::fill_n(v.GetTensorMutableData<float>(), v.GetTensorTypeAndShapeInfo().GetElementCount(), 0.F);
                states.push_back(std::move(v));
            } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                auto v = Ort::Value::CreateTensor<std::int64_t>(allocator, shape.data(), shape.size());
                std::fill_n(v.GetTensorMutableData<std::int64_t>(), v.GetTensorTypeAndShapeInfo().GetElementCount(), 0);
                states.push_back(std::move(v));
            } else throw std::runtime_error("Unsupported cache element type");
        }
        // The original Zipformer export embeds y directly (Gather): -1 would
        // select the last vocabulary entry. Zipformer2 masks negative padding.
        history.assign(context, type == "zipformer" ? 0 : -1);
        history.back() = 0;
        decoder_out = Ort::Value{nullptr};
        frame_offset = 0;
        quiet_samples = 0;
        active = false;
    }
    void run_decoder() {
        const std::array<std::int64_t, 2> shape{1, context};
        auto value = Ort::Value::CreateTensor<std::int64_t>(memory,
            history.data() + history.size() - context, context, shape.data(), shape.size());
        auto out = decoder.run(std::span<const Ort::Value>(&value, 1));
        decoder_out = std::move(out[0]);
    }
    void decode_ready() {
        if (!decoder_out) run_decoder();
        while (fbank->NumFramesReady() - frame_offset >= window) {
            for (int t = 0; t < window; ++t)
                std::copy_n(fbank->GetFrame(frame_offset + t), 80, features.data() + t * 80);
            const std::array<std::int64_t, 3> shape{1, window, 80};
            std::vector<Ort::Value> in;
            in.reserve(1 + states.size());
            in.push_back(Ort::Value::CreateTensor<float>(memory, features.data(), features.size(), shape.data(), shape.size()));
            for (auto& state : states) in.push_back(std::move(state));
            auto started = Clock::now();
            auto out = encoder.run(in);
            if (stats.encoder_ms.size() < 100000) stats.encoder_ms.push_back(seconds(started) * 1000);
            ++stats.encoder_calls;
            for (size_t i = 0; i < states.size(); ++i) states[i] = std::move(out[i + 1]);
            const auto dims = out[0].GetTensorTypeAndShapeInfo().GetShape();
            if (dims.size() != 3 || dims[0] != 1 || dims[2] != 512)
                throw std::runtime_error("Unexpected encoder output dimensions");
            const float* encoded = out[0].GetTensorData<float>();
            const std::array<std::int64_t, 2> join_shape{1, dims[2]};
            for (std::int64_t t = 0; t < dims[1]; ++t) {
                std::array<Ort::Value, 2> join_inputs{
                    Ort::Value::CreateTensor<float>(memory, const_cast<float*>(encoded + t * dims[2]),
                        static_cast<size_t>(dims[2]), join_shape.data(), join_shape.size()),
                    Ort::Value::CreateTensor<float>(memory, decoder_out.GetTensorMutableData<float>(),
                        static_cast<size_t>(dims[2]), join_shape.data(), join_shape.size())};
                auto logits = joiner.run(join_inputs);
                if (logits[0].GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<size_t>(vocabulary))
                    throw std::runtime_error("Joiner/token vocabulary mismatch");
                const float* values = logits[0].GetTensorData<float>();
                const auto id = std::max_element(values, values + vocabulary) - values;
                // Icefall streaming transducers use at most one symbol/frame.
                if (id != 0 && id != unknown) { history.push_back(id); run_decoder(); }
            }
            frame_offset += shift;
            fbank->Pop(shift); // retain only overlap, never the whole recording
        }
    }
    void feed(std::span<const float> samples) {
        fbank->AcceptWaveform(16000, samples.data(), static_cast<int>(samples.size()));
        decode_ready();
    }
    std::string current_text() const {
        std::string s;
        for (size_t i = static_cast<size_t>(context); i < history.size(); ++i) s += tokens[history[i]];
        const std::string marker = "\xE2\x96\x81";
        size_t p = 0;
        while ((p = s.find(marker, p)) != std::string::npos) { s.replace(p, marker.size(), " "); ++p; }
        const auto begin = s.find_first_not_of(' '), end = s.find_last_not_of(' ');
        return begin == std::string::npos ? "" : s.substr(begin, end - begin + 1);
    }
    void close_segment() {
        if (!active) return;
        // The export needs right context to flush final speech frames. Padding
        // is synthesized here and included in compute time, never audio duration.
        std::vector<float> tail(static_cast<size_t>(window) * 160 + 4800, 0.F);
        feed(tail);
        fbank->InputFinished();
        decode_ready();
        append_text(committed, current_text());
        new_segment();
    }
    void packet(std::span<const float> samples) {
        const auto started = Clock::now();
        double power = 0;
        for (float value : samples) {
            if (!std::isfinite(value)) throw std::invalid_argument("Non-finite audio sample");
            power += static_cast<double>(value) * value;
        }
        const bool voiced = !config.energy_gate || power > samples.size() * config.gate_rms * config.gate_rms;
        if (!active && !voiced) {
            stats.gated_samples += samples.size();
            pre_roll.insert(pre_roll.end(), samples.begin(), samples.end());
            const size_t cap = static_cast<size_t>(config.pre_roll_ms) * 16;
            if (pre_roll.size() > cap) pre_roll.erase(pre_roll.begin(), pre_roll.end() - cap);
        } else {
            if (!active) { active = true; if (!pre_roll.empty()) feed(pre_roll); pre_roll.clear(); }
            feed(samples);
            quiet_samples = voiced ? 0 : quiet_samples + static_cast<int>(samples.size());
            if (config.energy_gate && quiet_samples >= config.gate_hangover_ms * 16) close_segment();
        }
        if (stats.packet_ms.size() < 100000) stats.packet_ms.push_back(seconds(started) * 1000);
    }
};

StreamingOnnxAsr::StreamingOnnxAsr(StreamingAsrConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
StreamingOnnxAsr::~StreamingOnnxAsr() = default;
void StreamingOnnxAsr::accept(std::span<const float> samples) {
    auto& s = *impl_;
    if (s.finished) throw std::logic_error("accept() after finish(); call reset()");
    const auto started = Clock::now();
    s.stats.input_samples += samples.size();
    const size_t packet = static_cast<size_t>(s.config.packet_ms) * 16;
    while (!samples.empty()) {
        const size_t n = std::min(packet - s.pending.size(), samples.size());
        s.pending.insert(s.pending.end(), samples.begin(), samples.begin() + n);
        samples = samples.subspan(n);
        if (s.pending.size() == packet) { s.packet(s.pending); s.pending.clear(); }
    }
    s.stats.compute_seconds += seconds(started);
}
void StreamingOnnxAsr::finish() {
    auto& s = *impl_;
    if (s.finished) return;
    const auto started = Clock::now();
    if (!s.pending.empty()) { s.packet(s.pending); s.pending.clear(); }
    s.close_segment();
    s.pre_roll.clear();
    s.finished = true;
    s.stats.compute_seconds += seconds(started);
}
void StreamingOnnxAsr::reset() {
    auto& s = *impl_;
    s.new_segment(); s.committed.clear(); s.pending.clear(); s.pre_roll.clear();
    s.finished = false; s.stats = {};
}
std::string StreamingOnnxAsr::text() const {
    std::string result = impl_->committed;
    append_text(result, impl_->current_text());
    return result;
}
const StreamingAsrStats& StreamingOnnxAsr::stats() const { return impl_->stats; }
int StreamingOnnxAsr::model_chunk_ms() const { return impl_->shift * 10; }
double StreamingOnnxAsr::first_window_ms() const { return (impl_->window - 1) * 10 + 17.5; }
std::string StreamingOnnxAsr::model_type() const { return impl_->type; }
} // namespace captions

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\int8_zip\third_party\asr_python\numpy\_core\tests\data\generate_umath_validation_data.cpp ===
#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <cstdio>
#include <ctime>
#include <vector>

struct ufunc {
    std::string name;
    double (*f32func)(double);
    long double (*f64func)(long double);
    float f32ulp;
    float f64ulp;
};

template <typename T>
T
RandomFloat(T a, T b)
{
    T random = ((T)rand()) / (T)RAND_MAX;
    T diff = b - a;
    T r = random * diff;
    return a + r;
}

template <typename T>
void
append_random_array(std::vector<T> &arr, T min, T max, size_t N)
{
    for (size_t ii = 0; ii < N; ++ii)
        arr.emplace_back(RandomFloat<T>(min, max));
}

template <typename T1, typename T2>
std::vector<T1>
computeTrueVal(const std::vector<T1> &in, T2 (*mathfunc)(T2))
{
    std::vector<T1> out;
    for (T1 elem : in) {
        T2 elem_d = (T2)elem;
        T1 out_elem = (T1)mathfunc(elem_d);
        out.emplace_back(out_elem);
    }
    return out;
}

/*
 * FP range:
 * [-inf, -maxflt, -1., -minflt, -minden, 0., minden, minflt, 1., maxflt, inf]
 */

#define MINDEN std::numeric_limits<T>::denorm_min()
#define MINFLT std::numeric_limits<T>::min()
#define MAXFLT std::numeric_limits<T>::max()
#define INF std::numeric_limits<T>::infinity()
#define qNAN std::numeric_limits<T>::quiet_NaN()
#define sNAN std::numeric_limits<T>::signaling_NaN()

template <typename T>
std::vector<T>
generate_input_vector(std::string func)
{
    std::vector<T> input = {MINDEN,  -MINDEN, MINFLT, -MINFLT, MAXFLT,
                            -MAXFLT, INF,     -INF,   qNAN,    sNAN,
                            -1.0,    1.0,     0.0,    -0.0};

    // [-1.0, 1.0]
    if ((func == "arcsin") || (func == "arccos") || (func == "arctanh")) {
        append_random_array<T>(input, -1.0, 1.0, 700);
    }
    // (0.0, INF]
    else if ((func == "log2") || (func == "log10")) {
        append_random_array<T>(input, 0.0, 1.0, 200);
        append_random_array<T>(input, MINDEN, MINFLT, 200);
        append_random_array<T>(input, MINFLT, 1.0, 200);
        append_random_array<T>(input, 1.0, MAXFLT, 200);
    }
    // (-1.0, INF]
    else if (func == "log1p") {
        append_random_array<T>(input, -1.0, 1.0, 200);
        append_random_array<T>(input, -MINFLT, -MINDEN, 100);
        append_random_array<T>(input, -1.0, -MINFLT, 100);
        append_random_array<T>(input, MINDEN, MINFLT, 100);
        append_random_array<T>(input, MINFLT, 1.0, 100);
        append_random_array<T>(input, 1.0, MAXFLT, 100);
    }
    // [1.0, INF]
    else if (func == "arccosh") {
        append_random_array<T>(input, 1.0, 2.0, 400);
        append_random_array<T>(input, 2.0, MAXFLT, 300);
    }
    // [-INF, INF]
    else {
        append_random_array<T>(input, -1.0, 1.0, 100);
        append_random_array<T>(input, MINDEN, MINFLT, 100);
        append_random_array<T>(input, -MINFLT, -MINDEN, 100);
        append_random_array<T>(input, MINFLT, 1.0, 100);
        append_random_array<T>(input, -1.0, -MINFLT, 100);
        append_random_array<T>(input, 1.0, MAXFLT, 100);
        append_random_array<T>(input, -MAXFLT, -100.0, 100);
    }

    std::random_shuffle(input.begin(), input.end());
    return input;
}

int
main()
{
    srand(42);
    std::vector<struct ufunc> umathfunc = {
            {"sin", sin, sin, 1.49, 1.00},
            {"cos", cos, cos, 1.49, 1.00},
            {"tan", tan, tan, 3.91, 1.00},
            {"arcsin", asin, asin, 3.12, 1.00},
            {"arccos", acos, acos, 2.1, 1.00},
            {"arctan", atan, atan, 2.3, 1.00},
            {"sinh", sinh, sinh, 1.55, 1.00},
            {"cosh", cosh, cosh, 2.48, 1.00},
            {"tanh", tanh, tanh, 1.38, 2.00},
            {"arcsinh", asinh, asinh, 1.01, 1.00},
            {"arccosh", acosh, acosh, 1.16, 1.00},
            {"arctanh", atanh, atanh, 1.45, 1.00},
            {"cbrt", cbrt, cbrt, 1.94, 2.00},
            //{"exp",exp,exp,3.76,1.00},
            {"exp2", exp2, exp2, 1.01, 1.00},
            {"expm1", expm1, expm1, 2.62, 1.00},
            //{"log",log,log,1.84,1.00},
            {"log10", log10, log10, 3.5, 1.00},
            {"log1p", log1p, log1p, 1.96, 1.0},
            {"log2", log2, log2, 2.12, 1.00},
    };

    for (int ii = 0; ii < umathfunc.size(); ++ii) {
        // ignore sin/cos
        if ((umathfunc[ii].name != "sin") && (umathfunc[ii].name != "cos")) {
            std::string fileName =
                    "umath-validation-set-" + umathfunc[ii].name + ".csv";
            std::ofstream txtOut;
            txtOut.open(fileName, std::ofstream::trunc);
            txtOut << "dtype,input,output,ulperrortol" << std::endl;

            // Single Precision
            auto f32in = generate_input_vector<float>(umathfunc[ii].name);
            auto f32out = computeTrueVal<float, double>(f32in,
                                                        umathfunc[ii].f32func);
            for (int jj = 0; jj < f32in.size(); ++jj) {
                txtOut << "np.float32" << std::hex << ",0x"
                       << *reinterpret_cast<uint32_t *>(&f32in[jj]) << ",0x"
                       << *reinterpret_cast<uint32_t *>(&f32out[jj]) << ","
                       << ceil(umathfunc[ii].f32ulp) << std::endl;
            }

            // Double Precision
            auto f64in = generate_input_vector<double>(umathfunc[ii].name);
            auto f64out = computeTrueVal<double, long double>(
                    f64in, umathfunc[ii].f64func);
            for (int jj = 0; jj < f64in.size(); ++jj) {
                txtOut << "np.float64" << std::hex << ",0x"
                       << *reinterpret_cast<uint64_t *>(&f64in[jj]) << ",0x"
                       << *reinterpret_cast<uint64_t *>(&f64out[jj]) << ","
                       << ceil(umathfunc[ii].f64ulp) << std::endl;
            }
            txtOut.close();
        }
    }
    return 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\naive\fourier_method.cpp ===

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "audio_loadnorm.hpp"
#include "audio_framing.hpp"

#include <power_spectrum/fast_fft.hpp>
#include <filter/pre_emphasis_filter.hpp>
#include <spectral/spectral_analysis.hpp>
#include <spectral/spectral_to_vowel.hpp>

static float median_of_valid_values(const std::vector<float>& values) {
    std::vector<float> valid_values;
    for (float value : values) {
        if (value > 0.0f && std::isfinite(value)) {
            valid_values.push_back(value);
        }
    }
    if (valid_values.empty()) {
        return 0.0f;
    }

    std::sort(valid_values.begin(), valid_values.end());
    return valid_values[valid_values.size() / 2];
}

static float compute_zero_crossing_rate(const std::vector<float>& frame) {
    if (frame.size() < 2) {
        return 0.0f;
    }

    size_t crossings = 0;
    for (size_t index = 1; index < frame.size(); ++index) {
        const bool crossed_zero = (frame[index] >= 0.0f && frame[index - 1] < 0.0f) ||
                                  (frame[index] < 0.0f && frame[index - 1] >= 0.0f);
        if (crossed_zero) {
            ++crossings;
        }
    }
    return static_cast<float>(crossings) / static_cast<float>(frame.size() - 1);
}

int main(int argc, char* argv[]) {
    const std::string sample_file = argc > 1
        ? argv[1]
        : "data/audiomnist/digit_7_sample_0000.wav";

    std::vector<float> audio_buffer;

    std::cout << "Loading: " << sample_file << "\n";

    const bool loaded_and_normalized = load_and_preprocess_audio(sample_file, audio_buffer);

    if (!loaded_and_normalized || audio_buffer.empty()) {
        std::cerr << "No audio samples were loaded.\n";
        return 1;
    }

    std::cout << "Loaded " << audio_buffer.size() << " samples at 16 kHz\n";

    std::vector<float> filtered_audio_buffer = pre_emphasis_filter(audio_buffer);

    const std::vector<std::vector<float>> frames = chop_into_frames(filtered_audio_buffer);
    if (frames.empty()) {
        std::cerr << "Audio is shorter than one analysis frame.\n";
        return 1;
    }

    FastFFT fast_fft;
    SpectralVowelDB vowel_database;
    std::vector<std::string> vowel_string;

    std::vector<SpectralFeatures> spectral_features_crop;
    std::vector<float> frame_energies;
    std::vector<float> frame_zcrs;
    constexpr float kConsonantSpectralCentroidThreshold = 3200.0f; // Hz - above this is fricative
    constexpr float kConsonantZcrThreshold = 0.35f;

    // Batch collect spectral features for all frames
    for (const std::vector<float>& frame : frames) {
        // Compute energy
        float energy = 0.0f;
        for (float s : frame) energy += s * s;
        energy = std::sqrt(energy / frame.size());
        frame_energies.push_back(energy);
        frame_zcrs.push_back(compute_zero_crossing_rate(frame));

        // Compute power spectrum and extract spectral features
        const std::vector<float> power_spectrum = fast_fft.compute_power_spectrum(frame);
        SpectralFeatures features = SpectralAnalyzer::extract_features(power_spectrum);
        spectral_features_crop.push_back(features);
    }

    const float maximum_energy = *std::max_element(frame_energies.begin(), frame_energies.end());
    const float scaled_energy_threshold = maximum_energy * 0.08f;
    const float energy_threshold = scaled_energy_threshold > 0.00015f
        ? scaled_energy_threshold
        : 0.00015f;

    // Median smoothing reduces frame-to-frame spectral jitter while preserving silence.
    for (size_t frame_index = 0; frame_index < spectral_features_crop.size(); ++frame_index) {
        const SpectralFeatures& raw_features = spectral_features_crop[frame_index];

        // Frame rejection criteria:
        // 1. Too quiet (below energy threshold) -> Silence
        // 2. High ZCR or high spectral centroid -> Consonant/Fricative
        // 3. Not voiced -> Unvoiced consonant/silence
        if (frame_energies[frame_index] < energy_threshold ||
            frame_zcrs[frame_index] > kConsonantZcrThreshold ||
            raw_features.centroid_hz > kConsonantSpectralCentroidThreshold ||
            !raw_features.is_voiced) {
            vowel_string.push_back("SIL");
            continue;
        }

        // Smooth spectral features using median of neighborhood (frame_index-1, frame_index, frame_index+1)
        const size_t first = frame_index == 0 ? 0 : frame_index - 1;
        const size_t last = frame_index + 1 < spectral_features_crop.size()
            ? frame_index + 1
            : spectral_features_crop.size() - 1;

        // Collect neighborhood centroids and pitches
        std::vector<float> neighbor_centroids;
        std::vector<float> neighbor_pitches;
        std::vector<float> neighbor_tilts;

        for (size_t neighbor = first; neighbor <= last; ++neighbor) {
            neighbor_centroids.push_back(spectral_features_crop[neighbor].centroid_hz);
            neighbor_pitches.push_back(spectral_features_crop[neighbor].pitch_f0_hz);
            neighbor_tilts.push_back(spectral_features_crop[neighbor].spectral_tilt);
        }

        // Median smoothing
        SpectralFeatures smoothed_features{
            median_of_valid_values(neighbor_centroids),
            median_of_valid_values(neighbor_pitches),
            true, // Already determined to be voiced
            median_of_valid_values(neighbor_tilts),
            raw_features.voicing_confidence
        };

        // Match to vowel database
        const SpectralVowelDB::MatchResult result = vowel_database.find_nearest_phoneme(smoothed_features);
        vowel_string.push_back(result.key == "UNK" ? "SIL" : result.key);
    }

    std::cout << "\n=== FOURIER/SPECTRAL METHOD RESULTS ===\n";
    std::cout << "Vowel string (all predicted frames): ";
    if (vowel_string.empty()) {
        std::cout << "(none)";
    } else {
        for (const std::string& vowel : vowel_string) {
            std::cout << vowel << ' ';
        }
    }
    std::cout << "\n";

    std::cout << "Vowel string (collapsed): ";
    std::string previous_vowel;
    bool printed_vowel = false;
    for (const std::string& vowel : vowel_string) {
        if (vowel == previous_vowel) {
            continue;
        }
        std::cout << vowel << ' ';
        previous_vowel = vowel;
        printed_vowel = true;
    }
    if (!printed_vowel) {
        std::cout << "(none)";
    }
    std::cout << "\n";

    return 0;
}



// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\captions\models\naive\lpc_method.cpp ===

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "audio_loadnorm.hpp"
#include "audio_framing.hpp"


#include <formants/find_formants.hpp>
#include <formants/formant_to_vowel.hpp>
#include <filter/pre_emphasis_filter.hpp>

static float median_of_valid_values(const std::vector<float>& values) {
    std::vector<float> valid_values;
    for (float value : values) {
        if (value > 0.0f && std::isfinite(value)) {
            valid_values.push_back(value);
        }
    }
    if (valid_values.empty()) {
        return 0.0f;
    }

    std::sort(valid_values.begin(), valid_values.end());
    return valid_values[valid_values.size() / 2];
}

static float compute_zero_crossing_rate(const std::vector<float>& frame) {
    if (frame.size() < 2) {
        return 0.0f;
    }

    size_t crossings = 0;
    for (size_t index = 1; index < frame.size(); ++index) {
        const bool crossed_zero = (frame[index] >= 0.0f && frame[index - 1] < 0.0f) ||
                                  (frame[index] < 0.0f && frame[index - 1] >= 0.0f);
        if (crossed_zero) {
            ++crossings;
        }
    }
    return static_cast<float>(crossings) / static_cast<float>(frame.size() - 1);
}

int main(int argc, char* argv[]) {
    const std::string sample_file = argc > 1
        ? argv[1]
        : "data/audiomnist/digit_7_sample_0000.wav";

    std::vector<float> audio_buffer;

    std::cout << "Loading: " << sample_file << "\n";

    const bool loaded_and_normalized = load_and_preprocess_audio(sample_file, audio_buffer);

    if (!loaded_and_normalized || audio_buffer.empty()) {
        std::cerr << "No audio samples were loaded.\n";
        return 1;
    }

    std::cout << "Loaded " << audio_buffer.size() << " samples at 16 kHz\n";

    std::vector<float> filtered_audio_buffer = pre_emphasis_filter(audio_buffer);

    const std::vector<std::vector<float>> frames = chop_into_frames(filtered_audio_buffer);
    if (frames.empty()) {
        std::cerr << "Audio is shorter than one analysis frame.\n";
        return 1;
    }



    
    FormantTracker formant_tracker;
    FormantVectorDB vowel_database;
    std::vector<std::string> vowel_string;

    

    std::vector<std::array<float, 3>> formant_crop;
    std::vector<float> frame_energies;
    std::vector<float> frame_zcrs;
    constexpr float kConsonantZcrThreshold = 0.35f;

    
    // Batch collect the formants into this crop.
    // Compute all of the formants for the entire frames as well as muting the formants giving a moving average.
    for (const std::vector<float>& frame : frames) {



        float energy = 0.0f;
        for (float s : frame) energy += s * s;
        energy = std::sqrt(energy / frame.size());
        frame_energies.push_back(energy);
        frame_zcrs.push_back(compute_zero_crossing_rate(frame));

        const std::vector<Formant> formants = formant_tracker.extract_formants(frame);
        if (formants.size() < 2) {
            formant_crop.push_back({0.0f, 0.0f, 0.0f});
        } else {
            formant_crop.push_back({
                formants[0].frequency,
                formants[1].frequency,
                formants.size() > 2 ? formants[2].frequency : 0.0f
            });
        }
    }

    const float maximum_energy = *std::max_element(frame_energies.begin(), frame_energies.end());
    const float scaled_energy_threshold = maximum_energy * 0.08f;
    const float energy_threshold = scaled_energy_threshold > 0.00015f
        ? scaled_energy_threshold
        : 0.00015f;

    // Median smoothing reduces frame-to-frame LPC jitter while preserving silence.
    for (size_t frame_index = 0; frame_index < formant_crop.size(); ++frame_index) {
        const std::array<float, 3>& raw_formants = formant_crop[frame_index];
        if (frame_energies[frame_index] < energy_threshold ||
            frame_zcrs[frame_index] > kConsonantZcrThreshold ||
            raw_formants[0] <= 0.0f || raw_formants[1] <= 0.0f) {
            vowel_string.push_back("SIL");
            continue;
        }

        const size_t first = frame_index == 0 ? 0 : frame_index - 1;
        const size_t last = frame_index + 1 < formant_crop.size()
            ? frame_index + 1
            : formant_crop.size() - 1;
        std::array<float, 3> smoothed_formants{};
        for (size_t formant_index = 0; formant_index < smoothed_formants.size(); ++formant_index) {
            std::vector<float> neighborhood;
            for (size_t neighbor = first; neighbor <= last; ++neighbor) {
                neighborhood.push_back(formant_crop[neighbor][formant_index]);
            }
            smoothed_formants[formant_index] = median_of_valid_values(neighborhood);
        }

        const FormantVectorDB::MatchResult result = vowel_database.find_nearest_phoneme(
            smoothed_formants[0], smoothed_formants[1], smoothed_formants[2]);
        vowel_string.push_back(result.key == "UNK" ? "SIL" : result.key);
    }

    std::cout << "Vowel string (all predicted frames): ";
    if (vowel_string.empty()) {
        std::cout << "(none)";
    } else {
        for (const std::string& vowel : vowel_string) {
            std::cout << vowel << ' ';
        }
    }
    std::cout << "\n";

    std::cout << "Vowel string (collapsed): ";
    std::string previous_vowel;
    bool printed_vowel = false;
    for (const std::string& vowel : vowel_string) {
        if (vowel == previous_vowel) {
            continue;
        }
        std::cout << vowel << ' ';
        previous_vowel = vowel;
        printed_vowel = true;
    }
    if (!printed_vowel) {
        std::cout << "(none)";
    }
    std::cout << "\n";
    


    return 0;
}

