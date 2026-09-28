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
