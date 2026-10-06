#include <captions/caption_engine.hpp>
#include <input/wav_reader.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <thread>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#ifndef CAPTIONS_MODEL_ROOT
#define CAPTIONS_MODEL_ROOT "src/captions/models/int8_zip/models"
#endif

namespace fs = std::filesystem;
namespace captions {
namespace {
std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}
fs::path graph(const fs::path& dir, const std::string& prefix) {
    fs::path result;
    for (const auto& entry : fs::directory_iterator(dir)) {
        const auto name = entry.path().filename().string();
        if (name.starts_with(prefix) && name.ends_with(".int8.onnx")) {
            if (!result.empty()) throw std::runtime_error("Ambiguous " + prefix + " graph in " + dir.string());
            result = entry.path();
        }
    }
    if (result.empty()) throw std::runtime_error("Missing INT8 " + prefix + " graph in " + dir.string());
    return result;
}
fs::path find_ffmpeg(const fs::path& requested) {
    if (!requested.empty()) return requested;
    for (fs::path dir = fs::current_path();; dir = dir.parent_path()) {
        const auto candidate = dir / "dependencies/ffmpeg/bin/ffmpeg.exe";
        if (fs::exists(candidate)) return candidate;
        if (dir == dir.parent_path()) break;
    }
    return "ffmpeg";
}
std::string quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') { out += '\\'; out += static_cast<char>(c); }
        else if (c == '\n') out += "\\n";
        else if (c < 32) out += ' ';
        else out += static_cast<char>(c);
    }
    return out + '"';
}
std::string stamp(double t, char separator) {
    const auto ms = static_cast<long long>(std::llround(std::max(0.0, t) * 1000));
    std::ostringstream s;
    s << std::setfill('0') << std::setw(2) << ms / 3600000 << ':' << std::setw(2) << ms / 60000 % 60 << ':'
      << std::setw(2) << ms / 1000 % 60 << separator << std::setw(3) << ms % 1000;
    return s.str();
}
// The model emits uppercase without punctuation: lowercase it, restore the
// pronoun "I", and capitalize the first word after a pause.
std::string display_word(const std::string& word, bool capitalize) {
    std::string w = lower(word);
    if (!w.empty() && (capitalize || w == "i" || w.rfind("i'", 0) == 0))
        w[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(w[0])));
    return w;
}
} // namespace

// Model root fixed at build time; relocated builds search above the working
// directory for the source-tree layout instead.
fs::path model_root() {
    fs::path root = CAPTIONS_MODEL_ROOT;
    if (fs::exists(root)) return root;
    for (fs::path base = fs::current_path();; base = base.parent_path()) {
        const auto candidate = base / "src/captions/models/int8_zip/models";
        if (fs::exists(candidate)) return candidate;
        if (base == base.parent_path()) break;
    }
    return root;
}
fs::path default_model_dir() { return model_root() / "parakeet-tdt-110m"; }
fs::path live_model_dir() { return model_root() / "nemo-streaming-480ms"; }

ModelFiles model_files(const fs::path& dir) {
    if (!fs::is_directory(dir))
        throw std::runtime_error("Model directory not found: " + dir.string() +
                                 " (run src/captions/models/int8_zip/scripts/fetch_models.ps1)");
    // A CTC export is one graph (model.int8.onnx); transducers have three.
    if (fs::exists(dir / "model.int8.onnx")) return {dir / "model.int8.onnx", {}, {}, dir / "tokens.txt"};
    return {graph(dir, "encoder"), graph(dir, "decoder"), graph(dir, "joiner"), dir / "tokens.txt"};
}

std::vector<float> load_audio(const fs::path& path, const fs::path& ffmpeg_path) {
    if (!fs::exists(path)) throw std::runtime_error("Input not found: " + path.string());
    if (lower(path.extension().string()) == ".wav") {
        try { return read_wav_mono(path, 16000).samples; }
        catch (const std::exception&) { /* compressed/extensible WAV: fall through to ffmpeg */ }
    }
    const auto ffmpeg = find_ffmpeg(ffmpeg_path);
    // cmd.exe strips the outer quote pair, so the whole command is quoted once more.
    const std::string command = "\"\"" + ffmpeg.string() + "\" -nostdin -v error -i \"" + path.string() +
                                "\" -vn -ac 1 -ar 16000 -f f32le -\"";
#ifdef _WIN32
    FILE* pipe = _popen(command.c_str(), "rb");
#else
    FILE* pipe = popen(command.c_str(), "r");
#endif
    if (!pipe) throw std::runtime_error("Cannot start ffmpeg");
    std::vector<float> samples, buffer(1 << 16);
    size_t n;
    while ((n = std::fread(buffer.data(), sizeof(float), buffer.size(), pipe)) > 0)
        samples.insert(samples.end(), buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(n));
#ifdef _WIN32
    const int status = _pclose(pipe);
#else
    const int status = pclose(pipe);
#endif
    if (status != 0 || samples.empty())
        throw std::runtime_error("ffmpeg could not decode " + path.string() + " (set --ffmpeg <path>)");
    return samples;
}

CaptionEngine::CaptionEngine(CaptionEngineConfig config) : config_(std::move(config)) {
    if (config_.models.empty()) config_.models = default_model_dir();
    const auto files = model_files(config_.models);
    BatchAsrConfig c;
    c.encoder = files.encoder; c.decoder = files.decoder; c.joiner = files.joiner; c.tokens = files.tokens;
    c.workers = config_.workers;
    c.threads_per_worker = config_.threads_per_worker;
    c.batch = config_.batch;
    c.beam = config_.beam;
    c.share_sessions = config_.share_sessions;
    c.hotwords = config_.hotwords;
    c.hotword_boost = config_.hotword_boost;
    c.hotword_start = config_.hotword_start;
    asr_ = std::make_unique<BatchOnnxAsr>(c);
    if (config_.use_vad) {
        if (config_.vad.empty()) config_.vad = model_root() / "silero-vad" / "silero_vad.onnx";
        if (!fs::exists(config_.vad))
            throw std::runtime_error("VAD model not found: " + config_.vad.string() + " (or disable it with use_vad = false)");
        vad_ = std::make_unique<SileroVad>(config_.vad, 4);
    }
}
CaptionEngine::~CaptionEngine() = default;

std::vector<SegmentResult> CaptionEngine::transcribe_file(const fs::path& path) {
    const auto audio = load_audio(path, config_.ffmpeg);
    return transcribe_pcm(audio);
}

std::vector<SegmentResult> CaptionEngine::transcribe_pcm(std::span<const float> audio) {
    auto segmenter = config_.segmenter;
    if (config_.adaptive_segments) {
        // Aim for a few segments per worker; never shorter than 4 s (context) or
        // longer than the configured cap.
        const double seconds = audio.size() / 16000.0;
        const double per_worker = asr_->batch() > 1 ? 4.0 : 2.0;  // enough to fill each worker's batches
        const double target = seconds / (std::max(1, asr_->workers()) * per_worker);
        segmenter.max_seconds = std::clamp(target, 4.0, config_.segmenter.max_seconds);
        segmenter.min_seconds = std::min(segmenter.min_seconds, segmenter.max_seconds * 0.4);
    }
    return run({audio}, segmenter).front();
}

std::vector<std::vector<SegmentResult>> CaptionEngine::transcribe_batch(
    const std::vector<std::span<const float>>& recordings) {
    return run(recordings, config_.segmenter);
}

// VAD (recordings in parallel) -> segmentation -> one batched decode of every
// segment of every recording -> confidence/VAD filter -> results per recording.
std::vector<std::vector<SegmentResult>> CaptionEngine::run(const std::vector<std::span<const float>>& recordings,
                                                           const SegmenterConfig& segmenter) {
    for (const auto& r : recordings) audio_seconds_ += r.size() / 16000.0;
    std::vector<std::vector<float>> prob(recordings.size());
    if (vad_) {
        const auto started = std::chrono::steady_clock::now();
        std::atomic<size_t> next{0};
        std::vector<std::thread> threads;
        std::exception_ptr failure;
        std::mutex failure_mutex;
        const size_t n = std::min<size_t>(recordings.size(), std::max(1U, std::thread::hardware_concurrency()));
        for (size_t t = 0; t < n; ++t)
            threads.emplace_back([&] {
                try {
                    for (size_t i; (i = next.fetch_add(1)) < recordings.size();) prob[i] = vad_->probabilities(recordings[i]);
                } catch (...) { std::lock_guard lock(failure_mutex); failure = std::current_exception(); }
            });
        for (auto& t : threads) t.join();
        if (failure) std::rethrow_exception(failure);
        vad_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
    struct Piece { size_t recording; Span span; float speech; };
    std::vector<Piece> pieces;
    std::vector<std::span<const float>> clips;
    for (size_t r = 0; r < recordings.size(); ++r)
        for (const auto& s : segment_audio(recordings[r], segmenter, vad_ ? &prob[r] : nullptr)) {
            float speech = 1.F;  // without a VAD every segment counts as speech
            if (vad_ && !prob[r].empty()) {
                const size_t b = s.begin / SileroVad::kChunk;
                const size_t e = std::clamp<size_t>((s.end + SileroVad::kChunk - 1) / SileroVad::kChunk, b + 1, prob[r].size());
                double sum = 0;
                for (size_t i = std::min(b, e - 1); i < e; ++i) sum += prob[r][i];
                speech = static_cast<float>(sum / static_cast<double>(e - std::min(b, e - 1)));
            }
            pieces.push_back({r, s, speech});
            clips.push_back(recordings[r].subspan(s.begin, s.end - s.begin));
        }
    auto decoded = asr_->transcribe(clips);
    std::vector<std::vector<SegmentResult>> out(recordings.size());
    for (size_t i = 0; i < pieces.size(); ++i) {
        auto& seg = decoded[i];
        const double offset = pieces[i].span.begin / 16000.0;
        seg.start += offset; seg.end += offset;
        for (auto& w : seg.words) { w.start += offset; w.end += offset; }
        seg.speech_probability = pieces[i].speech;
        if (keep_segment(seg, config_.min_confidence, config_.speech_floor)) out[pieces[i].recording].push_back(std::move(seg));
    }
    return out;
}

bool keep_segment(const SegmentResult& s, float min_confidence, float speech_floor) {
    if (s.words.empty()) return false;
    double sum = 0;
    for (const auto& w : s.words) sum += w.confidence;
    const bool unsure = sum / static_cast<double>(s.words.size()) < min_confidence;
    return !(unsure && s.speech_probability < speech_floor);
}
std::vector<SegmentResult> CaptionEngine::transcribe_clips(const std::vector<std::span<const float>>& clips) {
    return asr_->transcribe(clips);
}
BatchAsrStats CaptionEngine::stats() const {
    auto s = asr_->stats();
    s.vad_seconds = vad_seconds_;
    if (audio_seconds_ > 0) s.audio_seconds = audio_seconds_;
    s.wall_seconds += vad_seconds_;
    return s;
}
const CaptionEngineConfig& CaptionEngine::config() const { return config_; }
const HotwordBiaser& CaptionEngine::hotwords() const { return asr_->hotwords(); }

std::vector<Cue> build_cues(const std::vector<SegmentResult>& segments, const CueOptions& o) {
    bool cased = false;
    for (const auto& s : segments)
        cased = cased || std::any_of(s.text.begin(), s.text.end(), [](unsigned char ch) { return std::islower(ch); });
    std::vector<Cue> cues;
    for (const auto& seg : segments) {
        bool sentence_start = true;
        std::vector<const Word*> current;
        size_t chars = 0;
        auto flush = [&] {
            if (current.empty()) return;
            std::vector<std::string> words;
            size_t total = 0;
            for (const auto* w : current) {
                // Models with their own casing (any lowercase letter) are kept verbatim.
                words.push_back(o.sentence_case && !cased ? display_word(w->text, sentence_start) : w->text);
                sentence_start = false;
                total += words.back().size() + 1;
            }
            // One line if it fits, otherwise two balanced lines.
            std::string text;
            size_t line = 0;
            bool broke = total <= o.line_chars + 1;
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
            const bool gap = !current.empty() && w.start - current.back()->end > o.gap_seconds;
            const bool full = !current.empty() && (chars + w.text.size() + 1 > o.line_chars * 2 ||
                                                  w.end - current.front()->start > o.max_seconds);
            if (gap || full) { flush(); if (gap) sentence_start = true; }
            current.push_back(&w);
            chars += w.text.size() + 1;
        }
        flush();
    }
    for (size_t i = 0; i < cues.size(); ++i) {
        double end = std::max(cues[i].end + o.linger_seconds, cues[i].start + 0.7);
        if (i + 1 < cues.size()) end = std::min(end, cues[i + 1].start);
        cues[i].end = std::max(end, cues[i].start + 0.001);
    }
    return cues;
}

CaptionFormat parse_format(const std::string& name) {
    const auto n = lower(name);
    if (n == "srt") return CaptionFormat::srt;
    if (n == "vtt") return CaptionFormat::vtt;
    if (n == "txt") return CaptionFormat::txt;
    if (n == "json") return CaptionFormat::json;
    throw std::invalid_argument("Unknown caption format '" + name + "' (srt, vtt, txt, json)");
}
const char* extension(CaptionFormat f) {
    switch (f) {
    case CaptionFormat::srt: return ".srt";
    case CaptionFormat::vtt: return ".vtt";
    case CaptionFormat::txt: return ".txt";
    case CaptionFormat::json: return ".json";
    }
    return "";
}
std::string render(CaptionFormat f, const std::vector<SegmentResult>& segments, const std::vector<Cue>& cues) {
    std::ostringstream out;
    switch (f) {
    case CaptionFormat::srt:
        for (size_t i = 0; i < cues.size(); ++i)
            out << i + 1 << '\n' << stamp(cues[i].start, ',') << " --> " << stamp(cues[i].end, ',') << '\n'
                << cues[i].text << "\n\n";
        break;
    case CaptionFormat::vtt:
        out << "WEBVTT\n\n";
        for (const auto& c : cues) out << stamp(c.start, '.') << " --> " << stamp(c.end, '.') << '\n' << c.text << "\n\n";
        break;
    case CaptionFormat::txt:
        for (const auto& c : cues) {
            std::string line = c.text;
            std::replace(line.begin(), line.end(), '\n', ' ');
            out << line << '\n';
        }
        break;
    case CaptionFormat::json:
        out << std::fixed << std::setprecision(3) << "{\"segments\":[\n";
        for (size_t i = 0; i < segments.size(); ++i) {
            const auto& s = segments[i];
            out << "{\"start\":" << s.start << ",\"end\":" << s.end << ",\"text\":" << quote(s.text) << ",\"words\":[";
            for (size_t j = 0; j < s.words.size(); ++j)
                out << (j ? "," : "") << "{\"word\":" << quote(s.words[j].text) << ",\"start\":" << s.words[j].start
                    << ",\"end\":" << s.words[j].end << ",\"confidence\":" << s.words[j].confidence << '}';
            out << "]}" << (i + 1 < segments.size() ? ",\n" : "\n");
        }
        out << "],\n\"cues\":[\n";
        for (size_t i = 0; i < cues.size(); ++i)
            out << "{\"start\":" << cues[i].start << ",\"end\":" << cues[i].end << ",\"text\":" << quote(cues[i].text)
                << '}' << (i + 1 < cues.size() ? ",\n" : "\n");
        out << "]}\n";
        break;
    }
    return out.str();
}
} // namespace captions
