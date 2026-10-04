#include <captions/caption_engine.hpp>
#include <input/wav_reader.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#ifndef CAPTIONS_DEFAULT_MODEL_DIR
#define CAPTIONS_DEFAULT_MODEL_DIR "src/captions/models/int8_zip/models/librispeech"
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

fs::path default_model_dir() {
    fs::path dir = CAPTIONS_DEFAULT_MODEL_DIR;
    if (fs::exists(dir)) return dir;
    // Relocated build: look for the source-tree layout above the working directory.
    for (fs::path base = fs::current_path();; base = base.parent_path()) {
        const auto candidate = base / "src/captions/models/int8_zip/models/librispeech";
        if (fs::exists(candidate)) return candidate;
        if (base == base.parent_path()) break;
    }
    return dir;
}

ModelFiles model_files(const fs::path& dir) {
    if (!fs::is_directory(dir))
        throw std::runtime_error("Model directory not found: " + dir.string() +
                                 " (fetch it with src/captions/models/int8_zip/scripts/fetch_streaming_asr.py)");
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
    asr_ = std::make_unique<BatchOnnxAsr>(c);
}
CaptionEngine::~CaptionEngine() = default;

std::vector<SegmentResult> CaptionEngine::transcribe_file(const fs::path& path) {
    const auto audio = load_audio(path, config_.ffmpeg);
    return transcribe_pcm(audio);
}
std::vector<SegmentResult> CaptionEngine::transcribe_pcm(std::span<const float> audio) {
    return asr_->transcribe_long(audio, config_.segmenter);
}
std::vector<SegmentResult> CaptionEngine::transcribe_clips(const std::vector<std::span<const float>>& clips) {
    return asr_->transcribe(clips);
}
const BatchAsrStats& CaptionEngine::stats() const { return asr_->stats(); }
const CaptionEngineConfig& CaptionEngine::config() const { return config_; }

std::vector<Cue> build_cues(const std::vector<SegmentResult>& segments, const CueOptions& o) {
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
                words.push_back(o.sentence_case ? display_word(w->text, sentence_start) : w->text);
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
                    << ",\"end\":" << s.words[j].end << '}';
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
