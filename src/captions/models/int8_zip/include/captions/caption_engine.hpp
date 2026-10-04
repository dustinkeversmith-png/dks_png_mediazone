#pragma once
// Production captioning API. One model: the streaming Zipformer2 transducer in
// models/int8_zip/models/librispeech (INT8, ONNX Runtime CPU), decoded with
// modified beam search (beam 4) on a multithreaded, length-sorted batch
// pipeline. For live microphone captions see StreamingOnnxAsr.

#include <captions/batch_asr.hpp>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace captions {

// Model directory used when CaptionEngineConfig::models is empty: the
// Zipformer2 checkpoint inside the source tree, fixed at build time.
std::filesystem::path default_model_dir();

// The encoder/decoder/joiner/tokens files of the Zipformer2 checkpoint in `dir`.
struct ModelFiles { std::filesystem::path encoder, decoder, joiner, tokens; };
ModelFiles model_files(const std::filesystem::path& dir);

struct CaptionEngineConfig {
    std::filesystem::path models;  // empty = default_model_dir()
    int workers = 6;
    int threads_per_worker = 2;
    int batch = 16;
    int beam = 4;
    SegmenterConfig segmenter;
    std::filesystem::path ffmpeg;  // empty = dependencies/ffmpeg/bin/ffmpeg.exe above cwd, else PATH
};

class CaptionEngine {
public:
    explicit CaptionEngine(CaptionEngineConfig config = {});
    ~CaptionEngine();
    CaptionEngine(const CaptionEngine&) = delete;
    CaptionEngine& operator=(const CaptionEngine&) = delete;

    // Any container/codec ffmpeg reads (video included); PCM WAV is read
    // directly. Timestamps are seconds from the start of the file.
    std::vector<SegmentResult> transcribe_file(const std::filesystem::path& path);
    // Normalized float samples, 16 kHz mono.
    std::vector<SegmentResult> transcribe_pcm(std::span<const float> audio_16k_mono);
    // Many independent clips decoded in one batched pass; one result per clip.
    std::vector<SegmentResult> transcribe_clips(const std::vector<std::span<const float>>& clips);

    [[nodiscard]] const BatchAsrStats& stats() const;
    [[nodiscard]] const CaptionEngineConfig& config() const;
private:
    CaptionEngineConfig config_;
    std::unique_ptr<BatchOnnxAsr> asr_;
};

// 16 kHz mono float from any media file (ffmpeg pipe; WAV read natively).
std::vector<float> load_audio(const std::filesystem::path& path, const std::filesystem::path& ffmpeg = {});

// ---- Caption formatting --------------------------------------------------
struct Cue { double start = 0, end = 0; std::string text; };
struct CueOptions {
    std::size_t line_chars = 42;  // two lines per cue at most
    double max_seconds = 6.0;
    double gap_seconds = 0.8;     // a pause longer than this starts a new cue
    double linger_seconds = 0.4;  // hold after the last word, never into the next cue
    bool sentence_case = true;    // false keeps the model's uppercase output
};
std::vector<Cue> build_cues(const std::vector<SegmentResult>& segments, const CueOptions& options = {});

enum class CaptionFormat { srt, vtt, txt, json };
CaptionFormat parse_format(const std::string& name);  // throws on unknown names
const char* extension(CaptionFormat format);           // ".srt", ...
std::string render(CaptionFormat format, const std::vector<SegmentResult>& segments,
                   const std::vector<Cue>& cues);
} // namespace captions
