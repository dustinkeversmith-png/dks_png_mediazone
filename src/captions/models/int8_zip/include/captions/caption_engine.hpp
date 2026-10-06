#pragma once
// Production captioning API, CPU only (ONNX Runtime, INT8 models):
//   - NVIDIA Parakeet TDT 110M (models/parakeet-tdt-110m): FastConformer
//     encoder trained on ~36k hours of multi-domain English, with punctuation
//     and casing; offline, full-context decoding of each segment.
//   - Silero VAD v5 (models/silero-vad): speech segmentation that ignores
//     silence and steady noise, plus the hallucination filter (keep_segment).
// Segments of one or many recordings are decoded by a shared-model worker
// pool. Live microphone captions use StreamingOnnxAsr with the NeMo
// cache-aware streaming FastConformer in models/nemo-streaming-480ms (live_model_dir()).

#include <captions/batch_asr.hpp>
#include <captions/vad.hpp>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace captions {

// Model directories inside the source tree (fixed at build time; relocated
// builds search upward from the working directory).
std::filesystem::path model_root();
std::filesystem::path default_model_dir();  // batch: model_root()/parakeet-tdt-110m
std::filesystem::path live_model_dir();     // live:  model_root()/nemo-streaming-480ms

// Graph files in `dir`: encoder/decoder/joiner (transducers) or model (CTC).
struct ModelFiles { std::filesystem::path encoder, decoder, joiner, tokens; };
ModelFiles model_files(const std::filesystem::path& dir);

struct CaptionEngineConfig {
    std::filesystem::path models;  // empty = default_model_dir(); a Zipformer2 dir also works
    int workers = 0;               // 0 = one per hardware thread
    int threads_per_worker = 1;    // each worker runs its own encoder call on its own thread
    int batch = 0;                 // segments per encoder call; 0 = per model (Parakeet 1, Zipformer2 16)
    int beam = 4;                  // Zipformer2 modified beam search; Parakeet decodes greedy TDT
    bool share_sessions = true;    // one model copy for all workers (load once)
    // Short recordings are cut into shorter segments so every worker has work.
    bool adaptive_segments = true;
    // Neural voice activity detection for segmentation; empty path =
    // model_root()/silero-vad/silero_vad.onnx, use_vad = false = RMS gate.
    bool use_vad = true;
    std::filesystem::path vad;
    float min_confidence = 0.6F;   // see keep_segment(); 0 disables the filter
    float speech_floor = 0.3F;
    // Domain terms (names, jargon) to favour; see hotwords.hpp. Boost is the
    // logit bonus per matching token.
    std::vector<HotwordPhrase> hotwords;
    float hotword_boost = 2.0F;
    float hotword_start = 0.25F;  // fraction of the boost given to a phrase's first token
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
    // Many whole recordings (e.g. a folder of files) in one batched pass:
    // segmented, decoded together, filtered; results per recording.
    std::vector<std::vector<SegmentResult>> transcribe_batch(const std::vector<std::span<const float>>& recordings);
    // Many independent clips decoded in one batched pass; one result per clip.
    std::vector<SegmentResult> transcribe_clips(const std::vector<std::span<const float>>& clips);

    [[nodiscard]] BatchAsrStats stats() const;
    [[nodiscard]] const HotwordBiaser& hotwords() const;
    [[nodiscard]] const CaptionEngineConfig& config() const;
private:
    CaptionEngineConfig config_;
    std::unique_ptr<BatchOnnxAsr> asr_;
    std::unique_ptr<SileroVad> vad_;
    double vad_seconds_ = 0, audio_seconds_ = 0;
    std::vector<std::vector<SegmentResult>> run(const std::vector<std::span<const float>>& recordings,
                                                const SegmenterConfig& segmenter);
};

// Hallucination filter. Models invent low-confidence words from noise and
// room tone; real but isolated words are also low-confidence yet sound like
// speech, and sung vocals are confident yet score low on the VAD. So a segment
// is dropped only when it is BOTH unsure (mean word confidence < min_confidence)
// AND unlike speech (mean VAD probability < speech_floor).
bool keep_segment(const SegmentResult& segment, float min_confidence, float speech_floor);

// 16 kHz mono float from any media file (ffmpeg pipe; WAV read natively).
std::vector<float> load_audio(const std::filesystem::path& path, const std::filesystem::path& ffmpeg = {});

// ---- Caption formatting --------------------------------------------------
struct Cue { double start = 0, end = 0; std::string text; };
struct CueOptions {
    std::size_t line_chars = 42;  // two lines per cue at most
    double max_seconds = 6.0;
    double gap_seconds = 0.8;     // a pause longer than this starts a new cue
    double linger_seconds = 0.4;  // hold after the last word, never into the next cue
    bool sentence_case = true;    // recase all-uppercase output; cased models are kept verbatim
};
std::vector<Cue> build_cues(const std::vector<SegmentResult>& segments, const CueOptions& options = {});

enum class CaptionFormat { srt, vtt, txt, json };
CaptionFormat parse_format(const std::string& name);  // throws on unknown names
const char* extension(CaptionFormat format);           // ".srt", ...
std::string render(CaptionFormat format, const std::vector<SegmentResult>& segments,
                   const std::vector<Cue>& cues);
} // namespace captions
