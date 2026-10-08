#include "cli_options.hpp"
#include "vocal/phonemizer.hpp"
#include "vocal/control_params.hpp"
#include <explicit_neural/include/explicit_acoustic_gen.hpp>
#include <explicit_neural/include/neural_vocoder.hpp>

#include <iostream>

int main(int argc, char** argv) {
    try {
        const vocal::cli::Options options(argc, argv, {
            "--assets", "--text", "--output", "--durations", "--f0", "--energy",
            "--frames-per-token", "--f0-hz", "--pitch-scale", "--speed", "--energy-scale",
            "--energy-variance", "--speaker-id", "--threads", "--thread-affinities", "--hop-length"});
        if (options.help || argc == 1) {
            std::cout << "Usage: explicit-tts --text TEXT [--assets model_assets/explicit_neural] [--output artifacts/explicit_neural/speech.wav]\n"
                      << "  --durations FILE --f0 FILE --energy FILE (CSV/whitespace; F0 Hz, normalized energy)\n"
                      << "  --frames-per-token 6 --f0-hz 180 --pitch-scale 1 --speed 1\n"
                      << "  --energy-scale 1 --energy-variance 1 --speaker-id 0 --threads 4 --hop-length 256\n"
                      << "  --thread-affinities ORT_AFFINITY_STRING (optional worker pinning)\n"
                      << "Requires compatible trained acoustic_generator.onnx, vocoder_hifigan.onnx, tokens.tsv and cmudict.dict.\n";
            return options.help ? 0 : 2;
        }
        const auto text = options.required("--text");
        const std::filesystem::path assets = options.get("--assets", "model_assets/explicit_neural");
        const std::filesystem::path output = options.get("--output", "artifacts/explicit_neural/speech.wav");
        const vocal::CmuPhonemizer phonemizer(assets / "cmudict.dict", assets / "tokens.tsv");
        const auto phonemes = phonemizer.phonemize(text);
        if (phonemes.words == 0) throw std::invalid_argument("text must contain words");
        if (phonemes.missing_model_symbols != 0)
            throw std::runtime_error("token map is missing phoneme/special symbols; use the acoustic export's complete vocabulary");
        vocal::ProsodyControls controls;
        if (options.has("--durations")) {
            controls.durations = vocal::cli::read_values<std::int64_t>(options.get("--durations"), 4096);
            const auto frames = vocal::duration_frames(controls.durations);
            controls.f0_contour.assign(frames, options.number<float>("--f0-hz", 180.0F));
            controls.energy_contour.assign(frames, 1.0F);
        } else controls = vocal::baseline_prosody(phonemes.token_ids.size(),
            options.number<std::int64_t>("--frames-per-token", 6), options.number<float>("--f0-hz", 180.0F));
        if (options.has("--f0")) controls.f0_contour = vocal::cli::read_values<float>(options.get("--f0"), vocal::maximum_prosody_frames);
        if (options.has("--energy")) controls.energy_contour = vocal::cli::read_values<float>(options.get("--energy"), vocal::maximum_prosody_frames);
        controls.speaker_id = options.number<std::int64_t>("--speaker-id", 0);
        const vocal::ProsodySliders sliders{options.number<float>("--pitch-scale", 1.0F),
            options.number<float>("--speed", 1.0F), options.number<float>("--energy-scale", 1.0F),
            options.number<float>("--energy-variance", 1.0F)};
        controls = vocal::apply_prosody_sliders(controls, sliders);
        const int threads = options.number<int>("--threads", 4);
        const auto affinities = options.get("--thread-affinities");
        vocal::ExplicitAcousticGenerator acoustic(assets / "acoustic_generator.onnx", {threads, affinities, "mel"});
        vocal::NeuralVocoderConfig vocoder_config;
        vocoder_config.intra_op_threads = threads;
        vocoder_config.thread_affinities = affinities;
        vocoder_config.hop_length = options.number<std::size_t>("--hop-length", 256);
        vocal::NeuralVocoder vocoder(assets / "vocoder_hifigan.onnx", vocoder_config);
        const auto mel = acoustic.infer(phonemes.token_ids, controls);
        const auto audio = vocoder.synthesize(mel);
        vocal::write_wav_pcm16(output, audio);
        auto report = vocal::cli::report_file(output);
        report << "{\n  \"engine\": \"explicit-acoustic-neural-vocoder\",\n  \"sample_rate\": " << audio.sample_rate_hz
               << ",\n  \"hop_length\": " << vocoder_config.hop_length << ",\n  \"tokens\": " << phonemes.token_ids.size()
               << ",\n  \"frames\": " << mel.frames << ",\n  \"samples\": " << audio.samples.size()
               << ",\n  \"acoustic_ms\": " << acoustic.inference_ms() << ",\n  \"vocoder_ms\": " << vocoder.inference_ms()
               << ",\n  \"dictionary_hits\": " << phonemes.dictionary_hits << ",\n  \"fallback_words\": " << phonemes.fallback_words
               << ",\n  \"speaker_id\": " << controls.speaker_id << ",\n  \"input_ids\": ";
        vocal::cli::array(report, phonemes.token_ids);
        report << ",\n  \"durations\": "; vocal::cli::array(report, controls.durations);
        report << ",\n  \"f0_hz\": "; vocal::cli::array(report, controls.f0_contour);
        report << ",\n  \"energy\": "; vocal::cli::array(report, controls.energy_contour);
        report << "\n}\n";
        if (!report) throw std::runtime_error("failed to write diagnostics");
        std::cout << output.string() << "\nframes=" << mel.frames << "\nsamples=" << audio.samples.size() << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
