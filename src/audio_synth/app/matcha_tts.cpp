#include "cli_options.hpp"
#include "vocal/phonemizer.hpp"
#include <matcha_onnx/matcha_acoustic_gen.hpp>
#include <explicit_neural/include/neural_vocoder.hpp>

#include <iostream>

int main(int argc, char** argv) {
    try {
        const vocal::cli::Options options(argc, argv, {"--assets", "--text", "--output", "--speed", "--noise-scale", "--threads"});
        if (options.help || argc == 1) {
            std::cout << "Usage: matcha-tts --text TEXT [--assets model_assets/matcha] [--output artifacts/matcha/speech.wav]\n"
                      << "  --speed 1 --noise-scale 1 --threads 4\n"
                      << "Native English Matcha acoustic graph + HiFi-GAN v2; resampled from 22.05 to 24 kHz PCM16.\n"
                      << "This export predicts durations; it does not expose pitch or energy contours.\n";
            return options.help ? 0 : 2;
        }
        const std::filesystem::path assets = options.get("--assets", "model_assets/matcha");
        const std::filesystem::path output = options.get("--output", "artifacts/matcha/speech.wav");
        const auto text = options.required("--text");
        const vocal::CmuPhonemizer phonemizer(assets / "cmudict.dict", assets / "tokens.tsv");
        const auto phonemes = phonemizer.phonemize(text);
        if (phonemes.words == 0) throw std::invalid_argument("text must contain words");
        if (phonemes.missing_model_symbols != 0) throw std::runtime_error("Matcha vocabulary is missing frontend symbols");
        const int threads = options.number<int>("--threads", 4);
        const float speed = options.number<float>("--speed", 1.0F), noise = options.number<float>("--noise-scale", 1.0F);
        vocal::MatchaAcousticGenerator acoustic(assets / "acoustic_generator.onnx", threads);
        vocal::NeuralVocoderConfig config;
        config.sample_rate_hz = 22'050;
        config.intra_op_threads = threads;
        vocal::NeuralVocoder vocoder(assets / "vocoder_hifigan.onnx", config);
        const auto mel = acoustic.infer(phonemes.token_ids, speed, noise);
        const auto native = vocoder.synthesize(mel);
        const auto audio = vocal::resample_waveform(native, 24'000);
        vocal::write_wav_pcm16(output, audio);
        auto report = vocal::cli::report_file(output);
        report << "{\n  \"engine\": \"matcha-icefall-hifigan-v2\",\n  \"native_sample_rate\": 22050,\n  \"sample_rate\": 24000"
               << ",\n  \"frames\": " << mel.frames << ",\n  \"native_samples\": " << native.samples.size()
               << ",\n  \"samples\": " << audio.samples.size() << ",\n  \"acoustic_ms\": " << acoustic.inference_ms()
               << ",\n  \"vocoder_ms\": " << vocoder.inference_ms() << ",\n  \"speed\": " << speed << ",\n  \"noise_scale\": " << noise
               << ",\n  \"dictionary_hits\": " << phonemes.dictionary_hits << ",\n  \"fallback_words\": " << phonemes.fallback_words
               << ",\n  \"missing_model_symbols\": " << phonemes.missing_model_symbols << "\n}\n";
        if (!report) throw std::runtime_error("failed to write diagnostics");
        std::cout << output.string() << "\naudio_seconds=" << static_cast<double>(audio.samples.size()) / audio.sample_rate_hz
                  << "\nacoustic_ms=" << acoustic.inference_ms() << "\nvocoder_ms=" << vocoder.inference_ms() << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
