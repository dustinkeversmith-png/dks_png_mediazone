#include "cli_options.hpp"
#include <piper_onnx/piper_voice.hpp>

#include <iostream>

int main(int argc, char** argv) {
    try {
        const vocal::cli::Options options(argc, argv, {"--assets", "--voice", "--text", "--output",
            "--speaker-id", "--length-scale", "--noise-scale", "--noise-w", "--threads"});
        if (options.help || argc == 1) {
            std::cout << "Usage: piper-tts --text TEXT [--voice en_US-lessac-medium] [--assets model_assets/piper]\n"
                      << "  --output artifacts/piper/speech.wav --speaker-id 0 --length-scale 1 --noise-scale 0.667 --noise-w 0.8 --threads 4\n";
            return options.help ? 0 : 2;
        }
        const std::filesystem::path assets = options.get("--assets", "model_assets/piper");
        const auto voice = options.get("--voice", "en_US-lessac-medium");
        const auto text = options.required("--text");
        const std::filesystem::path output = options.get("--output", "artifacts/piper/" + voice + ".wav");
        vocal::PiperVoiceConfig config{assets / (voice + ".onnx"), assets / (voice + ".properties"),
            assets / "cmudict.dict", assets / (voice + ".tokens.tsv")};
        std::ifstream properties(config.properties_path);
        for (std::string line; std::getline(properties, line);) {
            const auto equals = line.find('=');
            if (equals == std::string::npos) continue;
            const auto key = line.substr(0, equals);
            auto value = line.substr(equals + 1);
            if (!value.empty() && value.back() == '\r') value.pop_back();
            if (key == "length_scale") config.length_scale = vocal::cli::Options::parse<float>(value);
            else if (key == "noise_scale") config.noise_scale = vocal::cli::Options::parse<float>(value);
            else if (key == "noise_w") config.noise_w = vocal::cli::Options::parse<float>(value);
        }
        config.speaker_id = options.number<std::int64_t>("--speaker-id", 0);
        config.length_scale = options.number<float>("--length-scale", config.length_scale);
        config.noise_scale = options.number<float>("--noise-scale", config.noise_scale);
        config.noise_w = options.number<float>("--noise-w", config.noise_w);
        config.intra_op_threads = options.number<int>("--threads", 4);
        if (!std::isfinite(config.length_scale) || config.length_scale <= 0 ||
            !std::isfinite(config.noise_scale) || config.noise_scale < 0 ||
            !std::isfinite(config.noise_w) || config.noise_w < 0 || config.intra_op_threads <= 0)
            throw std::invalid_argument("Piper length/threads must be positive; noise finite and nonnegative");
        vocal::PiperVoiceSynthesizer model(config);
        const auto audio = model.synthesize(text);
        vocal::write_wav_pcm16(output, audio);
        const auto& d = model.diagnostics();
        auto report = vocal::cli::report_file(output);
        report << "{\n  \"engine\": \"piper-vits-onnx\",\n  \"sample_rate\": " << audio.sample_rate_hz
               << ",\n  \"audio_seconds\": " << d.audio_seconds << ",\n  \"inference_ms\": " << d.inference_ms
               << ",\n  \"real_time_factor\": " << d.real_time_factor << ",\n  \"chunks\": " << d.chunks
               << ",\n  \"phoneme_tokens\": " << d.phoneme_tokens << ",\n  \"dictionary_hits\": " << d.dictionary_hits
               << ",\n  \"fallback_words\": " << d.fallback_words << ",\n  \"missing_model_symbols\": " << d.missing_model_symbols
               << ",\n  \"speaker_id\": " << d.speaker_id << ",\n  \"length_scale\": " << config.length_scale
               << ",\n  \"noise_scale\": " << config.noise_scale << ",\n  \"noise_w\": " << config.noise_w << "\n}\n";
        if (!report) throw std::runtime_error("failed to write diagnostics");
        std::cout << output.string() << "\naudio_seconds=" << d.audio_seconds << "\ninference_ms=" << d.inference_ms << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
