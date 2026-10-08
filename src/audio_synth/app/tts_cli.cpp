#include "cli_options.hpp"
#include "asset_paths.hpp"
#include "vocal/phonemizer.hpp"
#include "vocal/control_params.hpp"
#include <explicit_neural/include/explicit_acoustic_gen.hpp>
#include <explicit_neural/include/neural_vocoder.hpp>
#include <explicit_neural/include/prosody_predictor.hpp>

#include <iostream>

namespace {
vocal::VocalEmotion parse_emotion(std::string_view name) {
    if (name == "neutral") return vocal::VocalEmotion::Neutral;
    if (name == "whisper") return vocal::VocalEmotion::Whisper;
    if (name == "excited") return vocal::VocalEmotion::Excited;
    if (name == "somber" || name == "calm") return vocal::VocalEmotion::Somber;
    if (name == "authoritative") return vocal::VocalEmotion::Authoritative;
    throw std::invalid_argument("unknown emotion; choose neutral, whisper, excited, somber, calm or authoritative");
}
}

int main(int argc, char** argv) {
    try {
        const vocal::cli::Options options(argc, argv, {
            "--assets", "--text", "--output", "--durations", "--f0", "--energy",
            "--frames-per-token", "--f0-hz", "--pitch-scale", "--speed", "--energy-scale",
            "--energy-variance", "--speaker-id", "--threads", "--thread-affinities", "--hop-length", "--emotion"});
        if (options.help || argc == 1) {
            std::cout << "Usage: explicit-tts --text TEXT [--assets model_assets/explicit_neural] [--output artifacts/explicit_neural/speech.wav]\n"
                      << "  --durations FILE --f0 FILE --energy FILE (CSV/whitespace; F0 Hz, normalized energy)\n"
                      << "  Supply all three curve files to bypass the prosody predictor; partial overrides use predicted defaults.\n"
                      << "  --frames-per-token N --f0-hz HZ (override trained predictions; fixture defaults 6/180)\n"
                      << "  --pitch-scale 1 --speed 1 --energy-scale 1 --energy-variance 1 --speaker-id 0 --threads 4\n"
                      << "  --hop-length N (defaults to pipeline.properties; fixture default 256)\n"
                      << "  --thread-affinities ORT_AFFINITY_STRING (optional worker pinning)\n"
                      << "  --emotion neutral|whisper|excited|somber|calm|authoritative (before scalar sliders)\n"
                      << "Requires compatible trained acoustic_generator.onnx, vocoder_hifigan.onnx, tokens.tsv and cmudict.dict.\n";
            return options.help ? 0 : 2;
        }
        const auto text = options.required("--text");
        const auto emotion_name = options.get("--emotion", "neutral");
        const auto emotion = parse_emotion(emotion_name);
        const std::filesystem::path assets = options.get("--assets", "model_assets/explicit_neural");
        const std::filesystem::path output = options.get("--output", "artifacts/explicit_neural/speech.wav");
        std::map<std::string, std::string> properties;
        std::ifstream configuration(assets / "pipeline.properties");
        for (std::string line; std::getline(configuration, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const auto equals = line.find('=');
            if (equals != std::string::npos) properties[line.substr(0, equals)] = line.substr(equals + 1);
        }
        const int threads = options.number<int>("--threads", 4);
        const auto affinities = options.get("--thread-affinities");
        const auto speaker = options.number<std::int64_t>("--speaker-id", 0);
        if (speaker < 0 || (properties.contains("num_speakers") &&
            speaker >= vocal::cli::Options::parse<std::int64_t>(properties.at("num_speakers"))))
            throw std::invalid_argument("speaker ID is outside the trained model's range");
        const auto dictionary = vocal::cli::resolve_dictionary(assets);
        const vocal::CmuPhonemizer phonemizer(dictionary, assets / "tokens.tsv");
        const auto phonemes = phonemizer.phonemize(text);
        if (phonemes.words == 0) throw std::invalid_argument("text must contain words");
        if (phonemes.missing_model_symbols != 0)
            throw std::runtime_error("token map is missing phoneme/special symbols; use the acoustic export's complete vocabulary");
        vocal::ProsodyControls controls;
        double prosody_ms = 0;
        std::string prosody_source = "illustrative-defaults";
        const auto predictor_path = assets / "prosody_predictor.onnx";
        if (options.has("--durations") && options.has("--f0") && options.has("--energy")) {
            controls.durations = vocal::cli::read_values<std::int64_t>(options.get("--durations"), 4096);
            if (controls.durations.size() != phonemes.token_ids.size())
                throw std::invalid_argument("durations must match the model token count");
            prosody_source = "manual-curves";
        } else if (std::filesystem::is_regular_file(predictor_path) || properties["prosody_predictor"] == "required") {
            std::vector<std::int64_t> durations;
            if (options.has("--durations")) {
                durations = vocal::cli::read_values<std::int64_t>(options.get("--durations"), 4096);
                if (durations.size() != phonemes.token_ids.size())
                    throw std::invalid_argument("durations must match the model token count");
            } else if (options.has("--frames-per-token")) {
                const auto count = options.number<std::int64_t>("--frames-per-token", 6);
                if (count <= 0) throw std::invalid_argument("frames per token must be positive");
                durations.assign(phonemes.token_ids.size(), count);
            }
            vocal::NeuralProsodyPredictor predictor(predictor_path, threads, affinities);
            controls = predictor.predict(phonemes.token_ids, speaker, durations);
            prosody_ms = predictor.inference_ms();
            prosody_source = "trained-vctk-predictor";
            if (options.has("--f0-hz"))
                controls.f0_contour.assign(controls.f0_contour.size(), options.number<float>("--f0-hz", 180.0F));
        } else if (options.has("--durations")) {
            controls.durations = vocal::cli::read_values<std::int64_t>(options.get("--durations"), 4096);
            const auto frames = vocal::duration_frames(controls.durations);
            controls.f0_contour.assign(frames, options.number<float>("--f0-hz", 180.0F));
            controls.energy_contour.assign(frames, 1.0F);
        } else controls = vocal::baseline_prosody(phonemes.token_ids.size(),
            options.number<std::int64_t>("--frames-per-token", 6), options.number<float>("--f0-hz", 180.0F));
        if (options.has("--f0")) controls.f0_contour = vocal::cli::read_values<float>(options.get("--f0"), vocal::maximum_prosody_frames);
        if (options.has("--energy")) controls.energy_contour = vocal::cli::read_values<float>(options.get("--energy"), vocal::maximum_prosody_frames);
        controls.speaker_id = speaker;
        controls.token_kinds = phonemes.token_kinds;
        controls = vocal::apply_emotion_preset(controls, emotion);
        const vocal::ProsodySliders sliders{options.number<float>("--pitch-scale", 1.0F),
            options.number<float>("--speed", 1.0F), options.number<float>("--energy-scale", 1.0F),
            options.number<float>("--energy-variance", 1.0F)};
        controls = vocal::apply_prosody_sliders(controls, sliders);
        vocal::ExplicitAcousticGenerator acoustic(assets / "acoustic_generator.onnx", {threads, affinities, "mel"});
        vocal::NeuralVocoderConfig vocoder_config;
        vocoder_config.intra_op_threads = threads;
        vocoder_config.thread_affinities = affinities;
        vocoder_config.sample_rate_hz = properties.contains("sample_rate") ?
            vocal::cli::Options::parse<int>(properties.at("sample_rate")) : 24'000;
        vocoder_config.hop_length = options.number<std::size_t>("--hop-length", properties.contains("hop_length") ?
            vocal::cli::Options::parse<std::size_t>(properties.at("hop_length")) : 256);
        vocal::NeuralVocoder vocoder(assets / "vocoder_hifigan.onnx", vocoder_config);
        const auto mel = acoustic.infer(phonemes.token_ids, controls);
        const auto audio = vocoder.synthesize(mel);
        vocal::write_wav_pcm16(output, audio);
        auto report = vocal::cli::report_file(output);
        report << "{\n  \"engine\": \"explicit-acoustic-neural-vocoder\",\n  \"sample_rate\": " << audio.sample_rate_hz
               << ",\n  \"hop_length\": " << vocoder_config.hop_length << ",\n  \"tokens\": " << phonemes.token_ids.size()
               << ",\n  \"frames\": " << mel.frames << ",\n  \"samples\": " << audio.samples.size()
               << ",\n  \"prosody_ms\": " << prosody_ms << ",\n  \"prosody_source\": \"" << prosody_source
               << "\",\n  \"acoustic_ms\": " << acoustic.inference_ms() << ",\n  \"vocoder_ms\": " << vocoder.inference_ms()
               << ",\n  \"dictionary_hits\": " << phonemes.dictionary_hits << ",\n  \"fallback_words\": " << phonemes.fallback_words
               << ",\n  \"speaker_id\": " << controls.speaker_id << ",\n  \"emotion\": \"" << emotion_name
               << "\",\n  \"input_ids\": ";
        vocal::cli::array(report, phonemes.token_ids);
        report << ",\n  \"durations\": "; vocal::cli::array(report, controls.durations);
        report << ",\n  \"f0_hz\": "; vocal::cli::array(report, controls.f0_contour);
        report << ",\n  \"energy\": "; vocal::cli::array(report, controls.energy_contour);
        report << "\n}\n";
        if (!report) throw std::runtime_error("failed to write diagnostics");
        std::cout << output.string() << "\ndictionary=" << dictionary.string()
                  << "\nprosody=" << prosody_source << "\nframes=" << mel.frames << "\nsamples=" << audio.samples.size() << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
