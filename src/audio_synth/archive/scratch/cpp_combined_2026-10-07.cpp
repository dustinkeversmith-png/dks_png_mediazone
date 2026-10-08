// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\app\piper_tts.cpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\app\tts_cli.cpp ===
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
        if (std::filesystem::is_regular_file(predictor_path) || properties["prosody_predictor"] == "required") {
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

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\build-ort\CMakeFiles\4.3.3\CompilerIdCXX\CMakeCXXCompilerId.cpp ===
/* This source file must have a .cpp extension so that all C++ compilers
   recognize the extension without flags.  Borland does not know .cxx for
   example.  */
#ifndef __cplusplus
# error "A C compiler has been selected for C++."
#endif

#if !defined(__has_include)
/* If the compiler does not have __has_include, pretend the answer is
   always no.  */
#  define __has_include(x) 0
#endif


/* Version number components: V=Version, R=Revision, P=Patch
   Version date components:   YYYY=Year, MM=Month,   DD=Day  */

#if defined(__INTEL_COMPILER) || defined(__ICC)
# define COMPILER_ID "Intel"
# if defined(_MSC_VER)
#  define SIMULATE_ID "MSVC"
# endif
# if defined(__GNUC__)
#  define SIMULATE_ID "GNU"
# endif
  /* __INTEL_COMPILER = VRP prior to 2021, and then VVVV for 2021 and later,
     except that a few beta releases use the old format with V=2021.  */
# if __INTEL_COMPILER < 2021 || __INTEL_COMPILER == 202110 || __INTEL_COMPILER == 202111
#  define COMPILER_VERSION_MAJOR DEC(__INTEL_COMPILER/100)
#  define COMPILER_VERSION_MINOR DEC(__INTEL_COMPILER/10 % 10)
#  if defined(__INTEL_COMPILER_UPDATE)
#   define COMPILER_VERSION_PATCH DEC(__INTEL_COMPILER_UPDATE)
#  else
#   define COMPILER_VERSION_PATCH DEC(__INTEL_COMPILER   % 10)
#  endif
# else
#  define COMPILER_VERSION_MAJOR DEC(__INTEL_COMPILER)
#  define COMPILER_VERSION_MINOR DEC(__INTEL_COMPILER_UPDATE)
   /* The third version component from --version is an update index,
      but no macro is provided for it.  */
#  define COMPILER_VERSION_PATCH DEC(0)
# endif
# if defined(__INTEL_COMPILER_BUILD_DATE)
   /* __INTEL_COMPILER_BUILD_DATE = YYYYMMDD */
#  define COMPILER_VERSION_TWEAK DEC(__INTEL_COMPILER_BUILD_DATE)
# endif
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define SIMULATE_VERSION_MAJOR DEC(_MSC_VER / 100)
#  define SIMULATE_VERSION_MINOR DEC(_MSC_VER % 100)
# endif
# if defined(__GNUC__)
#  define SIMULATE_VERSION_MAJOR DEC(__GNUC__)
# elif defined(__GNUG__)
#  define SIMULATE_VERSION_MAJOR DEC(__GNUG__)
# endif
# if defined(__GNUC_MINOR__)
#  define SIMULATE_VERSION_MINOR DEC(__GNUC_MINOR__)
# endif
# if defined(__GNUC_PATCHLEVEL__)
#  define SIMULATE_VERSION_PATCH DEC(__GNUC_PATCHLEVEL__)
# endif

#elif (defined(__clang__) && defined(__INTEL_CLANG_COMPILER)) || defined(__INTEL_LLVM_COMPILER)
# define COMPILER_ID "IntelLLVM"
#if defined(_MSC_VER)
# define SIMULATE_ID "MSVC"
#endif
#if defined(__GNUC__)
# define SIMULATE_ID "GNU"
#endif
/* __INTEL_LLVM_COMPILER = VVVVRP prior to 2021.2.0, VVVVRRPP for 2021.2.0 and
 * later.  Look for 6 digit vs. 8 digit version number to decide encoding.
 * VVVV is no smaller than the current year when a version is released.
 */
#if __INTEL_LLVM_COMPILER < 1000000L
# define COMPILER_VERSION_MAJOR DEC(__INTEL_LLVM_COMPILER/100)
# define COMPILER_VERSION_MINOR DEC(__INTEL_LLVM_COMPILER/10 % 10)
# define COMPILER_VERSION_PATCH DEC(__INTEL_LLVM_COMPILER    % 10)
#else
# define COMPILER_VERSION_MAJOR DEC(__INTEL_LLVM_COMPILER/10000)
# define COMPILER_VERSION_MINOR DEC(__INTEL_LLVM_COMPILER/100 % 100)
# define COMPILER_VERSION_PATCH DEC(__INTEL_LLVM_COMPILER     % 100)
#endif
#if defined(_MSC_VER)
  /* _MSC_VER = VVRR */
# define SIMULATE_VERSION_MAJOR DEC(_MSC_VER / 100)
# define SIMULATE_VERSION_MINOR DEC(_MSC_VER % 100)
#endif
#if defined(__GNUC__)
# define SIMULATE_VERSION_MAJOR DEC(__GNUC__)
#elif defined(__GNUG__)
# define SIMULATE_VERSION_MAJOR DEC(__GNUG__)
#endif
#if defined(__GNUC_MINOR__)
# define SIMULATE_VERSION_MINOR DEC(__GNUC_MINOR__)
#endif
#if defined(__GNUC_PATCHLEVEL__)
# define SIMULATE_VERSION_PATCH DEC(__GNUC_PATCHLEVEL__)
#endif

#elif defined(__PATHCC__)
# define COMPILER_ID "PathScale"
# define COMPILER_VERSION_MAJOR DEC(__PATHCC__)
# define COMPILER_VERSION_MINOR DEC(__PATHCC_MINOR__)
# if defined(__PATHCC_PATCHLEVEL__)
#  define COMPILER_VERSION_PATCH DEC(__PATHCC_PATCHLEVEL__)
# endif

#elif defined(__BORLANDC__) && defined(__CODEGEARC_VERSION__)
# define COMPILER_ID "Embarcadero"
# define COMPILER_VERSION_MAJOR HEX(__CODEGEARC_VERSION__>>24 & 0x00FF)
# define COMPILER_VERSION_MINOR HEX(__CODEGEARC_VERSION__>>16 & 0x00FF)
# define COMPILER_VERSION_PATCH DEC(__CODEGEARC_VERSION__     & 0xFFFF)

#elif defined(__BORLANDC__)
# define COMPILER_ID "Borland"
  /* __BORLANDC__ = 0xVRR */
# define COMPILER_VERSION_MAJOR HEX(__BORLANDC__>>8)
# define COMPILER_VERSION_MINOR HEX(__BORLANDC__ & 0xFF)

#elif defined(__WATCOMC__) && __WATCOMC__ < 1200
# define COMPILER_ID "Watcom"
   /* __WATCOMC__ = VVRR */
# define COMPILER_VERSION_MAJOR DEC(__WATCOMC__ / 100)
# define COMPILER_VERSION_MINOR DEC((__WATCOMC__ / 10) % 10)
# if (__WATCOMC__ % 10) > 0
#  define COMPILER_VERSION_PATCH DEC(__WATCOMC__ % 10)
# endif

#elif defined(__WATCOMC__)
# define COMPILER_ID "OpenWatcom"
   /* __WATCOMC__ = VVRP + 1100 */
# define COMPILER_VERSION_MAJOR DEC((__WATCOMC__ - 1100) / 100)
# define COMPILER_VERSION_MINOR DEC((__WATCOMC__ / 10) % 10)
# if (__WATCOMC__ % 10) > 0
#  define COMPILER_VERSION_PATCH DEC(__WATCOMC__ % 10)
# endif

#elif defined(__SUNPRO_CC)
# define COMPILER_ID "SunPro"
# if __SUNPRO_CC >= 0x5100
   /* __SUNPRO_CC = 0xVRRP */
#  define COMPILER_VERSION_MAJOR HEX(__SUNPRO_CC>>12)
#  define COMPILER_VERSION_MINOR HEX(__SUNPRO_CC>>4 & 0xFF)
#  define COMPILER_VERSION_PATCH HEX(__SUNPRO_CC    & 0xF)
# else
   /* __SUNPRO_CC = 0xVRP */
#  define COMPILER_VERSION_MAJOR HEX(__SUNPRO_CC>>8)
#  define COMPILER_VERSION_MINOR HEX(__SUNPRO_CC>>4 & 0xF)
#  define COMPILER_VERSION_PATCH HEX(__SUNPRO_CC    & 0xF)
# endif

#elif defined(__HP_aCC)
# define COMPILER_ID "HP"
  /* __HP_aCC = VVRRPP */
# define COMPILER_VERSION_MAJOR DEC(__HP_aCC/10000)
# define COMPILER_VERSION_MINOR DEC(__HP_aCC/100 % 100)
# define COMPILER_VERSION_PATCH DEC(__HP_aCC     % 100)

#elif defined(__DECCXX)
# define COMPILER_ID "Compaq"
  /* __DECCXX_VER = VVRRTPPPP */
# define COMPILER_VERSION_MAJOR DEC(__DECCXX_VER/10000000)
# define COMPILER_VERSION_MINOR DEC(__DECCXX_VER/100000  % 100)
# define COMPILER_VERSION_PATCH DEC(__DECCXX_VER         % 10000)

#elif defined(__IBMCPP__) && defined(__COMPILER_VER__)
# define COMPILER_ID "zOS"
  /* __IBMCPP__ = VRP */
# define COMPILER_VERSION_MAJOR DEC(__IBMCPP__/100)
# define COMPILER_VERSION_MINOR DEC(__IBMCPP__/10 % 10)
# define COMPILER_VERSION_PATCH DEC(__IBMCPP__    % 10)

#elif defined(__open_xl__) && defined(__clang__)
# define COMPILER_ID "IBMClang"
# define COMPILER_VERSION_MAJOR DEC(__open_xl_version__)
# define COMPILER_VERSION_MINOR DEC(__open_xl_release__)
# define COMPILER_VERSION_PATCH DEC(__open_xl_modification__)
# define COMPILER_VERSION_TWEAK DEC(__open_xl_ptf_fix_level__)
# define COMPILER_VERSION_INTERNAL_STR  __clang_version__


#elif defined(__ibmxl__) && defined(__clang__)
# define COMPILER_ID "XLClang"
# define COMPILER_VERSION_MAJOR DEC(__ibmxl_version__)
# define COMPILER_VERSION_MINOR DEC(__ibmxl_release__)
# define COMPILER_VERSION_PATCH DEC(__ibmxl_modification__)
# define COMPILER_VERSION_TWEAK DEC(__ibmxl_ptf_fix_level__)


#elif defined(__IBMCPP__) && !defined(__COMPILER_VER__) && __IBMCPP__ >= 800
# define COMPILER_ID "XL"
  /* __IBMCPP__ = VRP */
# define COMPILER_VERSION_MAJOR DEC(__IBMCPP__/100)
# define COMPILER_VERSION_MINOR DEC(__IBMCPP__/10 % 10)
# define COMPILER_VERSION_PATCH DEC(__IBMCPP__    % 10)

#elif defined(__IBMCPP__) && !defined(__COMPILER_VER__) && __IBMCPP__ < 800
# define COMPILER_ID "VisualAge"
  /* __IBMCPP__ = VRP */
# define COMPILER_VERSION_MAJOR DEC(__IBMCPP__/100)
# define COMPILER_VERSION_MINOR DEC(__IBMCPP__/10 % 10)
# define COMPILER_VERSION_PATCH DEC(__IBMCPP__    % 10)

#elif defined(__NVCOMPILER)
# define COMPILER_ID "NVHPC"
# define COMPILER_VERSION_MAJOR DEC(__NVCOMPILER_MAJOR__)
# define COMPILER_VERSION_MINOR DEC(__NVCOMPILER_MINOR__)
# if defined(__NVCOMPILER_PATCHLEVEL__)
#  define COMPILER_VERSION_PATCH DEC(__NVCOMPILER_PATCHLEVEL__)
# endif

#elif defined(__PGI)
# define COMPILER_ID "PGI"
# define COMPILER_VERSION_MAJOR DEC(__PGIC__)
# define COMPILER_VERSION_MINOR DEC(__PGIC_MINOR__)
# if defined(__PGIC_PATCHLEVEL__)
#  define COMPILER_VERSION_PATCH DEC(__PGIC_PATCHLEVEL__)
# endif

#elif defined(__clang__) && defined(__cray__)
# define COMPILER_ID "CrayClang"
# define COMPILER_VERSION_MAJOR DEC(__cray_major__)
# define COMPILER_VERSION_MINOR DEC(__cray_minor__)
# define COMPILER_VERSION_PATCH DEC(__cray_patchlevel__)
# define COMPILER_VERSION_INTERNAL_STR __clang_version__


#elif defined(_CRAYC)
# define COMPILER_ID "Cray"
# define COMPILER_VERSION_MAJOR DEC(_RELEASE_MAJOR)
# define COMPILER_VERSION_MINOR DEC(_RELEASE_MINOR)

#elif defined(__TI_COMPILER_VERSION__)
# define COMPILER_ID "TI"
  /* __TI_COMPILER_VERSION__ = VVVRRRPPP */
# define COMPILER_VERSION_MAJOR DEC(__TI_COMPILER_VERSION__/1000000)
# define COMPILER_VERSION_MINOR DEC(__TI_COMPILER_VERSION__/1000   % 1000)
# define COMPILER_VERSION_PATCH DEC(__TI_COMPILER_VERSION__        % 1000)

#elif defined(__CLANG_FUJITSU)
# define COMPILER_ID "FujitsuClang"
# define COMPILER_VERSION_MAJOR DEC(__FCC_major__)
# define COMPILER_VERSION_MINOR DEC(__FCC_minor__)
# define COMPILER_VERSION_PATCH DEC(__FCC_patchlevel__)
# define COMPILER_VERSION_INTERNAL_STR __clang_version__


#elif defined(__FUJITSU)
# define COMPILER_ID "Fujitsu"
# if defined(__FCC_version__)
#   define COMPILER_VERSION __FCC_version__
# elif defined(__FCC_major__)
#   define COMPILER_VERSION_MAJOR DEC(__FCC_major__)
#   define COMPILER_VERSION_MINOR DEC(__FCC_minor__)
#   define COMPILER_VERSION_PATCH DEC(__FCC_patchlevel__)
# endif
# if defined(__fcc_version)
#   define COMPILER_VERSION_INTERNAL DEC(__fcc_version)
# elif defined(__FCC_VERSION)
#   define COMPILER_VERSION_INTERNAL DEC(__FCC_VERSION)
# endif


#elif defined(__ghs__)
# define COMPILER_ID "GHS"
/* __GHS_VERSION_NUMBER = VVVVRP */
# ifdef __GHS_VERSION_NUMBER
# define COMPILER_VERSION_MAJOR DEC(__GHS_VERSION_NUMBER / 100)
# define COMPILER_VERSION_MINOR DEC(__GHS_VERSION_NUMBER / 10 % 10)
# define COMPILER_VERSION_PATCH DEC(__GHS_VERSION_NUMBER      % 10)
# endif

#elif defined(__TASKING__)
# define COMPILER_ID "Tasking"
  # define COMPILER_VERSION_MAJOR DEC(__VERSION__/1000)
  # define COMPILER_VERSION_MINOR DEC(__VERSION__ % 100)
# define COMPILER_VERSION_INTERNAL DEC(__VERSION__)

#elif defined(__ORANGEC__)
# define COMPILER_ID "OrangeC"
# define COMPILER_VERSION_MAJOR DEC(__ORANGEC_MAJOR__)
# define COMPILER_VERSION_MINOR DEC(__ORANGEC_MINOR__)
# define COMPILER_VERSION_PATCH DEC(__ORANGEC_PATCHLEVEL__)

#elif defined(__RENESAS__)
# define COMPILER_ID "Renesas"
/* __RENESAS_VERSION__ = 0xVVRRPP00 */
# define COMPILER_VERSION_MAJOR HEX(__RENESAS_VERSION__ >> 24 & 0xFF)
# define COMPILER_VERSION_MINOR HEX(__RENESAS_VERSION__ >> 16 & 0xFF)
# define COMPILER_VERSION_PATCH HEX(__RENESAS_VERSION__ >> 8  & 0xFF)

#elif defined(__SCO_VERSION__)
# define COMPILER_ID "SCO"

#elif defined(__ARMCC_VERSION) && !defined(__clang__)
# define COMPILER_ID "ARMCC"
#if __ARMCC_VERSION >= 1000000
  /* __ARMCC_VERSION = VRRPPPP */
  # define COMPILER_VERSION_MAJOR DEC(__ARMCC_VERSION/1000000)
  # define COMPILER_VERSION_MINOR DEC(__ARMCC_VERSION/10000 % 100)
  # define COMPILER_VERSION_PATCH DEC(__ARMCC_VERSION     % 10000)
#else
  /* __ARMCC_VERSION = VRPPPP */
  # define COMPILER_VERSION_MAJOR DEC(__ARMCC_VERSION/100000)
  # define COMPILER_VERSION_MINOR DEC(__ARMCC_VERSION/10000 % 10)
  # define COMPILER_VERSION_PATCH DEC(__ARMCC_VERSION    % 10000)
#endif


#elif defined(__clang__) && defined(__apple_build_version__)
# define COMPILER_ID "AppleClang"
# if defined(_MSC_VER)
#  define SIMULATE_ID "MSVC"
# endif
# define COMPILER_VERSION_MAJOR DEC(__clang_major__)
# define COMPILER_VERSION_MINOR DEC(__clang_minor__)
# define COMPILER_VERSION_PATCH DEC(__clang_patchlevel__)
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define SIMULATE_VERSION_MAJOR DEC(_MSC_VER / 100)
#  define SIMULATE_VERSION_MINOR DEC(_MSC_VER % 100)
# endif
# define COMPILER_VERSION_TWEAK DEC(__apple_build_version__)

#elif defined(__clang__) && defined(__ARMCOMPILER_VERSION)
# define COMPILER_ID "ARMClang"
  # define COMPILER_VERSION_MAJOR DEC(__ARMCOMPILER_VERSION/1000000)
  # define COMPILER_VERSION_MINOR DEC(__ARMCOMPILER_VERSION/10000 % 100)
  # define COMPILER_VERSION_PATCH DEC(__ARMCOMPILER_VERSION/100   % 100)
# define COMPILER_VERSION_INTERNAL DEC(__ARMCOMPILER_VERSION)

#elif defined(__clang__) && defined(__ti__)
# define COMPILER_ID "TIClang"
  # define COMPILER_VERSION_MAJOR DEC(__ti_major__)
  # define COMPILER_VERSION_MINOR DEC(__ti_minor__)
  # define COMPILER_VERSION_PATCH DEC(__ti_patchlevel__)
# define COMPILER_VERSION_INTERNAL DEC(__ti_version__)

#elif defined(__clang__)
# define COMPILER_ID "Clang"
# if defined(_MSC_VER)
#  define SIMULATE_ID "MSVC"
# endif
# define COMPILER_VERSION_MAJOR DEC(__clang_major__)
# define COMPILER_VERSION_MINOR DEC(__clang_minor__)
# define COMPILER_VERSION_PATCH DEC(__clang_patchlevel__)
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define SIMULATE_VERSION_MAJOR DEC(_MSC_VER / 100)
#  define SIMULATE_VERSION_MINOR DEC(_MSC_VER % 100)
# endif

#elif defined(__LCC__) && (defined(__GNUC__) || defined(__GNUG__) || defined(__MCST__))
# define COMPILER_ID "LCC"
# define COMPILER_VERSION_MAJOR DEC(__LCC__ / 100)
# define COMPILER_VERSION_MINOR DEC(__LCC__ % 100)
# if defined(__LCC_MINOR__)
#  define COMPILER_VERSION_PATCH DEC(__LCC_MINOR__)
# endif
# if defined(__GNUC__) && defined(__GNUC_MINOR__)
#  define SIMULATE_ID "GNU"
#  define SIMULATE_VERSION_MAJOR DEC(__GNUC__)
#  define SIMULATE_VERSION_MINOR DEC(__GNUC_MINOR__)
#  if defined(__GNUC_PATCHLEVEL__)
#   define SIMULATE_VERSION_PATCH DEC(__GNUC_PATCHLEVEL__)
#  endif
# endif

#elif defined(__GNUC__) || defined(__GNUG__)
# define COMPILER_ID "GNU"
# if defined(__GNUC__)
#  define COMPILER_VERSION_MAJOR DEC(__GNUC__)
# else
#  define COMPILER_VERSION_MAJOR DEC(__GNUG__)
# endif
# if defined(__GNUC_MINOR__)
#  define COMPILER_VERSION_MINOR DEC(__GNUC_MINOR__)
# endif
# if defined(__GNUC_PATCHLEVEL__)
#  define COMPILER_VERSION_PATCH DEC(__GNUC_PATCHLEVEL__)
# endif

#elif defined(_MSC_VER)
# define COMPILER_ID "MSVC"
  /* _MSC_VER = VVRR */
# define COMPILER_VERSION_MAJOR DEC(_MSC_VER / 100)
# define COMPILER_VERSION_MINOR DEC(_MSC_VER % 100)
# if defined(_MSC_FULL_VER)
#  if _MSC_VER >= 1400
    /* _MSC_FULL_VER = VVRRPPPPP */
#   define COMPILER_VERSION_PATCH DEC(_MSC_FULL_VER % 100000)
#  else
    /* _MSC_FULL_VER = VVRRPPPP */
#   define COMPILER_VERSION_PATCH DEC(_MSC_FULL_VER % 10000)
#  endif
# endif
# if defined(_MSC_BUILD)
#  define COMPILER_VERSION_TWEAK DEC(_MSC_BUILD)
# endif

#elif defined(_ADI_COMPILER)
# define COMPILER_ID "ADSP"
#if defined(__VERSIONNUM__)
  /* __VERSIONNUM__ = 0xVVRRPPTT */
#  define COMPILER_VERSION_MAJOR DEC(__VERSIONNUM__ >> 24 & 0xFF)
#  define COMPILER_VERSION_MINOR DEC(__VERSIONNUM__ >> 16 & 0xFF)
#  define COMPILER_VERSION_PATCH DEC(__VERSIONNUM__ >> 8 & 0xFF)
#  define COMPILER_VERSION_TWEAK DEC(__VERSIONNUM__ & 0xFF)
#endif

#elif defined(__IAR_SYSTEMS_ICC__) || defined(__IAR_SYSTEMS_ICC)
# define COMPILER_ID "IAR"
# if defined(__VER__) && defined(__ICCARM__)
#  define COMPILER_VERSION_MAJOR DEC((__VER__) / 1000000)
#  define COMPILER_VERSION_MINOR DEC(((__VER__) / 1000) % 1000)
#  define COMPILER_VERSION_PATCH DEC((__VER__) % 1000)
#  define COMPILER_VERSION_INTERNAL DEC(__IAR_SYSTEMS_ICC__)
# elif defined(__VER__) && (defined(__ICCAVR__) || defined(__ICCRX__) || defined(__ICCRH850__) || defined(__ICCRL78__) || defined(__ICC430__) || defined(__ICCRISCV__) || defined(__ICCV850__) || defined(__ICC8051__) || defined(__ICCSTM8__))
#  define COMPILER_VERSION_MAJOR DEC((__VER__) / 100)
#  define COMPILER_VERSION_MINOR DEC((__VER__) - (((__VER__) / 100)*100))
#  define COMPILER_VERSION_PATCH DEC(__SUBVERSION__)
#  define COMPILER_VERSION_INTERNAL DEC(__IAR_SYSTEMS_ICC__)
# endif

#elif defined(__DCC__) && defined(_DIAB_TOOL)
# define COMPILER_ID "Diab"
  # define COMPILER_VERSION_MAJOR DEC(__VERSION_MAJOR_NUMBER__)
  # define COMPILER_VERSION_MINOR DEC(__VERSION_MINOR_NUMBER__)
  # define COMPILER_VERSION_PATCH DEC(__VERSION_ARCH_FEATURE_NUMBER__)
  # define COMPILER_VERSION_TWEAK DEC(__VERSION_BUG_FIX_NUMBER__)



/* These compilers are either not known or too old to define an
  identification macro.  Try to identify the platform and guess that
  it is the native compiler.  */
#elif defined(__hpux) || defined(__hpua)
# define COMPILER_ID "HP"

#else /* unknown compiler */
# define COMPILER_ID ""
#endif

/* Construct the string literal in pieces to prevent the source from
   getting matched.  Store it in a pointer rather than an array
   because some compilers will just produce instructions to fill the
   array rather than assigning a pointer to a static array.  */
char const* info_compiler = "INFO" ":" "compiler[" COMPILER_ID "]";
#ifdef SIMULATE_ID
char const* info_simulate = "INFO" ":" "simulate[" SIMULATE_ID "]";
#endif

#ifdef __QNXNTO__
char const* qnxnto = "INFO" ":" "qnxnto[]";
#endif

#if defined(__CRAYXT_COMPUTE_LINUX_TARGET)
char const *info_cray = "INFO" ":" "compiler_wrapper[CrayPrgEnv]";
#endif

#define STRINGIFY_HELPER(X) #X
#define STRINGIFY(X) STRINGIFY_HELPER(X)

/* Identify known platforms by name.  */
#if defined(__linux) || defined(__linux__) || defined(linux)
# define PLATFORM_ID "Linux"

#elif defined(__MSYS__)
# define PLATFORM_ID "MSYS"

#elif defined(__CYGWIN__)
# define PLATFORM_ID "Cygwin"

#elif defined(__MINGW32__)
# define PLATFORM_ID "MinGW"

#elif defined(__APPLE__)
# define PLATFORM_ID "Darwin"

#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
# define PLATFORM_ID "Windows"

#elif defined(__FreeBSD__) || defined(__FreeBSD)
# define PLATFORM_ID "FreeBSD"

#elif defined(__NetBSD__) || defined(__NetBSD)
# define PLATFORM_ID "NetBSD"

#elif defined(__OpenBSD__) || defined(__OPENBSD)
# define PLATFORM_ID "OpenBSD"

#elif defined(__sun) || defined(sun)
# define PLATFORM_ID "SunOS"

#elif defined(_AIX) || defined(__AIX) || defined(__AIX__) || defined(__aix) || defined(__aix__)
# define PLATFORM_ID "AIX"

#elif defined(__hpux) || defined(__hpux__)
# define PLATFORM_ID "HP-UX"

#elif defined(__HAIKU__)
# define PLATFORM_ID "Haiku"

#elif defined(__BeOS) || defined(__BEOS__) || defined(_BEOS)
# define PLATFORM_ID "BeOS"

#elif defined(__QNX__) || defined(__QNXNTO__)
# define PLATFORM_ID "QNX"

#elif defined(__tru64) || defined(_tru64) || defined(__TRU64__)
# define PLATFORM_ID "Tru64"

#elif defined(__riscos) || defined(__riscos__)
# define PLATFORM_ID "RISCos"

#elif defined(__sinix) || defined(__sinix__) || defined(__SINIX__)
# define PLATFORM_ID "SINIX"

#elif defined(__UNIX_SV__)
# define PLATFORM_ID "UNIX_SV"

#elif defined(__bsdos__)
# define PLATFORM_ID "BSDOS"

#elif defined(_MPRAS) || defined(MPRAS)
# define PLATFORM_ID "MP-RAS"

#elif defined(__osf) || defined(__osf__)
# define PLATFORM_ID "OSF1"

#elif defined(_SCO_SV) || defined(SCO_SV) || defined(sco_sv)
# define PLATFORM_ID "SCO_SV"

#elif defined(__ultrix) || defined(__ultrix__) || defined(_ULTRIX)
# define PLATFORM_ID "ULTRIX"

#elif defined(__XENIX__) || defined(_XENIX) || defined(XENIX)
# define PLATFORM_ID "Xenix"

#elif defined(__WATCOMC__)
# if defined(__LINUX__)
#  define PLATFORM_ID "Linux"

# elif defined(__DOS__)
#  define PLATFORM_ID "DOS"

# elif defined(__OS2__)
#  define PLATFORM_ID "OS2"

# elif defined(__WINDOWS__)
#  define PLATFORM_ID "Windows3x"

# elif defined(__VXWORKS__)
#  define PLATFORM_ID "VxWorks"

# else /* unknown platform */
#  define PLATFORM_ID
# endif

#elif defined(__INTEGRITY)
# if defined(INT_178B)
#  define PLATFORM_ID "Integrity178"

# else /* regular Integrity */
#  define PLATFORM_ID "Integrity"
# endif

# elif defined(_ADI_COMPILER)
#  define PLATFORM_ID "ADSP"

#else /* unknown platform */
# define PLATFORM_ID

#endif

/* For windows compilers MSVC and Intel we can determine
   the architecture of the compiler being used.  This is because
   the compilers do not have flags that can change the architecture,
   but rather depend on which compiler is being used
*/
#if defined(_WIN32) && defined(_MSC_VER)
# if defined(_M_IA64)
#  define ARCHITECTURE_ID "IA64"

# elif defined(_M_ARM64EC)
#  define ARCHITECTURE_ID "ARM64EC"

# elif defined(_M_X64) || defined(_M_AMD64)
#  define ARCHITECTURE_ID "x64"

# elif defined(_M_IX86)
#  define ARCHITECTURE_ID "X86"

# elif defined(_M_ARM64)
#  define ARCHITECTURE_ID "ARM64"

# elif defined(_M_ARM)
#  if _M_ARM == 4
#   define ARCHITECTURE_ID "ARMV4I"
#  elif _M_ARM == 5
#   define ARCHITECTURE_ID "ARMV5I"
#  else
#   define ARCHITECTURE_ID "ARMV" STRINGIFY(_M_ARM)
#  endif

# elif defined(_M_MIPS)
#  define ARCHITECTURE_ID "MIPS"

# elif defined(_M_SH)
#  define ARCHITECTURE_ID "SHx"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__WATCOMC__)
# if defined(_M_I86)
#  define ARCHITECTURE_ID "I86"

# elif defined(_M_IX86)
#  define ARCHITECTURE_ID "X86"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__IAR_SYSTEMS_ICC__) || defined(__IAR_SYSTEMS_ICC)
# if defined(__ICCARM__)
#  define ARCHITECTURE_ID "ARM"

# elif defined(__ICCRX__)
#  define ARCHITECTURE_ID "RX"

# elif defined(__ICCRH850__)
#  define ARCHITECTURE_ID "RH850"

# elif defined(__ICCRL78__)
#  define ARCHITECTURE_ID "RL78"

# elif defined(__ICCRISCV__)
#  define ARCHITECTURE_ID "RISCV"

# elif defined(__ICCAVR__)
#  define ARCHITECTURE_ID "AVR"

# elif defined(__ICC430__)
#  define ARCHITECTURE_ID "MSP430"

# elif defined(__ICCV850__)
#  define ARCHITECTURE_ID "V850"

# elif defined(__ICC8051__)
#  define ARCHITECTURE_ID "8051"

# elif defined(__ICCSTM8__)
#  define ARCHITECTURE_ID "STM8"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__ghs__)
# if defined(__PPC64__)
#  define ARCHITECTURE_ID "PPC64"

# elif defined(__ppc__)
#  define ARCHITECTURE_ID "PPC"

# elif defined(__ARM__)
#  define ARCHITECTURE_ID "ARM"

# elif defined(__x86_64__)
#  define ARCHITECTURE_ID "x64"

# elif defined(__i386__)
#  define ARCHITECTURE_ID "X86"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__clang__) && defined(__ti__)
# if defined(__ARM_ARCH)
#  define ARCHITECTURE_ID "ARM"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__TI_COMPILER_VERSION__)
# if defined(__TI_ARM__)
#  define ARCHITECTURE_ID "ARM"

# elif defined(__MSP430__)
#  define ARCHITECTURE_ID "MSP430"

# elif defined(__TMS320C28XX__)
#  define ARCHITECTURE_ID "TMS320C28x"

# elif defined(__TMS320C6X__) || defined(_TMS320C6X)
#  define ARCHITECTURE_ID "TMS320C6x"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

# elif defined(__ADSPSHARC__)
#  define ARCHITECTURE_ID "SHARC"

# elif defined(__ADSPBLACKFIN__)
#  define ARCHITECTURE_ID "Blackfin"

#elif defined(__TASKING__)

# if defined(__CTC__) || defined(__CPTC__)
#  define ARCHITECTURE_ID "TriCore"

# elif defined(__CMCS__)
#  define ARCHITECTURE_ID "MCS"

# elif defined(__CARM__) || defined(__CPARM__)
#  define ARCHITECTURE_ID "ARM"

# elif defined(__CARC__)
#  define ARCHITECTURE_ID "ARC"

# elif defined(__C51__)
#  define ARCHITECTURE_ID "8051"

# elif defined(__CPCP__)
#  define ARCHITECTURE_ID "PCP"

# else
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__RENESAS__)
# if defined(__CCRX__)
#  define ARCHITECTURE_ID "RX"

# elif defined(__CCRL__)
#  define ARCHITECTURE_ID "RL78"

# elif defined(__CCRH__)
#  define ARCHITECTURE_ID "RH850"

# else
#  define ARCHITECTURE_ID ""
# endif

#else
#  define ARCHITECTURE_ID
#endif

/* Convert integer to decimal digit literals.  */
#define DEC(n)                   \
  ('0' + (((n) / 10000000)%10)), \
  ('0' + (((n) / 1000000)%10)),  \
  ('0' + (((n) / 100000)%10)),   \
  ('0' + (((n) / 10000)%10)),    \
  ('0' + (((n) / 1000)%10)),     \
  ('0' + (((n) / 100)%10)),      \
  ('0' + (((n) / 10)%10)),       \
  ('0' +  ((n) % 10))

/* Convert integer to hex digit literals.  */
#define HEX(n)             \
  ('0' + ((n)>>28 & 0xF)), \
  ('0' + ((n)>>24 & 0xF)), \
  ('0' + ((n)>>20 & 0xF)), \
  ('0' + ((n)>>16 & 0xF)), \
  ('0' + ((n)>>12 & 0xF)), \
  ('0' + ((n)>>8  & 0xF)), \
  ('0' + ((n)>>4  & 0xF)), \
  ('0' + ((n)     & 0xF))

/* Construct a string literal encoding the version number. */
#ifdef COMPILER_VERSION
char const* info_version = "INFO" ":" "compiler_version[" COMPILER_VERSION "]";

/* Construct a string literal encoding the version number components. */
#elif defined(COMPILER_VERSION_MAJOR)
char const info_version[] = {
  'I', 'N', 'F', 'O', ':',
  'c','o','m','p','i','l','e','r','_','v','e','r','s','i','o','n','[',
  COMPILER_VERSION_MAJOR,
# ifdef COMPILER_VERSION_MINOR
  '.', COMPILER_VERSION_MINOR,
#  ifdef COMPILER_VERSION_PATCH
   '.', COMPILER_VERSION_PATCH,
#   ifdef COMPILER_VERSION_TWEAK
    '.', COMPILER_VERSION_TWEAK,
#   endif
#  endif
# endif
  ']','\0'};
#endif

/* Construct a string literal encoding the internal version number. */
#ifdef COMPILER_VERSION_INTERNAL
char const info_version_internal[] = {
  'I', 'N', 'F', 'O', ':',
  'c','o','m','p','i','l','e','r','_','v','e','r','s','i','o','n','_',
  'i','n','t','e','r','n','a','l','[',
  COMPILER_VERSION_INTERNAL,']','\0'};
#elif defined(COMPILER_VERSION_INTERNAL_STR)
char const* info_version_internal = "INFO" ":" "compiler_version_internal[" COMPILER_VERSION_INTERNAL_STR "]";
#endif

/* Construct a string literal encoding the version number components. */
#ifdef SIMULATE_VERSION_MAJOR
char const info_simulate_version[] = {
  'I', 'N', 'F', 'O', ':',
  's','i','m','u','l','a','t','e','_','v','e','r','s','i','o','n','[',
  SIMULATE_VERSION_MAJOR,
# ifdef SIMULATE_VERSION_MINOR
  '.', SIMULATE_VERSION_MINOR,
#  ifdef SIMULATE_VERSION_PATCH
   '.', SIMULATE_VERSION_PATCH,
#   ifdef SIMULATE_VERSION_TWEAK
    '.', SIMULATE_VERSION_TWEAK,
#   endif
#  endif
# endif
  ']','\0'};
#endif

/* Construct the string literal in pieces to prevent the source from
   getting matched.  Store it in a pointer rather than an array
   because some compilers will just produce instructions to fill the
   array rather than assigning a pointer to a static array.  */
char const* info_platform = "INFO" ":" "platform[" PLATFORM_ID "]";
char const* info_arch = "INFO" ":" "arch[" ARCHITECTURE_ID "]";



#define CXX_STD_98 199711L
#define CXX_STD_11 201103L
#define CXX_STD_14 201402L
#define CXX_STD_17 201703L
#define CXX_STD_20 202002L
#define CXX_STD_23 202302L

#if defined(__INTEL_COMPILER) && defined(_MSVC_LANG)
#  if _MSVC_LANG > CXX_STD_17
#    define CXX_STD _MSVC_LANG
#  elif _MSVC_LANG == CXX_STD_17 && defined(__cpp_aggregate_paren_init)
#    define CXX_STD CXX_STD_20
#  elif _MSVC_LANG > CXX_STD_14 && __cplusplus > CXX_STD_17
#    define CXX_STD CXX_STD_20
#  elif _MSVC_LANG > CXX_STD_14
#    define CXX_STD CXX_STD_17
#  elif defined(__INTEL_CXX11_MODE__) && defined(__cpp_aggregate_nsdmi)
#    define CXX_STD CXX_STD_14
#  elif defined(__INTEL_CXX11_MODE__)
#    define CXX_STD CXX_STD_11
#  else
#    define CXX_STD CXX_STD_98
#  endif
#elif defined(_MSC_VER) && defined(_MSVC_LANG)
#  if _MSVC_LANG > __cplusplus
#    define CXX_STD _MSVC_LANG
#  else
#    define CXX_STD __cplusplus
#  endif
#elif defined(__NVCOMPILER)
#  if __cplusplus == CXX_STD_17 && defined(__cpp_aggregate_paren_init)
#    define CXX_STD CXX_STD_20
#  else
#    define CXX_STD __cplusplus
#  endif
#elif defined(__INTEL_COMPILER) || defined(__PGI)
#  if __cplusplus == CXX_STD_11 && defined(__cpp_namespace_attributes)
#    define CXX_STD CXX_STD_17
#  elif __cplusplus == CXX_STD_11 && defined(__cpp_aggregate_nsdmi)
#    define CXX_STD CXX_STD_14
#  else
#    define CXX_STD __cplusplus
#  endif
#elif (defined(__IBMCPP__) || defined(__ibmxl__)) && defined(__linux__)
#  if __cplusplus == CXX_STD_11 && defined(__cpp_aggregate_nsdmi)
#    define CXX_STD CXX_STD_14
#  else
#    define CXX_STD __cplusplus
#  endif
#elif __cplusplus == 1 && defined(__GXX_EXPERIMENTAL_CXX0X__)
#  define CXX_STD CXX_STD_11
#else
#  define CXX_STD __cplusplus
#endif

const char* info_language_standard_default = "INFO" ":" "standard_default["
#if CXX_STD > CXX_STD_23
  "26"
#elif CXX_STD > CXX_STD_20
  "23"
#elif CXX_STD > CXX_STD_17
  "20"
#elif CXX_STD > CXX_STD_14
  "17"
#elif CXX_STD > CXX_STD_11
  "14"
#elif CXX_STD >= CXX_STD_11
  "11"
#else
  "98"
#endif
"]";

const char* info_language_extensions_default = "INFO" ":" "extensions_default["
#if (defined(__clang__) || defined(__GNUC__) || defined(__xlC__) ||           \
     defined(__TI_COMPILER_VERSION__) || defined(__RENESAS__)) &&             \
  !defined(__STRICT_ANSI__)
  "ON"
#else
  "OFF"
#endif
"]";

/*--------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
  int require = 0;
  require += info_compiler[argc];
  require += info_platform[argc];
  require += info_arch[argc];
#ifdef COMPILER_VERSION_MAJOR
  require += info_version[argc];
#endif
#if defined(COMPILER_VERSION_INTERNAL) || defined(COMPILER_VERSION_INTERNAL_STR)
  require += info_version_internal[argc];
#endif
#ifdef SIMULATE_ID
  require += info_simulate[argc];
#endif
#ifdef SIMULATE_VERSION_MAJOR
  require += info_simulate_version[argc];
#endif
#if defined(__CRAYXT_COMPUTE_LINUX_TARGET)
  require += info_cray[argc];
#endif
  require += info_language_standard_default[argc];
  require += info_language_extensions_default[argc];
  (void)argv;
  return require;
}

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\explicit_neural\explicit_acoustic_gen.cpp ===
#include "include/explicit_acoustic_gen.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#if VA_HAS_ONNX_RUNTIME
#include "ort_contract.hpp"
#endif

namespace vocal {
struct ExplicitAcousticGenerator::Impl {
    ExplicitAcousticConfig config;
    double elapsed_ms{};
#if VA_HAS_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "explicit-acoustic"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    Impl(const std::filesystem::path& path, ExplicitAcousticConfig supplied) : config(std::move(supplied)) {
        if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error("missing acoustic generator: " + path.string());
        options = detail::explicit_session_options(config.intra_op_threads, config.thread_affinities);
        session = Ort::Session(environment, path.c_str(), options);
        if (session.GetInputCount() != 5)
            throw std::runtime_error("explicit acoustic generator must expose exactly five inputs");
        const std::array<std::int64_t, 2> dynamic{1, -1};
        const std::array<std::int64_t, 1> speaker{1};
        detail::require_input(session, "input_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, dynamic);
        detail::require_input(session, "durations", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, dynamic);
        detail::require_input(session, "f0", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dynamic);
        detail::require_input(session, "energy", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dynamic);
        detail::require_input(session, "sid", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, speaker);
        detail::require_output(session, config.mel_output);
    }
#else
    Impl(const std::filesystem::path&, ExplicitAcousticConfig supplied) : config(std::move(supplied)) {
        throw std::runtime_error("explicit-tts requires an ONNX Runtime-enabled build; use msvc-onnx");
    }
#endif
};

ExplicitAcousticGenerator::ExplicitAcousticGenerator(const std::filesystem::path& path, ExplicitAcousticConfig config)
    : impl_(std::make_unique<Impl>(path, std::move(config))) {}
ExplicitAcousticGenerator::~ExplicitAcousticGenerator() = default;
ExplicitAcousticGenerator::ExplicitAcousticGenerator(ExplicitAcousticGenerator&&) noexcept = default;
ExplicitAcousticGenerator& ExplicitAcousticGenerator::operator=(ExplicitAcousticGenerator&&) noexcept = default;
double ExplicitAcousticGenerator::inference_ms() const noexcept { return impl_->elapsed_ms; }

MelSpectrogram ExplicitAcousticGenerator::infer(std::span<const std::int64_t> token_ids, const ProsodyControls& controls) {
    validate_prosody(controls, token_ids.size());
    for (auto token : token_ids) if (token < 0) throw std::invalid_argument("token IDs must be nonnegative");
#if VA_HAS_ONNX_RUNTIME
    const auto frames = controls.f0_contour.size();
    const std::array<std::int64_t, 2> tokens_shape{1, static_cast<std::int64_t>(token_ids.size())};
    const std::array<std::int64_t, 2> frames_shape{1, static_cast<std::int64_t>(frames)};
    const std::array<std::int64_t, 1> speaker_shape{1};
    const std::array<const char*, 5> names{"input_ids", "durations", "f0", "energy", "sid"};
    detail::require_input(impl_->session, names[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, tokens_shape);
    detail::require_input(impl_->session, names[1], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, tokens_shape);
    detail::require_input(impl_->session, names[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, frames_shape);
    detail::require_input(impl_->session, names[3], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, frames_shape);
    // Owned, isolated writable buffers outlive all input tensors and Run().
    std::vector<std::int64_t> tokens(token_ids.begin(), token_ids.end());
    auto durations = controls.durations;
    auto f0 = controls.f0_contour;
    auto energy = controls.energy_contour;
    std::array<std::int64_t, 1> speaker{controls.speaker_id};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, tokens.data(), tokens.size(), tokens_shape.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, durations.data(), durations.size(), tokens_shape.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(memory, f0.data(), f0.size(), frames_shape.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(memory, energy.data(), energy.size(), frames_shape.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, speaker.data(), 1, speaker_shape.data(), 1));
    const char* output = impl_->config.mel_output.c_str();
    const auto started = std::chrono::steady_clock::now();
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, names.data(), inputs.data(), inputs.size(), &output, 1);
    impl_->elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const auto shape = detail::float_output_shape(outputs[0]);
    if (shape != std::vector<std::int64_t>{1, 80, static_cast<std::int64_t>(frames)})
        throw std::runtime_error("acoustic mel output must be [1, 80, sum(durations)]");
    // The framework uses frame-major layout; ONNX exports use channel-major.
    const float* data = outputs[0].GetTensorData<float>();
    MelSpectrogram mel{frames, 80, std::vector<float>(frames * 80)};
    for (std::size_t bin = 0; bin < 80; ++bin)
        for (std::size_t frame = 0; frame < frames; ++frame) {
            const float value = data[bin * frames + frame];
            if (!std::isfinite(value)) throw std::runtime_error("non-finite acoustic mel output");
            mel.log_mel[frame * 80 + bin] = value;
        }
    return mel;
#else
    throw std::runtime_error("ONNX Runtime support was not compiled");
#endif
}
} // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\explicit_neural\length_regulator.cpp ===
#include "include/length_regulator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace vocal {
namespace {
ProsodyControls remap_contours(const ProsodyControls& controls, std::vector<std::int64_t> durations) {
    ProsodyControls result;
    result.durations = std::move(durations);
    result.speaker_id = controls.speaker_id;
    result.token_kinds = controls.token_kinds;
    const auto frames = duration_frames(result.durations);
    result.f0_contour.reserve(frames);
    result.energy_contour.reserve(frames);
    std::size_t offset = 0;
    for (std::size_t token = 0; token < controls.durations.size(); ++token) {
        const auto old_count = static_cast<std::size_t>(controls.durations[token]);
        const auto new_count = static_cast<std::size_t>(result.durations[token]);
        for (std::size_t frame = 0; frame < new_count; ++frame) {
            const auto source = offset + std::min(old_count - 1, frame * old_count / new_count);
            result.f0_contour.push_back(controls.f0_contour[source]);
            result.energy_contour.push_back(controls.energy_contour[source]);
        }
        offset += old_count;
    }
    return result;
}
} // namespace

std::size_t duration_frames(std::span<const std::int64_t> durations, std::size_t maximum) {
    std::size_t total = 0;
    for (auto duration : durations) {
        if (duration < 0 || static_cast<std::uint64_t>(duration) > maximum - total)
            throw std::invalid_argument("negative duration or duration sum exceeds frame limit");
        total += static_cast<std::size_t>(duration);
    }
    return total;
}

void validate_prosody(const ProsodyControls& controls, std::size_t tokens, std::size_t maximum) {
    if (tokens == 0 || tokens > 4096 || controls.durations.size() != tokens)
        throw std::invalid_argument("durations must have one entry for each of 1..4096 tokens");
    if (!controls.token_kinds.empty() && controls.token_kinds.size() != tokens)
        throw std::invalid_argument("token annotations must be empty or match durations");
    for (auto kind : controls.token_kinds)
        if (kind < ProsodyTokenKind::Unknown || kind > ProsodyTokenKind::Boundary)
            throw std::invalid_argument("invalid token annotation");
    const auto frames = duration_frames(controls.durations, maximum);
    if (frames == 0 || controls.f0_contour.size() != frames || controls.energy_contour.size() != frames)
        throw std::invalid_argument("F0 and energy must each match the positive sum of durations");
    if (controls.speaker_id < 0) throw std::invalid_argument("speaker ID must be nonnegative");
    for (float value : controls.f0_contour)
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("F0 must be finite and nonnegative");
    for (float value : controls.energy_contour)
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("energy must be finite and nonnegative");
}

ProsodyControls baseline_prosody(std::size_t tokens, std::int64_t frames_per_token, float f0_hz, float energy) {
    if (tokens == 0 || tokens > 4096 || frames_per_token <= 0)
        throw std::invalid_argument("baseline needs 1..4096 tokens and positive frames per token");
    ProsodyControls controls;
    controls.durations.assign(tokens, frames_per_token);
    const auto frames = duration_frames(controls.durations);
    controls.f0_contour.assign(frames, f0_hz);
    controls.energy_contour.assign(frames, energy);
    validate_prosody(controls, tokens);
    return controls;
}

ProsodyControls apply_prosody_sliders(const ProsodyControls& controls, const ProsodySliders& sliders) {
    validate_prosody(controls, controls.durations.size());
    if (!std::isfinite(sliders.pitch_scale) || sliders.pitch_scale <= 0 ||
        !std::isfinite(sliders.speed) || sliders.speed <= 0 ||
        !std::isfinite(sliders.energy_scale) || sliders.energy_scale < 0 ||
        !std::isfinite(sliders.energy_variance) || sliders.energy_variance < 0)
        throw std::invalid_argument("pitch/speed must be positive, energy sliders nonnegative, all finite");
    std::vector<std::int64_t> durations;
    for (auto duration : controls.durations) {
        const double scaled = static_cast<double>(duration) / sliders.speed;
        if (scaled > maximum_prosody_frames)
            throw std::invalid_argument("speed exceeds frame limit");
        durations.push_back(duration == 0 ? 0 : std::max<std::int64_t>(1, std::llround(scaled)));
    }
    auto result = remap_contours(controls, std::move(durations));
    const double mean = std::accumulate(controls.energy_contour.begin(), controls.energy_contour.end(), 0.0) /
                        static_cast<double>(controls.energy_contour.size());
    for (std::size_t frame = 0; frame < result.f0_contour.size(); ++frame) {
        result.f0_contour[frame] *= sliders.pitch_scale;
        const double energy = mean + (result.energy_contour[frame] - mean) * sliders.energy_variance;
        result.energy_contour[frame] = static_cast<float>(std::max(0.0, energy) * sliders.energy_scale);
    }
    validate_prosody(result, result.durations.size());
    return result;
}

ProsodyControls apply_emotion_preset(const ProsodyControls& base, VocalEmotion emotion) {
    validate_prosody(base, base.durations.size());
    if (emotion == VocalEmotion::Neutral) return base;
    if (emotion < VocalEmotion::Neutral || emotion > VocalEmotion::Authoritative)
        throw std::invalid_argument("invalid emotion preset");
    auto result = apply_prosody_sliders(base, {1.0F,
        emotion == VocalEmotion::Excited ? 1.15F : emotion == VocalEmotion::Somber ? .88F : 1.0F, 1.0F, 1.0F});
    if (!base.token_kinds.empty()) {
        auto durations = result.durations;
        for (std::size_t token = 0; token < durations.size(); ++token) {
            if (durations[token] == 0) continue;
            if (emotion == VocalEmotion::Whisper && base.token_kinds[token] == ProsodyTokenKind::UnvoicedConsonant)
                durations[token] = static_cast<std::int64_t>(std::ceil(durations[token] * 1.1));
            if (emotion == VocalEmotion::Authoritative && base.token_kinds[token] == ProsodyTokenKind::Boundary)
                durations[token] = std::max<std::int64_t>(1, durations[token] / 2);
        }
        result = remap_contours(result, std::move(durations));
    }
    double voiced_sum = 0;
    std::size_t voiced_frames = 0;
    for (float f0 : base.f0_contour) if (f0 > 0) { voiced_sum += f0; ++voiced_frames; }
    const double mean_f0 = voiced_frames ? voiced_sum / voiced_frames : 0;
    const double mean_energy = std::accumulate(base.energy_contour.begin(), base.energy_contour.end(), 0.0) /
                               base.energy_contour.size();
    for (std::size_t frame = 0; frame < result.f0_contour.size(); ++frame) {
        auto& f0 = result.f0_contour[frame];
        auto& energy = result.energy_contour[frame];
        double pitch = f0;
        if (f0 > 0) {
            if (emotion == VocalEmotion::Whisper) pitch = mean_f0 + (f0 - mean_f0) * .15;
            else if (emotion == VocalEmotion::Excited) pitch = mean_f0 + 35 + (f0 - mean_f0) * 1.6;
            else if (emotion == VocalEmotion::Somber) pitch = mean_f0 - 25 + (f0 - mean_f0) * .65;
            else pitch = std::clamp(mean_f0, 140.0, 190.0) + (f0 - mean_f0) * .35;
            f0 = static_cast<float>(std::max(1.0, pitch));
        }
        double level = energy;
        if (emotion == VocalEmotion::Whisper) level *= .6;
        else if (emotion == VocalEmotion::Excited) level = (mean_energy + (level - mean_energy) * 1.35) * 1.1;
        else if (emotion == VocalEmotion::Somber) level = mean_energy * .85 + (level - mean_energy) * .4;
        else level *= 1.05;
        energy = static_cast<float>(std::max(0.0, level));
    }
    std::size_t offset = 0, phrase_begin = 0;
    bool initial_stress = true;
    auto falling_tail = [&](std::size_t end) {
        if (end <= phrase_begin) return;
        const auto tail = std::max<std::size_t>(1, (end - phrase_begin) / 5);
        for (std::size_t frame = end - tail; frame < end; ++frame)
            if (result.f0_contour[frame] > 0)
                result.f0_contour[frame] = std::max(1.0F, result.f0_contour[frame] *
                    static_cast<float>(1.0 - .1 * (frame - (end - tail) + 1) / tail));
    };
    for (std::size_t token = 0; token < result.durations.size(); ++token) {
        const auto count = static_cast<std::size_t>(result.durations[token]);
        const auto kind = result.token_kinds.empty() ? ProsodyTokenKind::Unknown : result.token_kinds[token];
        if (emotion == VocalEmotion::Authoritative && initial_stress && count && kind == ProsodyTokenKind::StressedVowel) {
            for (std::size_t frame = offset; frame < offset + count; ++frame) result.energy_contour[frame] *= 1.25F;
            // Adjacent diphthong vowel tokens share this first stressed nucleus.
            if (token + 1 == result.durations.size() || result.token_kinds[token + 1] != ProsodyTokenKind::StressedVowel)
                initial_stress = false;
        }
        if (kind == ProsodyTokenKind::Boundary) {
            if (emotion == VocalEmotion::Somber) falling_tail(offset);
            if (emotion == VocalEmotion::Authoritative)
                for (std::size_t frame = offset; frame < offset + count; ++frame) result.energy_contour[frame] *= .35F;
            phrase_begin = offset + count;
            initial_stress = true;
        }
        offset += count;
    }
    if (emotion == VocalEmotion::Somber) falling_tail(offset);
    validate_prosody(result, result.durations.size());
    return result;
}

ExpandedHiddenStates length_regulate(std::span<const float> hidden_states, std::size_t hidden_dim,
                                     std::span<const std::int64_t> durations, std::size_t maximum_frames) {
    if (hidden_dim == 0 || durations.size() > std::numeric_limits<std::size_t>::max() / hidden_dim ||
        hidden_states.size() != durations.size() * hidden_dim)
        throw std::invalid_argument("hidden states must have shape [1, durations.size(), hidden_dim]");
    const auto frames = duration_frames(durations, maximum_frames);
    if (frames > std::numeric_limits<std::size_t>::max() / hidden_dim)
        throw std::overflow_error("expanded hidden states size overflows");
    ExpandedHiddenStates result{frames, hidden_dim, {}};
    result.values.reserve(frames * hidden_dim);
    for (std::size_t token = 0; token < durations.size(); ++token) {
        const auto state = hidden_states.subspan(token * hidden_dim, hidden_dim);
        for (std::int64_t frame = 0; frame < durations[token]; ++frame)
            result.values.insert(result.values.end(), state.begin(), state.end());
    }
    return result;
}
} // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\explicit_neural\neural_vocoder.cpp ===
#include "include/neural_vocoder.hpp"
#include "vocal/control_params.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#if VA_HAS_ONNX_RUNTIME
#include "ort_contract.hpp"
#endif

namespace vocal {
struct NeuralVocoder::Impl {
    NeuralVocoderConfig config;
    double elapsed_ms{};
#if VA_HAS_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "explicit-vocoder"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    Impl(const std::filesystem::path& path, NeuralVocoderConfig supplied) : config(std::move(supplied)) {
        if ((config.sample_rate_hz != 24'000 && config.sample_rate_hz != 22'050) ||
            config.hop_length == 0 || config.hop_length > 4096)
            throw std::invalid_argument("vocoder requires 24 kHz or 22.05 kHz and a hop length in 1..4096");
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("missing neural vocoder: " + path.string());
        options = detail::explicit_session_options(config.intra_op_threads, config.thread_affinities);
        session = Ort::Session(environment, path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        const auto model_rate = session.GetModelMetadata().LookupCustomMetadataMapAllocated("sample_rate", allocator);
        if (model_rate && std::stoi(model_rate.get()) != config.sample_rate_hz)
            throw std::runtime_error("vocoder sample-rate metadata does not match configuration");
        if (session.GetInputCount() != 1) throw std::runtime_error("vocoder must expose one mel input");
        const std::array<std::int64_t, 3> shape{1, 80, -1};
        detail::require_input(session, config.mel_input, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape);
        detail::require_output(session, config.audio_output);
    }
#else
    Impl(const std::filesystem::path&, NeuralVocoderConfig supplied) : config(std::move(supplied)) {
        throw std::runtime_error("neural vocoder requires an ONNX Runtime-enabled build; use msvc-onnx");
    }
#endif
};

NeuralVocoder::NeuralVocoder(const std::filesystem::path& path, NeuralVocoderConfig config)
    : impl_(std::make_unique<Impl>(path, std::move(config))) {}
NeuralVocoder::~NeuralVocoder() = default;
NeuralVocoder::NeuralVocoder(NeuralVocoder&&) noexcept = default;
NeuralVocoder& NeuralVocoder::operator=(NeuralVocoder&&) noexcept = default;
double NeuralVocoder::inference_ms() const noexcept { return impl_->elapsed_ms; }

Waveform NeuralVocoder::synthesize(const MelSpectrogram& mel) {
    if (mel.bins != 80 || mel.frames == 0 || mel.frames > maximum_prosody_frames ||
        mel.log_mel.size() != mel.frames * mel.bins)
        throw std::invalid_argument("vocoder expects a bounded, nonempty 80-bin mel spectrogram");
#if VA_HAS_ONNX_RUNTIME
    std::vector<float> channels(mel.log_mel.size());
    for (std::size_t frame = 0; frame < mel.frames; ++frame)
        for (std::size_t bin = 0; bin < mel.bins; ++bin) {
            const float value = mel.log_mel[frame * mel.bins + bin];
            if (!std::isfinite(value)) throw std::invalid_argument("vocoder mel input contains non-finite values");
            channels[bin * mel.frames + frame] = value;
        }
    const std::array<std::int64_t, 3> shape{1, 80, static_cast<std::int64_t>(mel.frames)};
    detail::require_input(impl_->session, impl_->config.mel_input, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape);
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory, channels.data(), channels.size(), shape.data(), shape.size());
    const char* input_name = impl_->config.mel_input.c_str();
    const char* output_name = impl_->config.audio_output.c_str();
    const auto started = std::chrono::steady_clock::now();
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, &input_name, &input, 1, &output_name, 1);
    impl_->elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const auto actual = detail::float_output_shape(outputs[0]);
    const auto samples = mel.frames * impl_->config.hop_length;
    if (actual != std::vector<std::int64_t>{1, static_cast<std::int64_t>(samples)} &&
        actual != std::vector<std::int64_t>{1, 1, static_cast<std::int64_t>(samples)})
        throw std::runtime_error("vocoder audio output must be [1, frames * hop_length] or [1, 1, frames * hop_length]");
    const float* data = outputs[0].GetTensorData<float>();
    Waveform audio{impl_->config.sample_rate_hz, std::vector<float>(data, data + samples)};
    for (float value : audio.samples)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite vocoder audio output");
    return audio;
#else
    throw std::runtime_error("ONNX Runtime support was not compiled");
#endif
}
} // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\explicit_neural\prosody_predictor.cpp ===
#include "include/prosody_predictor.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>

#if VA_HAS_ONNX_RUNTIME
#include "ort_contract.hpp"
#endif

namespace vocal {
struct NeuralProsodyPredictor::Impl {
    double elapsed_ms{};
#if VA_HAS_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "explicit-prosody"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    Impl(const std::filesystem::path& path, int threads, const std::string& affinities) {
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("missing trained prosody predictor: " + path.string());
        options = detail::explicit_session_options(threads, affinities);
        session = Ort::Session(environment, path.c_str(), options);
        if (session.GetInputCount() != 2) throw std::runtime_error("prosody predictor needs input_ids and sid");
        detail::require_input(session, "input_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::array<std::int64_t, 2>{1, -1});
        detail::require_input(session, "sid", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::array<std::int64_t, 1>{1});
        for (const auto* name : {"durations", "token_f0_hz", "token_energy"}) detail::require_output(session, name);
    }
#else
    Impl(const std::filesystem::path&, int, const std::string&) {
        throw std::runtime_error("trained prosody predictor requires ONNX Runtime");
    }
#endif
};

NeuralProsodyPredictor::NeuralProsodyPredictor(const std::filesystem::path& path, int threads, const std::string& affinities)
    : impl_(std::make_unique<Impl>(path, threads, affinities)) {}
NeuralProsodyPredictor::~NeuralProsodyPredictor() = default;
double NeuralProsodyPredictor::inference_ms() const noexcept { return impl_->elapsed_ms; }

ProsodyControls NeuralProsodyPredictor::predict(std::span<const std::int64_t> token_ids, std::int64_t speaker,
                                               std::span<const std::int64_t> duration_override) {
    if (token_ids.empty() || token_ids.size() > 4096 || speaker < 0)
        throw std::invalid_argument("prosody predictor needs 1..4096 tokens and a nonnegative speaker ID");
    if (!duration_override.empty() && duration_override.size() != token_ids.size())
        throw std::invalid_argument("durations must match the model token count");
#if VA_HAS_ONNX_RUNTIME
    std::vector<std::int64_t> tokens(token_ids.begin(), token_ids.end());
    const std::array<std::int64_t, 2> token_shape{1, static_cast<std::int64_t>(tokens.size())};
    const std::array<std::int64_t, 1> sid_shape{1};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, tokens.data(), tokens.size(), token_shape.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, &speaker, 1, sid_shape.data(), 1));
    const char* names[] = {"input_ids", "sid"};
    const char* outputs[] = {"durations", "token_f0_hz", "token_energy"};
    const auto started = std::chrono::steady_clock::now();
    auto values = impl_->session.Run(Ort::RunOptions{nullptr}, names, inputs.data(), 2, outputs, 3);
    impl_->elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const std::vector<std::int64_t> expected{1, static_cast<std::int64_t>(tokens.size())};
    if (!values[0].IsTensor() || values[0].GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
        values[0].GetTensorTypeAndShapeInfo().GetShape() != expected ||
        detail::float_output_shape(values[1]) != expected || detail::float_output_shape(values[2]) != expected)
        throw std::runtime_error("prosody outputs must have one duration/F0/energy value per token");
    const auto* durations = values[0].GetTensorData<std::int64_t>();
    const auto* f0 = values[1].GetTensorData<float>();
    const auto* energy = values[2].GetTensorData<float>();
    ProsodyControls result;
    result.speaker_id = speaker;
    if (duration_override.empty()) result.durations.assign(durations, durations + tokens.size());
    else result.durations.assign(duration_override.begin(), duration_override.end());
    const auto frames = duration_frames(result.durations);
    result.f0_contour.reserve(frames); result.energy_contour.reserve(frames);
    for (std::size_t token = 0; token < tokens.size(); ++token) {
        if (!std::isfinite(f0[token]) || f0[token] < 0 || !std::isfinite(energy[token]) || energy[token] < 0)
            throw std::runtime_error("prosody predictor returned invalid F0 or energy");
        result.f0_contour.insert(result.f0_contour.end(), static_cast<std::size_t>(result.durations[token]), f0[token]);
        result.energy_contour.insert(result.energy_contour.end(), static_cast<std::size_t>(result.durations[token]), energy[token]);
    }
    validate_prosody(result, tokens.size());
    return result;
#else
    throw std::runtime_error("trained prosody predictor requires ONNX Runtime");
#endif
}
} // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\piper_onnx\onnx_runtime_model.cpp ===
#include <piper_onnx/onnx_runtime_model.hpp>

#include <chrono>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#if VA_HAS_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#include "ort_session_options.hpp"
#endif

namespace vocal {

double AlignmentDiagnostics::frames_per_token() const {
    return input_tokens ? static_cast<double>(output_frames) / input_tokens : 0.0;
}

struct OnnxRuntimeAcousticModel::Impl {
    OnnxModelConfig config;
    AlignmentDiagnostics diagnostics;
#if VA_HAS_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "vocal-acoustics"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};

    Impl(const std::filesystem::path& model_path, OnnxModelConfig supplied)
        : config(std::move(supplied)) {
        options = detail::cpu_session_options(config.intra_op_threads);
        session = Ort::Session(environment, model_path.c_str(), options);
    }
#else
    Impl(const std::filesystem::path&, OnnxModelConfig supplied) : config(std::move(supplied)) {
        throw std::runtime_error(
            "ONNX Runtime support is disabled; configure with "
            "-DVA_ENABLE_ONNX_RUNTIME=ON -DONNXRUNTIME_ROOT=<installation>");
    }
#endif
};

OnnxRuntimeAcousticModel::OnnxRuntimeAcousticModel(
    const std::filesystem::path& model_path, OnnxModelConfig config)
    : impl_(std::make_unique<Impl>(model_path, std::move(config))) {}
OnnxRuntimeAcousticModel::~OnnxRuntimeAcousticModel() = default;
OnnxRuntimeAcousticModel::OnnxRuntimeAcousticModel(OnnxRuntimeAcousticModel&&) noexcept = default;
OnnxRuntimeAcousticModel& OnnxRuntimeAcousticModel::operator=(OnnxRuntimeAcousticModel&&) noexcept = default;

bool OnnxRuntimeAcousticModel::compiled_with_runtime() noexcept {
#if VA_HAS_ONNX_RUNTIME
    return true;
#else
    return false;
#endif
}

const AlignmentDiagnostics& OnnxRuntimeAcousticModel::diagnostics() const noexcept {
    return impl_->diagnostics;
}

MelSpectrogram OnnxRuntimeAcousticModel::infer(const SynthesisRequest& request) {
#if VA_HAS_ONNX_RUNTIME
    std::vector<std::int64_t> tokens;
    tokens.reserve(request.text.size());
    // The export contract uses byte-level IDs 1..256 and 0 for padding. Replace
    // this tokenizer at the application boundary for phoneme/token-ID exports.
    for (unsigned char c : request.text) tokens.push_back(static_cast<std::int64_t>(c) + 1);
    if (tokens.empty()) tokens.push_back(1);
    const std::array<std::int64_t, 2> input_shape{1, static_cast<std::int64_t>(tokens.size())};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto token_tensor = Ort::Value::CreateTensor<std::int64_t>(
        memory, tokens.data(), tokens.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {impl_->config.token_input.c_str()};
    std::vector<const char*> output_names{impl_->config.mel_output.c_str()};
    if (!impl_->config.duration_output.empty()) output_names.push_back(impl_->config.duration_output.c_str());
    if (!impl_->config.alignment_output.empty()) output_names.push_back(impl_->config.alignment_output.c_str());

    const auto started = std::chrono::steady_clock::now();
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names, &token_tensor, 1,
                                      output_names.data(), output_names.size());
    const auto finished = std::chrono::steady_clock::now();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2 && shape.size() != 3) throw std::runtime_error("mel output must have rank 2 or 3");
    const std::size_t frames = static_cast<std::size_t>(shape[shape.size() - 2]);
    const std::size_t bins = static_cast<std::size_t>(shape.back());
    if (bins != impl_->config.mel_bins) throw std::runtime_error("ONNX mel-bin count does not match configuration");
    const float* mel_data = outputs[0].GetTensorData<float>();
    MelSpectrogram mel{frames, bins, std::vector<float>(mel_data, mel_data + frames * bins)};

    impl_->diagnostics = {};
    impl_->diagnostics.inference_ms =
        std::chrono::duration<double, std::milli>(finished - started).count();
    impl_->diagnostics.input_tokens = tokens.size();
    impl_->diagnostics.output_frames = frames;
    std::size_t output_index = 1;
    if (!impl_->config.duration_output.empty()) {
        const auto count = outputs[output_index].GetTensorTypeAndShapeInfo().GetElementCount();
        const auto type = outputs[output_index].GetTensorTypeAndShapeInfo().GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* values = outputs[output_index].GetTensorData<float>();
            impl_->diagnostics.durations.assign(values, values + count);
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            const auto* values = outputs[output_index].GetTensorData<std::int64_t>();
            impl_->diagnostics.durations.reserve(count);
            for (std::size_t i = 0; i < count; ++i) impl_->diagnostics.durations.push_back(static_cast<float>(values[i]));
        }
        ++output_index;
    }
    if (!impl_->config.alignment_output.empty()) {
        auto info = outputs[output_index].GetTensorTypeAndShapeInfo();
        impl_->diagnostics.alignment_shape = info.GetShape();
        const auto count = info.GetElementCount();
        const float* values = outputs[output_index].GetTensorData<float>();
        impl_->diagnostics.alignment.assign(values, values + count);
    }
    if (!impl_->diagnostics.durations.empty()) {
        double duration_sum = 0.0;
        for (float duration : impl_->diagnostics.durations) {
            duration_sum += duration;
            impl_->diagnostics.zero_duration_tokens += duration <= 0.0F;
        }
        impl_->diagnostics.duration_frame_error = std::abs(duration_sum - static_cast<double>(frames));
    }
    if (impl_->diagnostics.alignment_shape.size() >= 2 && !impl_->diagnostics.alignment.empty()) {
        const auto rows = static_cast<std::size_t>(impl_->diagnostics.alignment_shape[
            impl_->diagnostics.alignment_shape.size() - 2]);
        const auto columns = static_cast<std::size_t>(impl_->diagnostics.alignment_shape.back());
        if (rows && columns && rows * columns <= impl_->diagnostics.alignment.size()) {
            std::vector<bool> covered(columns);
            std::size_t previous = 0;
            for (std::size_t row = 0; row < rows; ++row) {
                std::size_t best = 0;
                for (std::size_t column = 1; column < columns; ++column)
                    if (impl_->diagnostics.alignment[row * columns + column] >
                        impl_->diagnostics.alignment[row * columns + best]) best = column;
                if (row && best < previous) ++impl_->diagnostics.alignment_monotonic_violations;
                previous = best;
                covered[best] = true;
            }
            impl_->diagnostics.alignment_token_coverage =
                static_cast<double>(std::count(covered.begin(), covered.end(), true)) / columns;
        }
    }
    return mel;
#else
    (void)request;
    throw std::runtime_error("ONNX Runtime support was not compiled");
#endif
}

}  // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\piper_onnx\phonemizer.cpp ===
#include "vocal/phonemizer.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vocal {
namespace {

std::vector<std::uint32_t> decode_utf8(std::string_view text) {
    std::vector<std::uint32_t> result;
    for (std::size_t i = 0; i < text.size();) {
        const auto first = static_cast<unsigned char>(text[i++]);
        std::uint32_t cp = first;
        std::size_t continuation = 0;
        if ((first & 0xe0U) == 0xc0U) { cp = first & 0x1fU; continuation = 1; }
        else if ((first & 0xf0U) == 0xe0U) { cp = first & 0x0fU; continuation = 2; }
        else if ((first & 0xf8U) == 0xf0U) { cp = first & 0x07U; continuation = 3; }
        for (std::size_t n = 0; n < continuation && i < text.size(); ++n)
            cp = (cp << 6) | (static_cast<unsigned char>(text[i++]) & 0x3fU);
        result.push_back(cp);
    }
    return result;
}

std::string encode_utf8(std::uint32_t cp) {
    std::string out;
    if (cp <= 0x7f) out.push_back(static_cast<char>(cp));
    else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
    return out;
}

std::string arpa_to_ipa(std::string phone) {
    char stress = 0;
    if (!phone.empty() && std::isdigit(static_cast<unsigned char>(phone.back()))) {
        stress = phone.back(); phone.pop_back();
    }
    if (phone == "AH") return (stress == '1' ? "ˈ" : stress == '2' ? "ˌ" : "") +
                               std::string(stress == '0' ? "ə" : "ʌ");
    if (phone == "ER") return (stress == '1' ? "ˈ" : stress == '2' ? "ˌ" : "") +
                               std::string(stress == '0' ? "ɚ" : "ɜː");
    static const std::unordered_map<std::string, std::string> map{
        {"AA", "ɑ"}, {"AE", "æ"},
        {"AO", "ɔ"}, {"AW", "aʊ"}, {"AY", "aɪ"}, {"EH", "ɛ"},
        {"EY", "eɪ"}, {"IH", "ɪ"},
        {"IY", "i"}, {"OW", "oʊ"}, {"OY", "ɔɪ"}, {"UH", "ʊ"}, {"UW", "u"},
        {"B", "b"}, {"CH", "tʃ"}, {"D", "d"}, {"DH", "ð"}, {"F", "f"},
        {"G", "ɡ"}, {"HH", "h"}, {"JH", "dʒ"}, {"K", "k"}, {"L", "l"},
        {"M", "m"}, {"N", "n"}, {"NG", "ŋ"}, {"P", "p"}, {"R", "ɹ"},
        {"S", "s"}, {"SH", "ʃ"}, {"T", "t"}, {"TH", "θ"}, {"V", "v"},
        {"W", "w"}, {"Y", "j"}, {"Z", "z"}, {"ZH", "ʒ"}};
    const auto found = map.find(phone);
    if (found == map.end()) return {};
    return (stress == '1' ? "ˈ" : stress == '2' ? "ˌ" : "") + found->second;
}

}  // namespace

CmuPhonemizer::CmuPhonemizer(const std::filesystem::path& dictionary_path,
                             const std::filesystem::path& token_map_path) {
    std::ifstream dictionary(dictionary_path);
    if (!dictionary) throw std::runtime_error("cannot open CMUdict: " + dictionary_path.string());
    for (std::string line; std::getline(dictionary, line);) {
        if (line.empty() || line[0] == ';') continue;
        std::istringstream row(line);
        std::string word; row >> word;
        const auto variant = word.find('(');
        if (variant != std::string::npos) word.resize(variant);
        if (dictionary_.contains(word)) continue;
        std::vector<std::string> phones;
        for (std::string phone; row >> phone;) phones.push_back(std::move(phone));
        if (!phones.empty()) dictionary_[std::move(word)] = std::move(phones);
    }
    std::ifstream tokens(token_map_path);
    if (!tokens) throw std::runtime_error("cannot open model token map: " + token_map_path.string());
    for (std::string line; std::getline(tokens, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "# frontend=arpabet") { arpabet_frontend_ = true; continue; }
        if (line.empty() || line[0] == '#') continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::vector<std::int64_t> ids;
        std::istringstream values(line.substr(tab + 1));
        for (std::string id; std::getline(values, id, ',');) ids.push_back(std::stoll(id));
        if (arpabet_frontend_) arpa_token_map_[line.substr(0, tab)] = std::move(ids);
        else {
            std::uint32_t cp{};
            const auto parsed = std::from_chars(line.data(), line.data() + tab, cp, 16);
            if (parsed.ec != std::errc{}) continue;
            token_map_[cp] = std::move(ids);
        }
    }
}

PhonemizationResult CmuPhonemizer::phonemize(std::string_view text) const {
    PhonemizationResult result;
    auto append_symbol = [&](std::uint32_t cp, bool pad, ProsodyTokenKind kind = ProsodyTokenKind::Unknown) {
        const auto found = token_map_.find(cp);
        if (found == token_map_.end()) { ++result.missing_model_symbols; return; }
        result.token_ids.insert(result.token_ids.end(), found->second.begin(), found->second.end());
        result.token_kinds.insert(result.token_kinds.end(), found->second.size(), kind);
        if (pad) {
            const auto padding = token_map_.find('_');
            if (padding != token_map_.end()) {
                result.token_ids.insert(result.token_ids.end(), padding->second.begin(), padding->second.end());
                result.token_kinds.insert(result.token_kinds.end(), padding->second.size(), ProsodyTokenKind::Unknown);
            }
        }
        result.ipa += encode_utf8(cp);
    };
    auto append_phone = [&](const std::string& phone, ProsodyTokenKind kind) {
        const auto found = arpa_token_map_.find(phone);
        if (found == arpa_token_map_.end()) { ++result.missing_model_symbols; return; }
        result.token_ids.insert(result.token_ids.end(), found->second.begin(), found->second.end());
        result.token_kinds.insert(result.token_kinds.end(), found->second.size(), kind);
        result.ipa += phone + " ";
    };
    if (!arpabet_frontend_) append_symbol('^', true);
    std::string word;
    auto flush_word = [&] {
        if (word.empty()) return;
        ++result.words;
        auto found = dictionary_.find(word);
        std::vector<std::string> fallback;
        const std::vector<std::string>* phones = nullptr;
        if (found != dictionary_.end()) { phones = &found->second; ++result.dictionary_hits; }
        else {
            // Spell OOV words using CMUdict's single-letter pronunciations.
            for (char letter : word) {
                const auto spelled = dictionary_.find(std::string(1, letter));
                if (spelled != dictionary_.end())
                    fallback.insert(fallback.end(), spelled->second.begin(), spelled->second.end());
            }
            phones = &fallback; ++result.fallback_words;
        }
        for (const auto& phone : *phones) {
            const auto ipa = arpa_to_ipa(phone);
            std::string arpa = phone;
            const bool stressed = !arpa.empty() && (arpa.back() == '1' || arpa.back() == '2');
            if (!arpa.empty() && std::isdigit(static_cast<unsigned char>(arpa.back()))) arpa.pop_back();
            const std::string vowels = " AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW ";
            const std::string unvoiced = " CH F HH K P S SH T TH ";
            auto kind = ProsodyTokenKind::Unknown;
            if (vowels.find(" " + arpa + " ") != std::string::npos)
                kind = stressed ? ProsodyTokenKind::StressedVowel : ProsodyTokenKind::Vowel;
            else if (unvoiced.find(" " + arpa + " ") != std::string::npos)
                kind = ProsodyTokenKind::UnvoicedConsonant;
            if (arpabet_frontend_) append_phone(phone, kind);
            else for (auto cp : decode_utf8(ipa))
                append_symbol(cp, true, cp == 0x02c8 || cp == 0x02cc || cp == 0x02d0 ? ProsodyTokenKind::Unknown : kind);
        }
        word.clear();
    };
    bool pending_space = false;
    for (unsigned char raw : text) {
        const char c = static_cast<char>(std::tolower(raw));
        if (std::isalnum(raw) || (c == '\'' && !word.empty())) { word += c; pending_space = false; }
        else {
            flush_word();
            if (std::string_view(".,!?;:").find(c) != std::string_view::npos) {
                if (arpabet_frontend_) append_phone("sp", ProsodyTokenKind::Boundary);
                else append_symbol(c, true, ProsodyTokenKind::Boundary);
            }
            if (std::isspace(raw)) pending_space = true;
            if (pending_space && result.words > 0) {
                if (!arpabet_frontend_) append_symbol(' ', true);
                pending_space = false;
            }
        }
    }
    flush_word();
    if (!arpabet_frontend_) append_symbol('$', false, ProsodyTokenKind::Boundary);
    else {
        // Publisher's merged English frontend omits trailing silence and BOS/EOS/padding.
        const auto silence = arpa_token_map_.find("sp");
        if (silence != arpa_token_map_.end() && !silence->second.empty() &&
            result.token_ids.size() >= silence->second.size() &&
            std::equal(silence->second.rbegin(), silence->second.rend(), result.token_ids.rbegin())) {
            result.token_ids.resize(result.token_ids.size() - silence->second.size());
            result.token_kinds.resize(result.token_ids.size());
        }
    }
    return result;
}

}  // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\models\piper_onnx\piper_voice.cpp ===
#include <piper_onnx/piper_voice.hpp>

#include <piper_onnx/phonemizer.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if VA_HAS_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace vocal {
namespace {

[[maybe_unused]] std::vector<std::string> chunks(std::string_view text, std::size_t maximum) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (begin < text.size()) {
        std::size_t end = std::min(text.size(), begin + maximum);
        if (end < text.size()) {
            const auto punctuation = text.find_last_of(".!?;:", end);
            const auto space = text.find_last_of(' ', end);
            const auto split = punctuation != std::string_view::npos && punctuation >= begin ? punctuation + 1 : space;
            if (split != std::string_view::npos && split > begin) end = split;
        }
        auto part = std::string(text.substr(begin, end - begin));
        if (!part.empty()) result.push_back(std::move(part));
        begin = end;
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    }
    if (result.empty()) result.emplace_back(" ");
    return result;
}

std::unordered_map<std::string, std::string> properties(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open voice properties: " + path.string());
    std::unordered_map<std::string, std::string> values;
    for (std::string line; std::getline(in, line);) {
        const auto equals = line.find('=');
        if (equals != std::string::npos) values[line.substr(0, equals)] = line.substr(equals + 1);
    }
    return values;
}

}  // namespace

struct PiperVoiceSynthesizer::Impl {
    PiperVoiceConfig config;
    CmuPhonemizer phonemizer;
    PiperDiagnostics diagnostics;
    int sample_rate{22'050};
    int num_speakers{1};
#if VA_HAS_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "vocal-piper"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
#endif

    explicit Impl(PiperVoiceConfig supplied)
        : config(std::move(supplied)),
          phonemizer(config.dictionary_path, config.token_map_path) {
        const auto values = properties(config.properties_path);
        if (values.contains("sample_rate")) sample_rate = std::stoi(values.at("sample_rate"));
        if (values.contains("num_speakers")) num_speakers = std::stoi(values.at("num_speakers"));
        if (config.speaker_id < 0 || config.speaker_id >= num_speakers)
            throw std::out_of_range("speaker ID is outside the voice model's range");
#if VA_HAS_ONNX_RUNTIME
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.EnableCpuMemArena();
        options.EnableMemPattern();
        const int threads = config.intra_op_threads > 0 ? config.intra_op_threads :
            static_cast<int>(std::max(1U, std::thread::hardware_concurrency() / 2));
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        session = Ort::Session(environment, config.model_path.c_str(), options);
#else
        throw std::runtime_error("Piper/VITS requires an ONNX Runtime-enabled build");
#endif
    }
};

PiperVoiceSynthesizer::PiperVoiceSynthesizer(PiperVoiceConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
PiperVoiceSynthesizer::~PiperVoiceSynthesizer() = default;
PiperVoiceSynthesizer::PiperVoiceSynthesizer(PiperVoiceSynthesizer&&) noexcept = default;
PiperVoiceSynthesizer& PiperVoiceSynthesizer::operator=(PiperVoiceSynthesizer&&) noexcept = default;

bool PiperVoiceSynthesizer::compiled_with_runtime() noexcept {
#if VA_HAS_ONNX_RUNTIME
    return true;
#else
    return false;
#endif
}

const PiperDiagnostics& PiperVoiceSynthesizer::diagnostics() const noexcept { return impl_->diagnostics; }

Waveform PiperVoiceSynthesizer::synthesize(std::string_view text) {
#if VA_HAS_ONNX_RUNTIME
    impl_->diagnostics = {};
    impl_->diagnostics.speaker_id = impl_->config.speaker_id;
    Waveform waveform{impl_->sample_rate, {}};
    const auto parts = chunks(text, impl_->config.maximum_chunk_characters);
    for (std::size_t part_index = 0; part_index < parts.size(); ++part_index) {
        auto phonemes = impl_->phonemizer.phonemize(parts[part_index]);
        impl_->diagnostics.words += phonemes.words;
        impl_->diagnostics.dictionary_hits += phonemes.dictionary_hits;
        impl_->diagnostics.fallback_words += phonemes.fallback_words;
        impl_->diagnostics.missing_model_symbols += phonemes.missing_model_symbols;
        impl_->diagnostics.phoneme_tokens += phonemes.token_ids.size();
        std::vector<std::int64_t> lengths{static_cast<std::int64_t>(phonemes.token_ids.size())};
        std::vector<float> scales{impl_->config.noise_scale, impl_->config.length_scale, impl_->config.noise_w};
        std::vector<std::int64_t> speaker{impl_->config.speaker_id};
        const std::array<std::int64_t, 2> token_shape{1, lengths[0]};
        const std::array<std::int64_t, 1> vector_shape{1};
        const std::array<std::int64_t, 1> scales_shape{3};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory,
            phonemes.token_ids.data(), phonemes.token_ids.size(),
            token_shape.data(), token_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, lengths.data(), lengths.size(),
            vector_shape.data(), vector_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(memory, scales.data(), scales.size(),
            scales_shape.data(), scales_shape.size()));
        std::vector<const char*> input_names{"input", "input_lengths", "scales"};
        if (impl_->num_speakers > 1) {
            inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(memory, speaker.data(), speaker.size(),
                vector_shape.data(), vector_shape.size()));
            input_names.push_back("sid");
        }
        const char* output_names[] = {"output"};
        const auto started = std::chrono::steady_clock::now();
        auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                                          output_names, 1);
        const auto stopped = std::chrono::steady_clock::now();
        impl_->diagnostics.inference_ms +=
            std::chrono::duration<double, std::milli>(stopped - started).count();
        const auto count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const float* samples = outputs[0].GetTensorData<float>();
        waveform.samples.insert(waveform.samples.end(), samples, samples + count);
        if (part_index + 1 < parts.size()) {
            waveform.samples.insert(waveform.samples.end(),
                static_cast<std::size_t>(impl_->sample_rate * impl_->config.sentence_silence_seconds), 0.0F);
        }
        ++impl_->diagnostics.chunks;
    }
    impl_->diagnostics.audio_seconds = static_cast<double>(waveform.samples.size()) / waveform.sample_rate_hz;
    impl_->diagnostics.real_time_factor = impl_->diagnostics.audio_seconds > 0.0 ?
        (impl_->diagnostics.inference_ms / 1000.0) / impl_->diagnostics.audio_seconds : 0.0;
    return waveform;
#else
    (void)text;
    throw std::runtime_error("Piper/VITS requires an ONNX Runtime-enabled build");
#endif
}

}  // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\src\audio.cpp ===
#include "vocal/audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace vocal {
namespace {

void u16(std::ostream& out, std::uint16_t value) {
    const char bytes[] = {static_cast<char>(value & 0xff), static_cast<char>((value >> 8) & 0xff)};
    out.write(bytes, 2);
}

void u32(std::ostream& out, std::uint32_t value) {
    const char bytes[] = {static_cast<char>(value & 0xff), static_cast<char>((value >> 8) & 0xff),
                          static_cast<char>((value >> 16) & 0xff), static_cast<char>((value >> 24) & 0xff)};
    out.write(bytes, 4);
}

}  // namespace

void normalize_peak(Waveform& audio, float peak) {
    float maximum = 0.0F;
    for (float sample : audio.samples) maximum = std::max(maximum, std::abs(sample));
    if (maximum <= 0.0F) return;
    const float scale = peak / maximum;
    for (float& sample : audio.samples) sample *= scale;
}

Waveform resample_waveform(const Waveform& audio, int sample_rate_hz) {
    if (audio.sample_rate_hz <= 0 || sample_rate_hz <= 0)
        throw std::invalid_argument("resampling requires positive sample rates");
    if (audio.sample_rate_hz == sample_rate_hz) return audio;
    const double ratio = static_cast<double>(sample_rate_hz) / audio.sample_rate_hz;
    const double size = audio.samples.size() * ratio;
    if (size > 0x7fffffffU) throw std::invalid_argument("resampled waveform exceeds sample limit");
    Waveform output{sample_rate_hz, std::vector<float>(static_cast<std::size_t>(std::llround(size)))};
    for (std::size_t i = 0; i < output.samples.size(); ++i) {
        const double position = static_cast<double>(i) / ratio;
        const auto left = std::min(static_cast<std::size_t>(position), audio.samples.size() - 1);
        const auto right = std::min(left + 1, audio.samples.size() - 1);
        const float fraction = static_cast<float>(position - left);
        output.samples[i] = audio.samples[left] * (1 - fraction) + audio.samples[right] * fraction;
    }
    return output;
}

void write_wav_pcm16(const std::filesystem::path& path, const Waveform& audio) {
    if (audio.sample_rate_hz <= 0 || audio.samples.size() > 0x7fffffffU) {
        throw std::invalid_argument("invalid audio for WAV output");
    }
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create WAV file: " + path.string());
    const auto data_bytes = static_cast<std::uint32_t>(audio.samples.size() * 2);
    out.write("RIFF", 4); u32(out, 36 + data_bytes); out.write("WAVE", 4);
    out.write("fmt ", 4); u32(out, 16); u16(out, 1); u16(out, 1);
    u32(out, static_cast<std::uint32_t>(audio.sample_rate_hz));
    u32(out, static_cast<std::uint32_t>(audio.sample_rate_hz * 2));
    u16(out, 2); u16(out, 16); out.write("data", 4); u32(out, data_bytes);
    for (float sample : audio.samples) {
        const auto clipped = std::clamp(sample, -1.0F, 1.0F);
        const auto pcm = static_cast<std::int16_t>(std::lrint(clipped * 32767.0F));
        u16(out, static_cast<std::uint16_t>(pcm));
    }
}

}  // namespace vocal


// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\src\dataset.cpp ===
#include "vocal/dataset.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vocal {

LibriTtsDataset::LibriTtsDataset(std::filesystem::path root) : root_(std::move(root)) {}

std::vector<Utterance> LibriTtsDataset::scan() const {
    if (!std::filesystem::exists(root_)) {
        throw std::runtime_error("dataset root does not exist: " + root_.string());
    }

    std::vector<Utterance> items;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".wav") {
            continue;
        }

        const auto stem = entry.path().stem().string();
        auto text_path = entry.path().parent_path() / (stem + ".normalized.txt");
        if (!std::filesystem::exists(text_path)) {
            continue;
        }

        std::ifstream text_file(text_path);
        std::ostringstream text;
        text << text_file.rdbuf();
        auto normalized = text.str();
        while (!normalized.empty() && (normalized.back() == '\n' || normalized.back() == '\r')) {
            normalized.pop_back();
        }

        const auto chapter_dir = entry.path().parent_path();
        const auto speaker_dir = chapter_dir.parent_path();
        items.push_back({stem, speaker_dir.filename().string(), chapter_dir.filename().string(),
                         std::move(normalized), entry.path()});
    }
    return items;
}

}  // namespace vocal


// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\src\evaluator.cpp ===
#include "vocal/evaluator.hpp"
#include "vocal/acoustic_model.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vocal {
namespace {

std::vector<double> parse_line(const std::string& line) {
    std::string normalized = line;
    for (auto& c : normalized) if (c == ',') c = ' ';
    std::istringstream input(normalized);
    std::vector<double> values;
    for (double value; input >> value;) values.push_back(value);
    return values;
}

}  // namespace

metrics::Frames Evaluator::load_matrix(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open feature matrix: " + path.string());
    metrics::Frames frames;
    for (std::string line; std::getline(input, line);) {
        auto frame = parse_line(line);
        if (!frame.empty()) {
            if (!frames.empty() && frame.size() != frames.front().size()) {
                throw std::runtime_error("inconsistent feature width: " + path.string());
            }
            frames.push_back(std::move(frame));
        }
    }
    return frames;
}

std::vector<double> Evaluator::load_vector(const std::filesystem::path& path) {
    auto rows = load_matrix(path);
    std::vector<double> result;
    for (const auto& row : rows) result.insert(result.end(), row.begin(), row.end());
    return result;
}

EvaluationResult Evaluator::evaluate(const EvaluationCase& item) const {
    EvaluationResult result{item.utterance_id,
                            metrics::mcd_db(load_matrix(item.reference_mcep),
                                            load_matrix(item.synthesized_mcep)),
                            std::nullopt,
                            std::nullopt};
    if (item.reference_f0 && item.synthesized_f0) {
        result.pitch = metrics::pitch(load_vector(*item.reference_f0),
                                      load_vector(*item.synthesized_f0));
    }
    if (item.reference_text && item.hypothesis_text) {
        result.wer = metrics::word_error_rate(*item.reference_text, *item.hypothesis_text);
    }
    return result;
}

std::span<const float> MelSpectrogram::frame(std::size_t index) const {
    if (index >= frames || log_mel.size() != frames * bins) {
        throw std::out_of_range("invalid mel-spectrogram frame");
    }
    return {log_mel.data() + index * bins, bins};
}

}  // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\src\features.cpp ===
#include "vocal/features.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vocal {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::uint16_t read_u16(std::istream& in) {
    unsigned char b[2]{};
    if (!in.read(reinterpret_cast<char*>(b), 2)) throw std::runtime_error("truncated WAV");
    return static_cast<std::uint16_t>(b[0] | (b[1] << 8));
}

std::uint32_t read_u32(std::istream& in) {
    unsigned char b[4]{};
    if (!in.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("truncated WAV");
    return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
           (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

double hz_to_mel(double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }
double mel_to_hz(double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); }

void fft(std::vector<std::complex<double>>& values) {
    const std::size_t n = values.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(values[i], values[j]);
    }
    for (std::size_t length = 2; length <= n; length <<= 1) {
        const auto angle = -2.0 * kPi / static_cast<double>(length);
        const std::complex<double> root(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += length) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t j = 0; j < length / 2; ++j) {
                const auto even = values[i + j];
                const auto odd = values[i + j + length / 2] * w;
                values[i + j] = even + odd;
                values[i + j + length / 2] = even - odd;
                w *= root;
            }
        }
    }
}

bool power_of_two(std::size_t value) { return value && !(value & (value - 1)); }

}  // namespace

Waveform load_wav_mono(const std::filesystem::path& path, int target_sample_rate_hz) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open WAV: " + path.string());
    char riff[4]{}, wave[4]{};
    in.read(riff, 4); (void)read_u32(in); in.read(wave, 4);
    if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("not a RIFF/WAVE file: " + path.string());
    }
    std::uint16_t format = 0, channels = 0, bits = 0;
    std::uint32_t sample_rate = 0;
    std::vector<unsigned char> data;
    while (in && (format == 0 || data.empty())) {
        char id[4]{};
        if (!in.read(id, 4)) break;
        const auto size = read_u32(in);
        const auto chunk_start = in.tellg();
        if (std::string(id, 4) == "fmt ") {
            format = read_u16(in); channels = read_u16(in); sample_rate = read_u32(in);
            (void)read_u32(in); (void)read_u16(in); bits = read_u16(in);
        } else if (std::string(id, 4) == "data") {
            data.resize(size);
            if (!in.read(reinterpret_cast<char*>(data.data()), size)) throw std::runtime_error("truncated WAV data");
        }
        in.clear();
        in.seekg(chunk_start + static_cast<std::streamoff>(size + (size & 1U)));
    }
    if (channels == 0 || sample_rate == 0 || data.empty() ||
        !((format == 1 && bits == 16) || (format == 3 && bits == 32))) {
        throw std::runtime_error("WAV must be PCM16 or float32: " + path.string());
    }
    const std::size_t bytes_per_sample = bits / 8;
    const std::size_t frames = data.size() / (bytes_per_sample * channels);
    std::vector<float> mono(frames);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        double sum = 0.0;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const auto offset = (frame * channels + channel) * bytes_per_sample;
            if (format == 1) {
                const auto raw = static_cast<std::uint16_t>(data[offset] | (data[offset + 1] << 8));
                sum += static_cast<std::int16_t>(raw) / 32768.0;
            } else {
                float sample{};
                std::memcpy(&sample, data.data() + offset, sizeof(float));
                sum += sample;
            }
        }
        mono[frame] = static_cast<float>(sum / channels);
    }
    if (static_cast<int>(sample_rate) == target_sample_rate_hz) return {target_sample_rate_hz, std::move(mono)};
    if (target_sample_rate_hz <= 0) throw std::invalid_argument("target sample rate must be positive");
    const double ratio = static_cast<double>(target_sample_rate_hz) / sample_rate;
    std::vector<float> resampled(static_cast<std::size_t>(std::floor(mono.size() * ratio)));
    for (std::size_t i = 0; i < resampled.size(); ++i) {
        const double source = static_cast<double>(i) / ratio;
        const auto left = std::min(static_cast<std::size_t>(source), mono.size() - 1);
        const auto right = std::min(left + 1, mono.size() - 1);
        const double fraction = source - left;
        resampled[i] = static_cast<float>(mono[left] * (1.0 - fraction) + mono[right] * fraction);
    }
    return {target_sample_rate_hz, std::move(resampled)};
}

AcousticFeatures extract_features(const Waveform& audio, const FeatureConfig& config) {
    if (audio.sample_rate_hz != config.sample_rate_hz || !power_of_two(config.fft_size) ||
        config.hop_samples == 0 || config.mel_bins == 0 || config.mcep_coefficients == 0) {
        throw std::invalid_argument("invalid feature configuration or non-canonical sample rate");
    }
    const auto padded_size = std::max(audio.samples.size(), config.fft_size);
    const auto frame_count = 1 + (padded_size - config.fft_size + config.hop_samples - 1) /
                                     config.hop_samples;
    const auto spectrum_bins = config.fft_size / 2 + 1;
    const double maximum_hz = std::min(config.maximum_hz, config.sample_rate_hz / 2.0);
    std::vector<std::size_t> mel_edges(config.mel_bins + 2);
    const double low_mel = hz_to_mel(config.minimum_hz), high_mel = hz_to_mel(maximum_hz);
    for (std::size_t i = 0; i < mel_edges.size(); ++i) {
        const double hz = mel_to_hz(low_mel + (high_mel - low_mel) * i / (mel_edges.size() - 1));
        mel_edges[i] = std::min(spectrum_bins - 1,
            static_cast<std::size_t>(std::floor((config.fft_size + 1) * hz / config.sample_rate_hz)));
    }

    AcousticFeatures result;
    result.log_mel.reserve(frame_count); result.mcep.reserve(frame_count);
    std::vector<std::complex<double>> buffer(config.fft_size);
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const auto offset = frame * config.hop_samples;
        for (std::size_t i = 0; i < config.fft_size; ++i) {
            const double sample = offset + i < audio.samples.size() ? audio.samples[offset + i] : 0.0;
            const double window = .5 - .5 * std::cos(2.0 * kPi * i / (config.fft_size - 1));
            buffer[i] = sample * window;
        }
        fft(buffer);
        std::vector<double> power(spectrum_bins);
        for (std::size_t bin = 0; bin < spectrum_bins; ++bin) power[bin] = std::norm(buffer[bin]);
        std::vector<double> mel(config.mel_bins);
        for (std::size_t band = 0; band < config.mel_bins; ++band) {
            double energy = 0.0;
            const auto left = mel_edges[band], center = mel_edges[band + 1], right = mel_edges[band + 2];
            for (std::size_t bin = left; bin < center; ++bin)
                energy += power[bin] * (bin - left) / static_cast<double>(std::max<std::size_t>(1, center - left));
            for (std::size_t bin = center; bin <= right; ++bin)
                energy += power[bin] * (right - bin) / static_cast<double>(std::max<std::size_t>(1, right - center));
            mel[band] = std::log(std::max(config.log_floor, energy));
        }
        std::vector<double> mcep(config.mcep_coefficients);
        for (std::size_t coefficient = 0; coefficient < mcep.size(); ++coefficient) {
            for (std::size_t band = 0; band < mel.size(); ++band) {
                mcep[coefficient] += mel[band] * std::cos(kPi * coefficient *
                    (static_cast<double>(band) + .5) / mel.size());
            }
            mcep[coefficient] *= std::sqrt(2.0 / mel.size());
            if (coefficient == 0) mcep[coefficient] /= std::sqrt(2.0);
        }
        result.log_mel.push_back(std::move(mel));
        result.mcep.push_back(std::move(mcep));
    }
    return result;
}

void write_feature_matrix(const std::filesystem::path& path, const metrics::Frames& matrix) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create feature matrix: " + path.string());
    out << std::setprecision(10);
    for (const auto& row : matrix) {
        for (std::size_t i = 0; i < row.size(); ++i) out << (i ? "," : "") << row[i];
        out << '\n';
    }
}

}  // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\src\metrics.cpp ===
#include "vocal/metrics.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>

namespace vocal::metrics {
namespace {

// 10 / ln(10) * sqrt(2). A literal is used because MSVC 19.44 does not yet
// accept std::log/std::sqrt in constant evaluation in this language mode.
constexpr double kMcdScale = 6.141851463713754;

double cepstral_distance(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) {
        throw std::invalid_argument("MCEP frames must have the same non-zero dimension");
    }
    // Conventionally c0 is energy and is excluded from MCD.
    const std::size_t first = a.size() > 1 ? 1 : 0;
    double squared = 0.0;
    for (std::size_t i = first; i < a.size(); ++i) {
        const double delta = a[i] - b[i];
        squared += delta * delta;
    }
    return kMcdScale * std::sqrt(squared);
}

std::vector<std::string> words(std::string_view text) {
    std::string normalized;
    normalized.reserve(text.size());
    for (unsigned char c : text) {
        normalized.push_back(std::isalnum(c) || c == '\'' ? static_cast<char>(std::tolower(c)) : ' ');
    }
    std::istringstream stream(normalized);
    std::vector<std::string> result;
    for (std::string word; stream >> word;) result.push_back(std::move(word));
    return result;
}

}  // namespace

double mcd_db(const Frames& reference, const Frames& synthesized, bool use_dtw) {
    if (reference.empty() || synthesized.empty()) {
        throw std::invalid_argument("MCD requires non-empty feature sequences");
    }
    if (!use_dtw && reference.size() != synthesized.size()) {
        throw std::invalid_argument("unaligned MCD inputs require DTW");
    }
    if (!use_dtw) {
        double sum = 0.0;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            sum += cepstral_distance(reference[i], synthesized[i]);
        }
        return sum / static_cast<double>(reference.size());
    }

    const std::size_t n = reference.size(), m = synthesized.size();
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> previous(m + 1, inf), current(m + 1, inf);
    std::vector<std::size_t> previous_steps(m + 1), current_steps(m + 1);
    previous[0] = 0.0;
    for (std::size_t i = 1; i <= n; ++i) {
        current.assign(m + 1, inf);
        current_steps.assign(m + 1, 0);
        for (std::size_t j = 1; j <= m; ++j) {
            double best = previous[j - 1];
            std::size_t steps = previous_steps[j - 1];
            if (previous[j] < best) { best = previous[j]; steps = previous_steps[j]; }
            if (current[j - 1] < best) { best = current[j - 1]; steps = current_steps[j - 1]; }
            current[j] = best + cepstral_distance(reference[i - 1], synthesized[j - 1]);
            current_steps[j] = steps + 1;
        }
        std::swap(previous, current);
        std::swap(previous_steps, current_steps);
    }
    return previous[m] / static_cast<double>(previous_steps[m]);
}

PitchScore pitch(std::span<const double> reference_hz, std::span<const double> synthesized_hz) {
    if (reference_hz.empty() || reference_hz.size() != synthesized_hz.size()) {
        throw std::invalid_argument("pitch contours must have equal non-zero length");
    }
    double squared = 0.0;
    std::size_t vuv_errors = 0, voiced = 0;
    for (std::size_t i = 0; i < reference_hz.size(); ++i) {
        const bool ref_voiced = reference_hz[i] > 0.0;
        const bool syn_voiced = synthesized_hz[i] > 0.0;
        vuv_errors += ref_voiced != syn_voiced;
        if (ref_voiced && syn_voiced) {
            const auto delta = reference_hz[i] - synthesized_hz[i];
            squared += delta * delta;
            ++voiced;
        }
    }
    return {voiced ? std::sqrt(squared / static_cast<double>(voiced)) : 0.0,
            static_cast<double>(vuv_errors) / static_cast<double>(reference_hz.size()), voiced};
}

double word_error_rate(std::string_view reference, std::string_view hypothesis) {
    const auto ref = words(reference), hyp = words(hypothesis);
    if (ref.empty()) return hyp.empty() ? 0.0 : 1.0;
    std::vector<std::size_t> previous(hyp.size() + 1), current(hyp.size() + 1);
    for (std::size_t j = 0; j <= hyp.size(); ++j) previous[j] = j;
    for (std::size_t i = 1; i <= ref.size(); ++i) {
        current[0] = i;
        for (std::size_t j = 1; j <= hyp.size(); ++j) {
            current[j] = std::min({previous[j] + 1, current[j - 1] + 1,
                                   previous[j - 1] + (ref[i - 1] == hyp[j - 1] ? 0U : 1U)});
        }
        std::swap(previous, current);
    }
    return static_cast<double>(previous.back()) / static_cast<double>(ref.size());
}

}  // namespace vocal::metrics

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\src\reporting.cpp ===
#include "vocal/reporting.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace vocal {
namespace {

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') { field += '"'; ++i; }
            else quoted = !quoted;
        } else if (c == ',' && !quoted) { fields.push_back(field); field.clear(); }
        else field += c;
    }
    fields.push_back(field);
    return fields;
}

std::optional<double> number(const std::string& value) {
    if (value.empty()) return std::nullopt;
    double result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error("invalid numeric score: " + value);
    }
    return result;
}

ConfidenceInterval bootstrap(std::vector<double> values, std::size_t samples, std::uint64_t seed) {
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    ConfidenceInterval result{values.size(), sum / values.size(), 0.0, 0.0};
    if (values.size() == 1 || samples == 0) {
        result.lower_95 = result.upper_95 = result.mean;
        return result;
    }
    std::mt19937_64 random(seed);
    std::uniform_int_distribution<std::size_t> choose(0, values.size() - 1);
    std::vector<double> means(samples);
    for (double& mean : means) {
        double sample_sum = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) sample_sum += values[choose(random)];
        mean = sample_sum / values.size();
    }
    std::sort(means.begin(), means.end());
    result.lower_95 = means[static_cast<std::size_t>(.025 * (means.size() - 1))];
    result.upper_95 = means[static_cast<std::size_t>(.975 * (means.size() - 1))];
    return result;
}

std::optional<ConfidenceInterval> metric(const std::vector<const ScoredUtterance*>& rows,
    const std::function<std::optional<double>(const ScoredUtterance&)>& getter,
    std::size_t samples, std::uint64_t seed) {
    std::vector<double> values;
    for (const auto* row : rows) if (auto value = getter(*row)) values.push_back(*value);
    if (values.empty()) return std::nullopt;
    return bootstrap(std::move(values), samples, seed);
}

std::string escape_json(const std::string& value) {
    std::string result;
    for (char c : value) {
        if (c == '"' || c == '\\') result += '\\';
        result += c;
    }
    return result;
}

void csv_metric(std::ostream& out, const std::optional<ConfidenceInterval>& value) {
    if (value) out << value->mean << ',' << value->lower_95 << ',' << value->upper_95 << ',' << value->count;
    else out << ",,,0";
}

void json_metric(std::ostream& out, std::string_view name,
                 const std::optional<ConfidenceInterval>& value, bool comma) {
    out << "      \"" << name << "\": ";
    if (!value) out << "null";
    else out << "{\"n\": " << value->count << ", \"mean\": " << value->mean
             << ", \"lower_95\": " << value->lower_95 << ", \"upper_95\": " << value->upper_95 << '}';
    out << (comma ? ",\n" : "\n");
}

}  // namespace

std::vector<ScoredUtterance> load_score_csv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open score CSV: " + path.string());
    std::string header;
    if (!std::getline(in, header)) return {};
    const auto columns = split_csv(header);
    std::map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < columns.size(); ++i) index[columns[i]] = i;
    if (!index.contains("utterance_id") || !index.contains("speaker_id"))
        throw std::runtime_error("score CSV requires utterance_id and speaker_id columns");
    auto field = [&](const std::vector<std::string>& row, std::string_view name) -> std::string {
        const auto found = index.find(std::string(name));
        return found == index.end() || found->second >= row.size() ? "" : row[found->second];
    };
    std::vector<ScoredUtterance> scores;
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        const auto row = split_csv(line);
        scores.push_back({field(row, "utterance_id"), field(row, "speaker_id"),
                          number(field(row, "mcd_db")), number(field(row, "f0_rmse_hz")),
                          number(field(row, "vuv_error")), number(field(row, "wer"))});
    }
    return scores;
}

std::vector<SpeakerSummary> summarize_by_speaker(const std::vector<ScoredUtterance>& scores,
                                                  std::size_t samples, std::uint64_t seed) {
    std::map<std::string, std::vector<const ScoredUtterance*>> groups;
    for (const auto& score : scores) { groups[score.speaker_id].push_back(&score); groups["__all__"].push_back(&score); }
    std::vector<SpeakerSummary> summaries;
    for (const auto& [speaker, rows] : groups) {
        auto make = [&](auto member, std::uint64_t salt) {
            return metric(rows, [member](const ScoredUtterance& row) { return row.*member; }, samples, seed ^ salt);
        };
        summaries.push_back({speaker, rows.size(), make(&ScoredUtterance::mcd_db, 1),
            make(&ScoredUtterance::f0_rmse_hz, 2), make(&ScoredUtterance::vuv_error, 3),
            make(&ScoredUtterance::wer, 4)});
    }
    return summaries;
}

void write_summaries_csv(const std::filesystem::path& path, const std::vector<SpeakerSummary>& summaries) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create summary CSV: " + path.string());
    out << "speaker_id,utterances,mcd_mean,mcd_lower95,mcd_upper95,mcd_n,"
           "f0_mean,f0_lower95,f0_upper95,f0_n,vuv_mean,vuv_lower95,vuv_upper95,vuv_n,"
           "wer_mean,wer_lower95,wer_upper95,wer_n\n" << std::setprecision(10);
    for (const auto& row : summaries) {
        out << row.speaker_id << ',' << row.utterances << ','; csv_metric(out, row.mcd_db); out << ',';
        csv_metric(out, row.f0_rmse_hz); out << ','; csv_metric(out, row.vuv_error); out << ',';
        csv_metric(out, row.wer); out << '\n';
    }
}

void write_summaries_json(const std::filesystem::path& path, const std::vector<SpeakerSummary>& summaries,
                          std::size_t samples, std::uint64_t seed) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create summary JSON: " + path.string());
    out << std::setprecision(10) << "{\n  \"bootstrap_samples\": " << samples
        << ",\n  \"seed\": " << seed << ",\n  \"speakers\": [\n";
    for (std::size_t i = 0; i < summaries.size(); ++i) {
        const auto& row = summaries[i];
        out << "    {\n      \"speaker_id\": \"" << escape_json(row.speaker_id)
            << "\",\n      \"utterances\": " << row.utterances << ",\n";
        json_metric(out, "mcd_db", row.mcd_db, true); json_metric(out, "f0_rmse_hz", row.f0_rmse_hz, true);
        json_metric(out, "vuv_error", row.vuv_error, true); json_metric(out, "wer", row.wer, false);
        out << "    }" << (i + 1 == summaries.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
}

}  // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\src\study.cpp ===
#include "vocal/study.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace vocal {
namespace {

struct Stimulus { std::string utterance; std::string system; std::filesystem::path path; };

std::string json_escape(const std::string& value) {
    std::string result;
    for (char c : value) { if (c == '"' || c == '\\') result += '\\'; result += c; }
    return result;
}

std::vector<Stimulus> discover(const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory)) throw std::runtime_error("study input is not a directory");
    std::vector<Stimulus> result;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".wav") continue;
        const auto stem = entry.path().stem().string();
        const auto separator = stem.find("__");
        result.push_back({separator == std::string::npos ? "demo" : stem.substr(0, separator),
                          separator == std::string::npos ? stem : stem.substr(separator + 2),
                          std::filesystem::absolute(entry.path())});
    }
    if (result.empty()) throw std::runtime_error("no WAV files found for listening study");
    return result;
}

bool reference_name(const std::string& system) {
    return system == "reference" || system == "natural" || system == "ground_truth";
}

}  // namespace

void write_listening_study(const std::filesystem::path& wav_directory,
                           const std::filesystem::path& json_path,
                           const std::filesystem::path& csv_path,
                           StudyManifestOptions options) {
    auto stimuli = discover(wav_directory);
    std::map<std::string, std::vector<Stimulus>> groups;
    for (auto& stimulus : stimuli) groups[stimulus.utterance].push_back(std::move(stimulus));
    std::mt19937_64 random(options.seed);
    for (auto& [utterance, rows] : groups) { (void)utterance; std::shuffle(rows.begin(), rows.end(), random); }

    std::ofstream json(json_path), csv(csv_path);
    if (!json || !csv) throw std::runtime_error("cannot create listening-study outputs");
    csv << "trial_index,utterance_id,position,blind_id,system,file,role\n";
    json << "{\n  \"schema_version\": 1,\n  \"seed\": " << options.seed
         << ",\n  \"mos\": {\"enabled\": " << (options.include_mos ? "true" : "false")
         << ", \"scale\": [1, 5], \"labels\": [\"bad\", \"poor\", \"fair\", \"good\", \"excellent\"]},\n"
         << "  \"mushra\": {\"enabled\": " << (options.include_mushra ? "true" : "false")
         << ", \"scale\": [0, 100], \"requires_hidden_reference\": true, \"requires_anchor\": true},\n"
         << "  \"warning\": \"Trials without natural/reference and anchor systems are marked incomplete.\",\n"
         << "  \"trials\": [\n";
    std::size_t trial_index = 0;
    for (auto group = groups.begin(); group != groups.end(); ++group, ++trial_index) {
        const auto& utterance = group->first;
        const auto& rows = group->second;
        bool has_reference = false, has_anchor = false;
        for (const auto& row : rows) { has_reference |= reference_name(row.system); has_anchor |= row.system.find("anchor") != std::string::npos; }
        json << "    {\"trial_index\": " << trial_index << ", \"utterance_id\": \""
             << json_escape(utterance) << "\", \"mushra_complete\": "
             << (has_reference && has_anchor ? "true" : "false") << ", \"stimuli\": [\n";
        for (std::size_t position = 0; position < rows.size(); ++position) {
            const auto blind = "T" + std::to_string(trial_index + 1) + "_S" + std::to_string(position + 1);
            const std::string role = reference_name(rows[position].system) ? "hidden_reference" :
                (rows[position].system.find("anchor") != std::string::npos ? "anchor" : "candidate");
            json << "      {\"blind_id\": \"" << blind << "\", \"system\": \""
                 << json_escape(rows[position].system) << "\", \"file\": \""
                 << json_escape(rows[position].path.generic_string()) << "\", \"role\": \"" << role << "\"}"
                 << (position + 1 == rows.size() ? "\n" : ",\n");
            csv << trial_index << ',' << utterance << ',' << position << ',' << blind << ','
                << rows[position].system << ",\"" << rows[position].path.string() << "\"," << role << '\n';
        }
        json << "    ]}" << (std::next(group) == groups.end() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";
}

}  // namespace vocal

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\tests\explicit_runtime_test.cpp ===
#include <explicit_neural/include/explicit_acoustic_gen.hpp>
#include <explicit_neural/include/neural_vocoder.hpp>

#include <cmath>
#include <iostream>

int main() {
    try {
        const auto fixtures = std::filesystem::path(VA_TEST_SOURCE_DIR) / "tests/fixtures/explicit";
        vocal::ExplicitAcousticGenerator acoustic(fixtures / "acoustic_generator.onnx");
        vocal::NeuralVocoder vocoder(fixtures / "vocoder_hifigan.onnx");
        const std::vector<std::int64_t> tokens{4, 5, 6};
        const vocal::ProsodyControls controls{{2, 0, 2}, {0, 100, 200, 0}, {1, 2, 3, 4}, 1};
        const auto mel = acoustic.infer(tokens, controls);
        if (mel.frames != 4 || mel.bins != 80) return 1;
        // Fixture formula: f0/1000 + energy/100 + sum(ids)/100000 + sum(durations)/1000000 + sid/1000 + bin/100.
        for (std::size_t frame = 0; frame < 4; ++frame)
            for (std::size_t bin = 0; bin < 80; ++bin) {
                const float expected = controls.f0_contour[frame] / 1000 + controls.energy_contour[frame] / 100 +
                    15.0F / 100000 + 4.0F / 1000000 + .001F + static_cast<float>(bin) / 100;
                if (std::abs(mel.log_mel[frame * 80 + bin] - expected) > 1e-5F) return 1;
            }
        const auto audio = vocoder.synthesize(mel);
        if (audio.sample_rate_hz != 24'000 || audio.samples.size() != 4 * 256) return 1;
        // The fixture vocoder consumes channel 0 in order and repeats each frame 256 times.
        for (std::size_t sample = 0; sample < audio.samples.size(); ++sample)
            if (std::abs(audio.samples[sample] - std::tanh(mel.log_mel[(sample / 256) * 80])) > 1e-5F) return 1;
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\tests\metrics_test.cpp ===
#include "vocal/metrics.hpp"
#include "vocal/features.hpp"
#include "vocal/phonemizer.hpp"
#include "vocal/reporting.hpp"

#include <cmath>
#include <iostream>
#include <filesystem>

int main() {
    int failures = 0;
    const vocal::metrics::Frames a{{9.0, 1.0, 2.0}, {8.0, 2.0, 3.0}};
    if (std::abs(vocal::metrics::mcd_db(a, a)) > 1e-12) ++failures;
    if (std::abs(vocal::metrics::word_error_rate("Hello, world!", "hello world")) > 1e-12) ++failures;
    if (std::abs(vocal::metrics::word_error_rate("one two three", "one three") - 1.0 / 3.0) > 1e-12) ++failures;
    const double ref[] = {100.0, 0.0, 120.0, 130.0};
    const double syn[] = {110.0, 90.0, 120.0, 0.0};
    const auto score = vocal::metrics::pitch(ref, syn);
    if (score.jointly_voiced_frames != 2 || std::abs(score.vuv_error_rate - 0.5) > 1e-12) ++failures;
    // A known analytic waveform tests feature extraction without an archived synthesizer.
    vocal::Waveform canonical_audio{24'000, std::vector<float>(2'400)};
    for (std::size_t i = 0; i < canonical_audio.samples.size(); ++i)
        canonical_audio.samples[i] = .2F * static_cast<float>(std::sin(2.0 * 3.141592653589793 * 200.0 * i / 24'000.0));
    const auto features = vocal::extract_features(canonical_audio);
    if (features.log_mel.empty() || features.log_mel.front().size() != 80 ||
        features.mcep.front().size() != 25) ++failures;
    const auto wav_path = std::filesystem::current_path() / "vocal_acoustics_roundtrip.wav";
    vocal::write_wav_pcm16(wav_path, canonical_audio);
    const auto loaded = vocal::load_wav_mono(wav_path);
    std::filesystem::remove(wav_path);
    if (loaded.sample_rate_hz != 24'000 || loaded.samples.size() != canonical_audio.samples.size()) ++failures;
    const auto resampled = vocal::resample_waveform({22'050, std::vector<float>(22'050, .25F)});
    if (resampled.sample_rate_hz != 24'000 || resampled.samples.size() != 24'000 ||
        std::abs(resampled.samples[12'000] - .25F) > 1e-6F) ++failures;
    if (!vocal::resample_waveform({22'050, {}}).samples.empty()) ++failures;
    const std::vector<vocal::ScoredUtterance> scored{
        {"a", "speaker-a", 4.0, 10.0, .1, .05},
        {"b", "speaker-a", 6.0, 14.0, .2, .15},
        {"c", "speaker-b", 5.0, std::nullopt, .3, .10}};
    const auto summaries = vocal::summarize_by_speaker(scored, 100, 42);
    if (summaries.size() != 3 || !summaries.front().mcd_db) ++failures;
    const auto fixture_root = std::filesystem::path(VA_TEST_SOURCE_DIR) / "tests" / "fixtures";
    const vocal::CmuPhonemizer phonemizer(fixture_root / "cmudict.dict", fixture_root / "tokens.tsv");
    const auto pronunciation = phonemizer.phonemize("hello");
    if (pronunciation.dictionary_hits != 1 || pronunciation.fallback_words != 0 ||
        pronunciation.missing_model_symbols != 0 || pronunciation.token_ids.empty()) ++failures;
    if (pronunciation.token_kinds.size() != pronunciation.token_ids.size()) ++failures;
    bool unvoiced_found = false, stressed_found = false;
    for (auto kind : pronunciation.token_kinds) {
        unvoiced_found |= kind == vocal::ProsodyTokenKind::UnvoicedConsonant;
        stressed_found |= kind == vocal::ProsodyTokenKind::StressedVowel;
    }
    if (!unvoiced_found || !stressed_found || pronunciation.token_kinds.back() != vocal::ProsodyTokenKind::Boundary) ++failures;
    if (failures) std::cerr << failures << " test(s) failed\n";
    return failures == 0 ? 0 : 1;
}

// === C:\Users\Cutie Magic 500\projects\creative\generative-media-research\src\audio_synth\tests\prosody_test.cpp ===
#include <explicit_neural/include/length_regulator.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int failures{};
void check(bool ok, const char* label) { if (!ok) { ++failures; std::cerr << label << '\n'; } }
template<class F> void rejects(F call, const char* label) {
    try { call(); check(false, label); } catch (const std::exception&) {}
}
}

int main() {
    const std::vector<float> hidden{1, 2, 3, 4, 5, 6};
    const std::vector<std::int64_t> durations{2, 0, 3};
    const auto expanded = vocal::length_regulate(hidden, 2, durations);
    check(expanded.frames == 5 && expanded.hidden_dim == 2 &&
          expanded.values == std::vector<float>{1, 2, 1, 2, 5, 6, 5, 6, 5, 6}, "token ordering / zero-duration expansion");
    check(vocal::length_regulate(hidden, 2, std::vector<std::int64_t>{0, 0, 0}).values.empty(), "all-zero regulator");
    rejects([&] { (void)vocal::length_regulate(hidden, 2, std::vector<std::int64_t>{-1, 1, 1}); }, "negative duration");
    rejects([&] { (void)vocal::length_regulate(hidden, 2, durations, 4); }, "frame budget");
    rejects([&] { (void)vocal::length_regulate(hidden, 0, durations); }, "zero hidden dimension");
    rejects([&] { (void)vocal::length_regulate(hidden, 3, durations); }, "hidden shape mismatch");
    rejects([&] { (void)vocal::duration_frames(std::vector<std::int64_t>{std::numeric_limits<std::int64_t>::max(), 1}); }, "duration overflow");
    vocal::ProsodyControls controls{{2, 0, 2}, {0, 100, 200, 0}, {1, 3, 1, 3}, 2};
    const auto scaled = vocal::apply_prosody_sliders(controls, {1.5F, 2.0F, 2.0F, 0.0F});
    check(scaled.durations == std::vector<std::int64_t>{1, 0, 1}, "cadence preserves skipped tokens");
    check(scaled.f0_contour == std::vector<float>{0, 300}, "pitch preserves unvoiced zeros and token boundaries");
    check(scaled.energy_contour == std::vector<float>{4, 4} && scaled.speaker_id == 2, "energy mean and speaker");
    rejects([&] { (void)vocal::apply_prosody_sliders(controls, {1, 0, 1, 1}); }, "zero speed");
    rejects([&] { (void)vocal::apply_prosody_sliders(controls, {std::numeric_limits<float>::quiet_NaN(), 1, 1, 1}); }, "nonfinite pitch");
    controls.f0_contour.pop_back();
    rejects([&] { vocal::validate_prosody(controls, 3); }, "contour mismatch");
    controls = vocal::baseline_prosody(3);
    check(controls.f0_contour.size() == 18 && controls.energy_contour.size() == 18, "baseline contour length");
    controls.energy_contour[0] = std::numeric_limits<float>::infinity();
    rejects([&] { vocal::validate_prosody(controls, 3); }, "nonfinite energy");
    using Kind = vocal::ProsodyTokenKind;
    using Emotion = vocal::VocalEmotion;
    const vocal::ProsodyControls expressive{{10, 10, 0, 10, 4},
        std::vector<float>(34, 150), std::vector<float>(34, 1), 7,
        {Kind::UnvoicedConsonant, Kind::StressedVowel, Kind::Unknown, Kind::Vowel, Kind::Boundary}};
    auto base = expressive;
    for (std::size_t i = 0; i < 10; ++i) base.f0_contour[i] = 0;
    for (std::size_t i = 10; i < 20; ++i) base.f0_contour[i] = 130;
    for (std::size_t i = 20; i < 30; ++i) base.f0_contour[i] = 170;
    for (std::size_t i = 30; i < 34; ++i) base.f0_contour[i] = 0;
    base.energy_contour[12] = 2;
    const auto neutral = vocal::apply_emotion_preset(base, Emotion::Neutral);
    check(neutral.durations == base.durations && neutral.f0_contour == base.f0_contour &&
          neutral.energy_contour == base.energy_contour && neutral.token_kinds == base.token_kinds,
          "neutral is an identity transform");
    const auto whisper = vocal::apply_emotion_preset(base, Emotion::Whisper);
    check(whisper.durations == std::vector<std::int64_t>{11, 10, 0, 10, 4}, "whisper lengthens only annotated unvoiced consonants");
    check(whisper.f0_contour[0] == 0 && std::abs(whisper.f0_contour[11] - 147) < 1e-4F &&
          std::abs(whisper.f0_contour[21] - 153) < 1e-4F, "whisper flattens voiced range and preserves unvoiced zeros");
    check(std::abs(whisper.energy_contour[13] - 1.2F) < 1e-5F, "whisper energy falls forty percent");
    const auto excited = vocal::apply_emotion_preset(base, Emotion::Excited);
    check(excited.durations == std::vector<std::int64_t>{9, 9, 0, 9, 3}, "excited speeds cadence with per-token rounding");
    check(std::abs(excited.f0_contour[9] - 153) < 1e-4F && std::abs(excited.f0_contour[18] - 217) < 1e-4F,
          "excited raises voiced mean by 35 Hz and range by 1.6");
    check(excited.energy_contour[11] > 2 && excited.f0_contour[0] == 0, "excited boosts peaks without voicing consonants");
    const auto somber = vocal::apply_emotion_preset(base, Emotion::Somber);
    check(somber.durations == std::vector<std::int64_t>{11, 11, 0, 11, 5}, "somber extends cadence at speed 0.88");
    check(std::abs(somber.f0_contour[11] - 112) < 1e-4F && somber.f0_contour[32] < somber.f0_contour[22],
          "somber lowers voiced mean and falls at sentence ending");
    check(somber.energy_contour[14] < base.energy_contour[12], "somber dampens energy peak");
    const auto authoritative = vocal::apply_emotion_preset(base, Emotion::Authoritative);
    check(authoritative.durations == std::vector<std::int64_t>{10, 10, 0, 10, 2}, "authoritative sharpens boundary cadence");
    check(std::abs(authoritative.f0_contour[10] - 143) < 1e-4F && authoritative.energy_contour[10] > authoritative.energy_contour[20]
          && authoritative.energy_contour[30] < authoritative.energy_contour[20], "authoritative stress and boundary dynamics");
    for (auto emotion : {Emotion::Neutral, Emotion::Whisper, Emotion::Excited, Emotion::Somber, Emotion::Authoritative}) {
        const auto result = vocal::apply_emotion_preset(base, emotion);
        vocal::validate_prosody(result, base.durations.size());
        check(result.speaker_id == 7 && result.durations[2] == 0 && result.token_kinds == base.token_kinds,
              "every preset preserves speaker, skipped token and annotations");
    }
    check(base.durations == expressive.durations && base.energy_contour[12] == 2, "presets leave base unmodified");
    auto unvoiced = vocal::baseline_prosody(2, 5, 0);
    unvoiced.token_kinds = {Kind::Unknown, Kind::Boundary};
    for (auto emotion : {Emotion::Whisper, Emotion::Excited, Emotion::Somber, Emotion::Authoritative}) {
        const auto result = vocal::apply_emotion_preset(unvoiced, emotion);
        for (auto value : result.f0_contour) check(value == 0, "all-unvoiced input stays unvoiced");
    }
    auto no_annotations = base;
    no_annotations.token_kinds.clear();
    check(vocal::apply_emotion_preset(no_annotations, Emotion::Whisper).durations == base.durations,
          "missing annotations never guess consonant identity");
    const vocal::ProsodyControls phrases{{2, 2, 0, 2, 2}, std::vector<float>(8, 150),
        std::vector<float>(8, 1), 0,
        {Kind::StressedVowel, Kind::Boundary, Kind::Unknown, Kind::StressedVowel, Kind::Boundary}};
    const auto phrase_controls = vocal::apply_emotion_preset(phrases, Emotion::Authoritative);
    check(std::abs(phrase_controls.energy_contour[0] - 1.3125F) < 1e-5F &&
          std::abs(phrase_controls.energy_contour[3] - 1.3125F) < 1e-5F,
          "authoritative emphasis resets at each phrase boundary");
    const auto low_pitch = vocal::apply_emotion_preset(vocal::baseline_prosody(1, 2, 10), Emotion::Somber);
    for (auto value : low_pitch.f0_contour) check(value > 0, "somber floor never converts voiced input to unvoiced");
    auto invalid = base;
    invalid.token_kinds.pop_back();
    rejects([&] { (void)vocal::apply_emotion_preset(invalid, Emotion::Whisper); }, "annotation mismatch rejected");
    invalid = base;
    invalid.token_kinds[0] = static_cast<Kind>(99);
    rejects([&] { (void)vocal::apply_emotion_preset(invalid, Emotion::Whisper); }, "unknown token annotation rejected");
    rejects([&] { (void)vocal::apply_emotion_preset(base, static_cast<Emotion>(99)); }, "unknown emotion rejected");
    invalid = vocal::baseline_prosody(1, vocal::maximum_prosody_frames);
    invalid.token_kinds = {Kind::UnvoicedConsonant};
    rejects([&] { (void)vocal::apply_emotion_preset(invalid, Emotion::Whisper); }, "emotion expansion respects frame limit");
    return failures ? 1 : 0;
}

