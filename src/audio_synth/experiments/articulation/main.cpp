#include "articulation.hpp"
#include "app/asset_paths.hpp"
#include "app/cli_options.hpp"
#include <explicit_neural/include/explicit_acoustic_gen.hpp>
#include <explicit_neural/include/neural_vocoder.hpp>
#include <explicit_neural/include/prosody_predictor.hpp>
#include <iomanip>
#include <iostream>

namespace {
using vocal::experiment::ArticulationEdit;
using vocal::experiment::Articulation;
using vocal::experiment::ArticulationCase;
void quoted(std::ostream& out, std::string_view value) {
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 32) out << "\\u00" << "0123456789abcdef"[c >> 4] << "0123456789abcdef"[c & 15];
        else out << static_cast<char>(c);
    }
    out << '"';
}
void guard_output(const std::filesystem::path& assets, const std::filesystem::path& output) {
    auto asset_key = std::filesystem::absolute(assets).lexically_normal().generic_string();
    while (!asset_key.empty() && asset_key.back() == '/') asset_key.pop_back();
    asset_key += '/';
    auto output_key = std::filesystem::absolute(output).lexically_normal().generic_string();
    auto lower = [](unsigned char value) { return static_cast<char>(std::tolower(value)); };
    std::transform(asset_key.begin(),asset_key.end(),asset_key.begin(),lower);
    std::transform(output_key.begin(),output_key.end(),output_key.begin(),lower);
    if (output.extension() != ".wav" || output_key.starts_with(asset_key))
        throw std::invalid_argument("experiment output must be a .wav outside model assets");
}
}

int main(int argc, char** argv) {
    try {
        const vocal::cli::Options options(argc, argv, {"--assets", "--text", "--output", "--speaker-id", "--threads",
            "--word-index", "--phone", "--duration-scale", "--pitch-scale", "--energy-scale", "--pitch-rise-hz",
            "--energy-to-pitch", "--vowel-peak-lengthening", "--edits", "--sweep"});
        if (options.help || argc == 1) {
            std::cout << "Usage: articulation-lab --text TEXT [--assets model_assets/explicit_neural]\n"
                      << "  --word-index N (zero-based, -1 all words) --phone all|vowels|stressed|consonants|unvoiced|ARPABET\n"
                      << "  --duration-scale 1 --pitch-scale 1 --energy-scale 1 --pitch-rise-hz 0\n"
                      << "  --energy-to-pitch 0 --vowel-peak-lengthening 0 --speaker-id 0 --threads 4\n"
                      << "  --output artifacts/articulation/custom.wav OR --sweep artifacts/articulation/sweep\n"
                      << "  --edits FILE (8 columns: word selector duration pitch energy rise effort peak-lengthening)\n"
                      << "Separate experimental runner; does not change assets or production controls.\n";
            return options.help ? 0 : 2;
        }
        const auto text = options.required("--text");
        if (options.has("--sweep") || options.has("--edits")) {
            for (const auto* flag : {"--phone", "--duration-scale", "--pitch-scale", "--energy-scale",
                                    "--pitch-rise-hz", "--energy-to-pitch", "--vowel-peak-lengthening"})
                if (options.has(flag)) throw std::invalid_argument("sweep/edit table cannot be combined with scalar edit flags");
            if (options.has("--edits") && options.has("--word-index"))
                throw std::invalid_argument("edit table supplies its own word indices");
        }
        const std::filesystem::path assets = options.get("--assets", "model_assets/explicit_neural");
        std::map<std::string, std::string> properties;
        std::ifstream config(assets / "pipeline.properties");
        for (std::string line; std::getline(config,line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const auto separator = line.find('=');
            if (separator != std::string::npos) properties[line.substr(0,separator)] = line.substr(separator+1);
        }
        if (properties["sample_rate"] != "24000" || properties["hop_length"] != "300")
            throw std::invalid_argument("experiment requires the staged native 24 kHz/hop-300 VCTK pair");
        const auto speaker = options.number<std::int64_t>("--speaker-id",0);
        if (speaker < 0 || speaker >= vocal::cli::Options::parse<std::int64_t>(properties.at("num_speakers")))
            throw std::invalid_argument("speaker ID outside model range");
        const auto threads = options.number<int>("--threads",4);
        if (threads < 1) throw std::invalid_argument("threads must be positive");
        const vocal::CmuPhonemizer frontend(vocal::cli::resolve_dictionary(assets), assets / "tokens.tsv");
        const auto aligned = vocal::experiment::align_text(frontend,text,
            vocal::experiment::read_vocabulary(assets / "tokens.tsv"));
        vocal::NeuralProsodyPredictor predictor(assets / "prosody_predictor.onnx",threads);
        auto base = predictor.predict(aligned.phonemes.token_ids,speaker);
        base.speaker_id = speaker; base.token_kinds = aligned.phonemes.token_kinds;
        vocal::ExplicitAcousticGenerator acoustic(assets / "acoustic_generator.onnx",{threads,{},"mel"});
        vocal::NeuralVocoderConfig vocoder_config;
        vocoder_config.hop_length = 300; vocoder_config.intra_op_threads = threads;
        vocal::NeuralVocoder vocoder(assets / "vocoder_hifigan.onnx",vocoder_config);
        const auto word = options.number<std::int64_t>("--word-index",-1);
        std::vector<ArticulationCase> cases;
        if (options.has("--sweep")) {
            if (options.has("--edits") || options.has("--output"))
                throw std::invalid_argument("sweep cannot be combined with edits/output");
            if (word < 0 || static_cast<std::size_t>(word) >= aligned.words.size())
                throw std::invalid_argument("sweep requires a valid explicit --word-index");
            cases = Articulation::expressive_sweep(word);
        } else {
            auto edits = options.has("--edits") ? vocal::experiment::read_edits(options.get("--edits")) :
                std::vector<ArticulationEdit>{{word,options.get("--phone","all"),
                    options.number<float>("--duration-scale",1), options.number<float>("--pitch-scale",1),
                    options.number<float>("--energy-scale",1), options.number<float>("--pitch-rise-hz",0),
                    options.number<float>("--energy-to-pitch",0), options.number<float>("--vowel-peak-lengthening",0)}};
            cases.push_back({"custom",std::move(edits)});
        }
        const std::filesystem::path output_root = options.get("--sweep","artifacts/articulation");
        guard_output(assets, options.has("--sweep") ? output_root / "baseline.wav" :
            std::filesystem::path(options.get("--output","artifacts/articulation/custom.wav")));
        // Prevalidate the entire sweep before writing samples. No silently skipped selectors.
        Articulation articulation(aligned, base);
        std::vector<vocal::ProsodyControls> prepared;
        for (const auto& item : cases) {
            articulation.reset().apply(item.edits);
            prepared.push_back(articulation.controls());
        }
        if (options.has("--sweep")) std::filesystem::create_directories(output_root);
        std::ofstream summary;
        if (options.has("--sweep")) {
            summary.open(output_root / "measurements.csv");
            if (!summary) throw std::runtime_error("cannot write sweep summary");
            summary << "case,seconds,frames,rms,peak,clipped_samples,acoustic_ms,vocoder_ms\n";
        }
        for (std::size_t c = 0; c < cases.size(); ++c) {
            const auto& item = cases[c]; const auto& controls = prepared[c];
            const std::filesystem::path output = options.has("--sweep") ? output_root / (item.name + ".wav") :
                std::filesystem::path(options.get("--output","artifacts/articulation/custom.wav"));
            const auto mel = acoustic.infer(aligned.phonemes.token_ids,controls);
            const auto audio = vocoder.synthesize(mel);
            double squared = 0; float peak = 0; std::size_t clipped = 0;
            for (float sample : audio.samples) {
                squared += sample * sample; peak = std::max(peak,std::abs(sample));
                if (std::abs(sample) > 1) ++clipped;
            }
            const double rms = std::sqrt(squared / static_cast<double>(audio.samples.size()));
            vocal::write_wav_pcm16(output,audio);
            auto report = vocal::cli::report_file(output);
            report << std::setprecision(9) << "{\n\"experiment\":\"articulation-lab\",\"text\":";
            quoted(report,text);
            report << ",\"sample_rate\":24000,\"hop_length\":300,\"speaker_id\":" << speaker
                   << ",\"frames\":" << mel.frames << ",\"samples\":" << audio.samples.size()
                   << ",\"rms\":" << rms << ",\"peak\":" << peak << ",\"clipped_samples\":" << clipped
                   << ",\"acoustic_ms\":" << acoustic.inference_ms() << ",\"vocoder_ms\":" << vocoder.inference_ms()
                   << ",\"dictionary_hits\":" << aligned.phonemes.dictionary_hits
                   << ",\"fallback_words\":" << aligned.phonemes.fallback_words << ",\"words\":[";
            std::vector<std::size_t> offsets(controls.durations.size()+1);
            for (std::size_t i = 0; i < controls.durations.size(); ++i)
                offsets[i+1] = offsets[i] + static_cast<std::size_t>(controls.durations[i]);
            for (std::size_t i = 0; i < aligned.words.size(); ++i) {
                const auto& w = aligned.words[i];
                if (i) report << ',';
                report << "{\"index\":" << i << ",\"text\":"; quoted(report,w.text);
                report << ",\"byte_begin\":" << w.byte_begin << ",\"byte_end\":" << w.byte_end
                       << ",\"token_begin\":" << w.token_begin << ",\"token_end\":" << w.token_end
                       << ",\"frame_begin\":" << offsets[w.token_begin] << ",\"frame_end\":" << offsets[w.token_end] << '}';
            }
            report << "],\"edits\":[";
            for (std::size_t i = 0; i < item.edits.size(); ++i) {
                const auto& e = item.edits[i]; if (i) report << ',';
                report << "{\"word_index\":" << e.word_index << ",\"selector\":"; quoted(report,e.selector);
                report << ",\"duration_scale\":" << e.duration_scale << ",\"pitch_scale\":" << e.pitch_scale
                       << ",\"energy_scale\":" << e.energy_scale << ",\"pitch_rise_hz\":" << e.pitch_rise_hz
                       << ",\"effort_gain\":" << e.effort_gain << ",\"vowel_peak_gain\":" << e.vowel_peak_gain << '}';
            }
            report << "],\"phones\":[";
            for (std::size_t i = 0; i < aligned.phones.size(); ++i) { if (i) report << ','; quoted(report,aligned.phones[i]); }
            report << "],\"input_ids\":"; vocal::cli::array(report,aligned.phonemes.token_ids);
            report << ",\"base_durations\":"; vocal::cli::array(report,base.durations);
            report << ",\"durations\":"; vocal::cli::array(report,controls.durations);
            report << ",\"f0_hz\":"; vocal::cli::array(report,controls.f0_contour);
            report << ",\"energy\":"; vocal::cli::array(report,controls.energy_contour);
            report << "}\n";
            if (!report) throw std::runtime_error("failed to write experiment diagnostics");
            if (summary.is_open()) summary << item.name << ',' << static_cast<double>(audio.samples.size()) / 24000. << ','
                << mel.frames << ',' << rms << ',' << peak << ',' << clipped << ',' << acoustic.inference_ms()
                << ',' << vocoder.inference_ms() << '\n';
            std::cout << output.string() << " frames=" << mel.frames << " peak=" << peak << '\n';
        }
        if (summary.is_open() && !summary) throw std::runtime_error("failed to write sweep summary");
        return 0;
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
