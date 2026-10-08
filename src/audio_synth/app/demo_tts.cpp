#include "cli_options.hpp"
#include "demo_tts.hpp"
#include <dsp_paradigms/synthesizers.hpp>

#include <iostream>

int run_demo_tts(vocal::SynthesizerKind kind, int argc, char** argv) {
    try {
        const std::string model(vocal::name(kind));
        const vocal::cli::Options options(argc, argv, {"--text", "--output"});
        if (options.help) {
            std::cout << "Usage: " << model << "-tts [--text TEXT] [--output artifacts/" << model << "/speech.wav]\n"
                      << "Deterministic untrained synthesis sketch, mono 24 kHz PCM16.\n";
            return 0;
        }
        const auto text = options.get("--text", "We synthesize a clear acoustic voice.");
        if (text.empty() || text.size() > 10'000) throw std::invalid_argument("text must have 1..10000 characters");
        const std::filesystem::path output = options.get("--output", "artifacts/" + model + "/speech.wav");
        const auto audio = vocal::synthesize_demo(kind, text);
        vocal::write_wav_pcm16(output, audio);
        auto report = vocal::cli::report_file(output);
        report << "{\n  \"model\": \"" << model << "\",\n  \"pretrained\": false,\n  \"sample_rate\": "
               << audio.sample_rate_hz << ",\n  \"samples\": " << audio.samples.size() << "\n}\n";
        if (!report) throw std::runtime_error("failed to write diagnostics");
        std::cout << output.string() << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
