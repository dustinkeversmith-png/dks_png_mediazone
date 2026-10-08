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
