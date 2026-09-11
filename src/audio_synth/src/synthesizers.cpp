#include "vocal/synthesizers.hpp"

#include "vocal/neural_models.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>

namespace vocal {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct Control {
    double f0{118.0};
    double f1{500.0};
    double f2{1500.0};
    double f3{2500.0};
    double voiced{1.0};
    double energy{0.7};
    double seconds{0.085};
};

bool is_vowel(char c) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

Control control_for(char input, std::size_t index) {
    const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(input)));
    Control p;
    p.f0 += 8.0 * std::sin(static_cast<double>(index) * 0.55);
    switch (c) {
    case 'a': p.f1 = 730; p.f2 = 1090; p.f3 = 2440; p.seconds = .12; break;
    case 'e': p.f1 = 530; p.f2 = 1840; p.f3 = 2480; p.seconds = .11; break;
    case 'i': case 'y': p.f1 = 300; p.f2 = 2250; p.f3 = 3000; p.seconds = .105; break;
    case 'o': p.f1 = 570; p.f2 = 840; p.f3 = 2410; p.seconds = .12; break;
    case 'u': p.f1 = 300; p.f2 = 870; p.f3 = 2240; p.seconds = .115; break;
    case 's': case 'f': case 'h': case 'x': case 'z':
        p.voiced = 0.0; p.f1 = 3500; p.f2 = 5200; p.f3 = 6800; p.seconds = .09; break;
    case 'm': case 'n': case 'l': case 'r': case 'w':
        p.f1 = 300; p.f2 = 1100; p.f3 = 2200; p.energy = .48; break;
    case ' ': p.voiced = 0.0; p.energy = 0.0; p.seconds = .065; break;
    case '.': case ',': case '!': case '?':
        p.voiced = 0.0; p.energy = 0.0; p.seconds = .14; break;
    default:
        p.f1 = 420 + static_cast<unsigned char>(c) % 170;
        p.f2 = 1250 + static_cast<unsigned char>(c) % 650;
        p.f3 = 2400 + static_cast<unsigned char>(c) % 500;
        p.voiced = .65; p.energy = .55; p.seconds = .072; break;
    }
    return p;
}

std::vector<Control> controls(std::string_view text, SynthesizerKind kind) {
    std::vector<Control> result;
    result.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        auto p = control_for(text[i], i);
        if (kind == SynthesizerKind::statistical_hmm && !result.empty()) {
            // SPSS mean trajectories demonstrate the familiar over-smoothed sound.
            p.f0 = .72 * result.back().f0 + .28 * p.f0;
            p.f1 = .68 * result.back().f1 + .32 * p.f1;
            p.f2 = .68 * result.back().f2 + .32 * p.f2;
            p.f3 = .68 * result.back().f3 + .32 * p.f3;
            p.energy *= .78;
        } else if (kind == SynthesizerKind::articulatory) {
            // A tract-length/opening proxy: constrain resonances to a tube family.
            const double length_cm = 16.5 + 1.2 * std::sin(i * .37);
            const double quarter_wave = 35000.0 / (4.0 * length_cm);
            const double opening = std::clamp((p.f1 - 280.0) / 500.0, 0.0, 1.0);
            p.f1 = quarter_wave * (.72 + .55 * opening);
            p.f2 = quarter_wave * (2.65 - .7 * opening);
            p.f3 = quarter_wave * (4.8 - .35 * opening);
        } else if (kind == SynthesizerKind::homebrew_neural ||
                   kind == SynthesizerKind::onnx_style) {
            const bool onnx = kind == SynthesizerKind::onnx_style;
            const double latent = std::tanh(std::sin((static_cast<unsigned char>(text[i]) + 3) *
                                                     (onnx ? .113 : .079)));
            p.f0 += latent * (onnx ? 18.0 : 30.0);
            p.f1 *= 1.0 + latent * (onnx ? .035 : .075);
            p.f2 *= 1.0 - latent * (onnx ? .025 : .06);
            p.energy *= onnx ? .9 : .82;
            if (onnx && is_vowel(text[i])) p.seconds *= 1.22;
        }
        result.push_back(p);
    }
    if (result.empty()) result.push_back(control_for(' ', 0));
    return result;
}

double resonant_harmonics(const Control& p, double time, double bandwidth_scale) {
    double value = 0.0;
    const std::array<double, 3> formants{p.f1, p.f2, p.f3};
    const std::array<double, 3> bandwidths{90.0, 130.0, 180.0};
    for (int harmonic = 1; harmonic <= 48; ++harmonic) {
        const double frequency = p.f0 * harmonic;
        double gain = 0.0;
        for (std::size_t f = 0; f < formants.size(); ++f) {
            const double delta = (frequency - formants[f]) / (bandwidths[f] * bandwidth_scale);
            gain += std::exp(-.5 * delta * delta) / (1.0 + .18 * f);
        }
        value += gain * std::sin(2.0 * kPi * frequency * time) / std::sqrt(harmonic);
    }
    return value * .09;
}

double sine_formants(const Control& p, double time) {
    return .55 * std::sin(2 * kPi * p.f1 * time) +
           .30 * std::sin(2 * kPi * p.f2 * time + .4) +
           .15 * std::sin(2 * kPi * p.f3 * time + .9);
}

double mel_bank_sample(const MelSpectrogram& mel, std::size_t frame, double time, bool onnx) {
    const auto values = mel.frame(std::min(frame, mel.frames - 1));
    double sample = 0.0;
    // Lightweight deterministic oscillator-bank vocoder for audible graph output.
    for (std::size_t band = 0; band < values.size(); band += 4) {
        const double frequency = 80.0 * std::pow(7600.0 / 80.0,
                                                static_cast<double>(band) / values.size());
        const double amplitude = std::exp(std::clamp(static_cast<double>(values[band]), -11.0, -1.0));
        sample += amplitude * std::sin(2 * kPi * frequency * time + band * .17);
    }
    return sample * (onnx ? 1.8 : 1.35);
}

Waveform render(std::string_view text, SynthesizerKind kind, int sample_rate) {
    if (sample_rate < 8'000) throw std::invalid_argument("sample rate must be at least 8 kHz");
    const auto sequence = controls(text, kind);
    double total_seconds = 0.0;
    for (const auto& item : sequence) total_seconds += item.seconds;
    Waveform output{sample_rate, {}};
    std::mt19937 random(0x564f4341U + static_cast<unsigned>(kind));
    std::uniform_real_distribution<double> noise(-1.0, 1.0);
    MelSpectrogram mel;
    if (kind == SynthesizerKind::homebrew_neural) {
        HomebrewAcousticModel model;
        mel = model.infer({text, "demo", sample_rate, 80});
    } else if (kind == SynthesizerKind::onnx_style) {
        OnnxStyleAcousticModel model;
        mel = model.infer({text, "demo", sample_rate, 80});
    }

    double global_time = 0.0;
    for (std::size_t unit = 0; unit < sequence.size(); ++unit) {
        const auto& p = sequence[unit];
        const auto count = static_cast<std::size_t>(p.seconds * sample_rate);
        for (std::size_t i = 0; i < count; ++i) {
            const double local = static_cast<double>(i) / sample_rate;
            const double edge = std::min({1.0, local / .012, (p.seconds - local) / .018});
            double sample = 0.0;
            if (p.energy > 0.0) {
                switch (kind) {
                case SynthesizerKind::sine_wave:
                    sample = sine_formants(p, global_time); break;
                case SynthesizerKind::lpc:
                    sample = resonant_harmonics(p, global_time, .52) + .06 * noise(random); break;
                case SynthesizerKind::articulatory:
                    sample = .75 * resonant_harmonics(p, global_time, 1.4) +
                             .12 * std::sin(2 * kPi * p.f0 * global_time); break;
                case SynthesizerKind::statistical_hmm:
                    sample = resonant_harmonics(p, global_time, 2.1); break;
                case SynthesizerKind::unit_selection: {
                    const double pitch_jitter = 1.0 + .018 * std::sin(unit * 7.0);
                    Control recorded = p; recorded.f0 *= pitch_jitter;
                    sample = resonant_harmonics(recorded, local, .75); break;
                }
                case SynthesizerKind::homebrew_neural:
                case SynthesizerKind::onnx_style: {
                    const auto frame = std::min<std::size_t>(static_cast<std::size_t>(
                        (global_time / total_seconds) * static_cast<double>(mel.frames)), mel.frames - 1);
                    sample = .45 * resonant_harmonics(p, global_time, 1.2) +
                             mel_bank_sample(mel, frame, global_time,
                                             kind == SynthesizerKind::onnx_style);
                    break;
                }
                case SynthesizerKind::formant:
                    sample = resonant_harmonics(p, global_time, .9); break;
                }
                if (p.voiced < .8) sample = p.voiced * sample + (1.0 - p.voiced) * .14 * noise(random);
            }
            output.samples.push_back(static_cast<float>(sample * p.energy * std::max(0.0, edge)));
            global_time += 1.0 / sample_rate;
        }
    }
    normalize_peak(output);
    return output;
}

}  // namespace

std::string_view name(SynthesizerKind kind) {
    switch (kind) {
    case SynthesizerKind::formant: return "formant";
    case SynthesizerKind::statistical_hmm: return "statistical_hmm";
    case SynthesizerKind::articulatory: return "articulatory";
    case SynthesizerKind::lpc: return "lpc";
    case SynthesizerKind::sine_wave: return "sine_wave";
    case SynthesizerKind::unit_selection: return "unit_selection";
    case SynthesizerKind::homebrew_neural: return "homebrew_neural";
    case SynthesizerKind::onnx_style: return "onnx_style";
    }
    return "unknown";
}

Waveform synthesize_demo(SynthesizerKind kind, std::string_view text, int sample_rate_hz) {
    return render(text, kind, sample_rate_hz);
}

std::vector<std::filesystem::path> render_comparison_suite(
    const std::filesystem::path& output_directory, std::string_view text, int sample_rate_hz) {
    static constexpr std::array kinds{
        SynthesizerKind::formant, SynthesizerKind::statistical_hmm,
        SynthesizerKind::articulatory, SynthesizerKind::lpc,
        SynthesizerKind::sine_wave, SynthesizerKind::unit_selection,
        SynthesizerKind::homebrew_neural, SynthesizerKind::onnx_style};
    std::filesystem::create_directories(output_directory);
    std::vector<std::filesystem::path> paths;
    std::ofstream manifest(output_directory / "manifest.csv");
    manifest << "model,file,sample_rate,text\n";
    for (auto kind : kinds) {
        const auto path = output_directory / (std::string(name(kind)) + ".wav");
        write_wav_pcm16(path, synthesize_demo(kind, text, sample_rate_hz));
        paths.push_back(path);
        manifest << name(kind) << ',' << path.filename().string() << ',' << sample_rate_hz << ",\"";
        for (char c : text) manifest << (c == '\"' ? '\'' : c);
        manifest << "\"\n";
    }
    const auto comparison = compare_neural_models({text, "demo", sample_rate_hz, 80});
    std::ofstream report(output_directory / "neural_comparison.json");
    report << "{\n"
           << "  \"note\": \"Untrained deterministic architecture comparison\",\n"
           << "  \"homebrew_frames\": " << comparison.homebrew_frames << ",\n"
           << "  \"onnx_style_frames\": " << comparison.onnx_style_frames << ",\n"
           << "  \"mel_bins\": " << comparison.mel_bins << ",\n"
           << "  \"duration_ratio\": " << comparison.duration_ratio << ",\n"
           << "  \"log_mel_mae\": " << comparison.log_mel_mae << ",\n"
           << "  \"log_mel_rmse\": " << comparison.log_mel_rmse << "\n}\n";
    return paths;
}

}  // namespace vocal
