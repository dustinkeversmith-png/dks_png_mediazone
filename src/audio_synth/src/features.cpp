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
