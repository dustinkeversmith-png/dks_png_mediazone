#include <cmath>
#include <vector>

// Precompute Hamming Window coefficients: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N - 1))
std::vector<float> create_hamming_window(int frame_length) {
    std::vector<float> window(frame_length);
    for (int n = 0; n < frame_length; ++n) {
        window[n] = 0.54f - 0.46f * std::cos((2.0f * M_PI * n) / (frame_length - 1));
    }
    return window;
}

// Slice the raw audio stream into windowed frames ready for FFT / LPC
//Frame Length ($N$): $25\text{ ms} \rightarrow 0.025 \times 16000 = \mathbf{400\text{ samples}}$ (padded with zeros to $512$ for power-of-two FFTs).Frame Step / Hop Size ($M$): $10\text{ ms} \rightarrow 0.010 \times 16000 = \mathbf{160\text{ samples}}$ (yields a 60% overlap).Windowing Function: Multiply by a Hamming or Hanning window before spectral analysis to eliminate spectral leakage at the frame edges.
std::vector<std::vector<float>> chop_into_frames(const std::vector<float>& audio, 
                                                 int frame_length = 400, 
                                                 int hop_size = 160, 
                                                 int fft_size = 512) {
    std::vector<float> window = create_hamming_window(frame_length);
    std::vector<std::vector<float>> frames;

    for (size_t start = 0; start + frame_length <= audio.size(); start += hop_size) {
        std::vector<float> frame(fft_size, 0.0f); // Zero-padded buffer for FFT
        
        for (int i = 0; i < frame_length; ++i) {
            frame[i] = audio[start + i] * window[i];
        }
        frames.push_back(std::move(frame));
    }
    return frames;
}