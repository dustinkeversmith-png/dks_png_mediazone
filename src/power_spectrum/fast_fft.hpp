#include <fftw3.h>
#include <vector>
#include <cmath>

class FastFFT {
public:
    int n_fft;
    int num_bins;
    float* in_buf;
    fftwf_complex* out_buf;
    fftwf_plan plan;

public:
    FastFFT(int fft_size = 512) : n_fft(fft_size), num_bins(fft_size / 2 + 1) {
        // SIMD-aligned allocation
        in_buf = fftwf_alloc_real(n_fft);
        out_buf = fftwf_alloc_complex(num_bins);
        
        // Plan r2c 1D transform
        plan = fftwf_plan_dft_r2c_1d(n_fft, in_buf, out_buf, FFTW_ESTIMATE);
    }

    ~FastFFT() {
        fftwf_destroy_plan(plan);
        fftwf_free(in_buf);
        fftwf_free(out_buf);
    }

    // Computes Power Spectrum: |X(f)|^2
    std::vector<float> compute_power_spectrum(const std::vector<float>& frame) {
        // Copy frame into aligned FFTW input buffer
        for (int i = 0; i < n_fft; ++i) {
            in_buf[i] = (i < frame.size()) ? frame[i] : 0.0f;
        }

        fftwf_execute(plan);

        std::vector<float> power_spec(num_bins);
        for (int k = 0; k < num_bins; ++k) {
            float re = out_buf[k][0];
            float im = out_buf[k][1];
            power_spec[k] = (re * re + im * im) / n_fft; // Normalized power
        }
        return power_spec;
    }
};