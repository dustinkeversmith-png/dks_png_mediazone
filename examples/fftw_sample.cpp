#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    constexpr int sample_count = 8;
    constexpr float pi = 3.14159265358979323846f;

    float* input = static_cast<float*>(fftwf_malloc(sizeof(float) * sample_count));
    fftwf_complex* spectrum = static_cast<fftwf_complex*>(
        fftwf_malloc(sizeof(fftwf_complex) * (sample_count / 2 + 1)));
    float* output = static_cast<float*>(fftwf_malloc(sizeof(float) * sample_count));
    if (input == nullptr || spectrum == nullptr || output == nullptr) {
        std::cerr << "FFTW sample failed: allocation error\n";
        fftwf_free(input);
        fftwf_free(spectrum);
        fftwf_free(output);
        return 1;
    }

    for (int index = 0; index < sample_count; ++index) {
        input[index] = std::sin(2.0f * pi * index / sample_count) +
                       0.25f * std::sin(4.0f * pi * index / sample_count);
    }

    fftwf_plan forward = fftwf_plan_dft_r2c_1d(sample_count, input, spectrum, FFTW_ESTIMATE);
    fftwf_plan inverse = fftwf_plan_dft_c2r_1d(sample_count, spectrum, output, FFTW_ESTIMATE);
    if (forward == nullptr || inverse == nullptr) {
        std::cerr << "FFTW sample failed: plan creation error\n";
        fftwf_destroy_plan(forward);
        fftwf_destroy_plan(inverse);
        fftwf_free(input);
        fftwf_free(spectrum);
        fftwf_free(output);
        return 1;
    }

    fftwf_execute(forward);
    fftwf_execute(inverse);

    float maximum_error = 0.0f;
    for (int index = 0; index < sample_count; ++index) {
        output[index] /= static_cast<float>(sample_count);
        maximum_error = std::max(maximum_error, std::fabs(output[index] - input[index]));
    }

    std::cout << "FFTW single-precision round trip error = " << maximum_error << "\n";

    fftwf_destroy_plan(forward);
    fftwf_destroy_plan(inverse);
    fftwf_free(input);
    fftwf_free(spectrum);
    fftwf_free(output);
    fftwf_cleanup();

    if (maximum_error > 1e-5f) {
        std::cerr << "FFTW sample failed: reconstruction error is too large\n";
        return 1;
    }

    std::cout << "FFTW sample passed\n";
    return 0;
}
