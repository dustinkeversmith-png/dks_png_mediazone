#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

struct Formant {
    float frequency; // in Hz
    float bandwidth; // in Hz
};

class FormantTracker {
public:
    // Levinson-Durbin Recursion
    static Eigen::VectorXf levinson_durbin(const Eigen::VectorXf& r, int order) {
        Eigen::VectorXf a(order + 1);
        a.setZero();
        a(0) = 1.0f;

        float e = r(0);
        if (e <= 0.0f) return a; // Avoid division by zero on silence

        for (int i = 1; i <= order; ++i) {
            float lambda = 0.0f;
            for (int j = 1; j < i; ++j) {
                lambda += a(j) * r(i - j);
            }
            lambda = (r(i) - lambda) / e;

            Eigen::VectorXf a_prev = a;
            a(i) = lambda;
            for (int j = 1; j < i; ++j) {
                a(j) = a_prev(j) - lambda * a_prev(i - j);
            }
            e *= (1.0f - lambda * lambda);
        }
        return a;
    }

    // Extract F1, F2, F3, etc.
    static std::vector<Formant> extract_formants(const std::vector<float>& frame, 
                                                 float sample_rate = 16000.0f, 
                                                 int order = 16) {
        int N = frame.size();

        // 1. Autocorrelation
        Eigen::VectorXf r(order + 1);
        r.setZero();
        for (int k = 0; k <= order; ++k) {
            for (int n = 0; n < N - k; ++n) {
                r(k) += frame[n] * frame[n + k];
            }
        }

        // 2. Levinson-Durbin
        Eigen::VectorXf a = levinson_durbin(r, order);

        // 3. Build Companion Matrix for Root Finding
        // Polynomial: z^p - a_1*z^(p-1) - a_2*z^(p-2) ... - a_p = 0
        Eigen::MatrixXf companion = Eigen::MatrixXf::Zero(order, order);
        for (int col = 0; col < order; ++col) {
            companion(0, col) = a(col + 1);
        }
        for (int row = 1; row < order; ++row) {
            companion(row, row - 1) = 1.0f;
        }

        // 4. Compute Eigenvalues
        Eigen::EigenSolver<Eigen::MatrixXf> solver(companion, /* computeEigenvectors = */ false);
        auto roots = solver.eigenvalues();

        // 5. Convert Roots to Formant Frequencies & Bandwidths
        std::vector<Formant> formants;
        for (int i = 0; i < roots.size(); ++i) {
            std::complex<float> z = roots(i);
            
            // Look only at the upper half of the unit circle (positive frequencies)
            if (z.imag() > 0.0f) {
                float freq = (std::atan2(z.imag(), z.real()) * sample_rate) / (2.0f * M_PI);
                float bw = -(sample_rate / M_PI) * std::log(std::abs(z));

                // Standard speech filters: Bandwidth should be sharp (< 400Hz) and Freq in range
                if (freq > 50.0f && freq < (sample_rate / 2.0f - 50.0f) && bw < 400.0f) {
                    formants.push_back({freq, bw});
                }
            }
        }

        // Sort ascending by frequency (F1 < F2 < F3)
        std::sort(formants.begin(), formants.end(), [](const Formant& a, const Formant& b) {
            return a.frequency < b.frequency;
        });

        return formants;
    }
};