// Monophone acoustic model: 3-state left-to-right HMM per phone, one diagonal
// covariance Gaussian per state. This is the GMM-HMM structure that legacy
// engines (PocketSphinx, Kaldi's mono stage) use, with a single mixture
// component - trained here from TIMIT's hand-aligned .phn boundaries, so no
// EM bootstrap or neural net is involved.
//
// Speed: the whole inventory is 40 phones * 3 states = 120 Gaussians. The
// decoder scores all 120 once per frame into a flat table, then every active
// token reads that table - so acoustic scoring costs ~120*39 multiply-adds per
// frame no matter how wide the search beam gets.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "mfcc.hpp"
#include "phone_set.hpp"

namespace models {

class AcousticModel {
public:
    static constexpr int kMaxMixtures = 16;

    AcousticModel() { resize(); }

    // ---- training -------------------------------------------------------

    void begin_training() {
        resize();
        sum_.assign(static_cast<size_t>(num_states()) * kFeatureDim, 0.0);
        sum_squares_.assign(static_cast<size_t>(num_states()) * kFeatureDim, 0.0);
        counts_.assign(num_states(), 0.0);
        phone_frames_.assign(num_phones(), 0.0);
        phone_segments_.assign(num_phones(), 0.0);
    }

    // Accumulate one aligned phone segment: frames [start, end) belong to
    // `phone`, split evenly across its three states.
    void accumulate_segment(const FeatureMatrix& features, int start, int end, int phone) {
        if (phone < 0 || end <= start) return;
        const int length = end - start;
        phone_frames_[phone] += length;
        phone_segments_[phone] += 1.0;

        for (int t = start; t < end; ++t) {
            if (t < 0 || t >= features.num_frames) continue;
            int state = ((t - start) * kNumStatesPerPhone) / length;
            if (state >= kNumStatesPerPhone) state = kNumStatesPerPhone - 1;
            const int index = state_index(phone, state);
            const float* row = features.frame(t);
            double* sum = &sum_[static_cast<size_t>(index) * kFeatureDim];
            double* sq = &sum_squares_[static_cast<size_t>(index) * kFeatureDim];
            for (int d = 0; d < kFeatureDim; ++d) {
                sum[d] += row[d];
                sq[d] += static_cast<double>(row[d]) * row[d];
            }
            counts_[index] += 1.0;
        }
    }

    // Turn accumulators into Gaussians and duration-derived transitions.
    // `variance_floor` guards states with too few frames from collapsing.
    void finish_training(double variance_floor = 0.01) {
        // Global variance is the fallback for starved states.
        std::vector<double> global_mean(kFeatureDim, 0.0), global_var(kFeatureDim, 0.0);
        double total = 0.0;
        for (int s = 0; s < num_states(); ++s) {
            total += counts_[s];
            for (int d = 0; d < kFeatureDim; ++d) {
                global_mean[d] += sum_[static_cast<size_t>(s) * kFeatureDim + d];
                global_var[d] += sum_squares_[static_cast<size_t>(s) * kFeatureDim + d];
            }
        }
        if (total > 0) {
            for (int d = 0; d < kFeatureDim; ++d) {
                global_mean[d] /= total;
                global_var[d] = global_var[d] / total - global_mean[d] * global_mean[d];
                if (global_var[d] < variance_floor) global_var[d] = variance_floor;
            }
        }

        for (int s = 0; s < num_states(); ++s) {
            float* mean = &means_[static_cast<size_t>(s) * kFeatureDim];
            float* inv_var = &inv_variances_[static_cast<size_t>(s) * kFeatureDim];
            const double count = counts_[s];
            double log_det = 0.0;

            for (int d = 0; d < kFeatureDim; ++d) {
                double m, v;
                if (count >= 3.0) {
                    m = sum_[static_cast<size_t>(s) * kFeatureDim + d] / count;
                    v = sum_squares_[static_cast<size_t>(s) * kFeatureDim + d] / count - m * m;
                    // Shrink toward the global model when the state is thin.
                    const double weight = count / (count + 10.0);
                    m = weight * m + (1.0 - weight) * global_mean[d];
                    v = weight * v + (1.0 - weight) * global_var[d];
                } else {
                    m = global_mean[d];
                    v = global_var[d];
                }
                if (v < variance_floor) v = variance_floor;
                mean[d] = static_cast<float>(m);
                inv_var[d] = static_cast<float>(1.0 / v);
                log_det += std::log(v);
            }
            // -0.5 * (D*log(2*pi) + log|Sigma|), the constant part of log N(x).
            constexpr double kLog2Pi = 1.8378770664093453;
            log_constants_[s] = static_cast<float>(-0.5 * (kFeatureDim * kLog2Pi + log_det));
            state_occupancy_[s] = static_cast<float>(count);
        }

        // Self-loop probability from the observed mean phone duration:
        // a state visited for `d` frames leaves with probability 1/d.
        for (int p = 0; p < num_phones(); ++p) {
            double frames_per_state = 3.0;
            if (phone_segments_[p] > 0.0) {
                frames_per_state =
                    (phone_frames_[p] / phone_segments_[p]) / kNumStatesPerPhone;
            }
            if (frames_per_state < 1.2) frames_per_state = 1.2;
            const double exit = 1.0 / frames_per_state;
            log_self_loop_[p] = static_cast<float>(std::log(1.0 - exit));
            log_exit_[p] = static_cast<float>(std::log(exit));
        }
    }

    // ---- mixture training ----------------------------------------------

    // Grow each state's single Gaussian into `mixtures` components and refine
    // them with EM on the frames assigned to that state.
    //
    // This is the classical GMM-HMM recipe: split the component with the most
    // occupancy along its principal axis (mean +/- 0.2 sigma), re-estimate,
    // repeat. A monophone with one Gaussian cannot model the fact that the
    // same phone sounds different across speakers and contexts; mixtures buy
    // exactly that, at a scoring cost of components x dimensions per state.
    void train_mixtures(const std::vector<float>& frames, const std::vector<int>& state_of_frame,
                        int mixtures, int em_iterations = 6, double variance_floor = 0.01) {
        if (mixtures <= 1) return;
        mixtures_ = mixtures;

        // Group frame indices by state.
        std::vector<std::vector<int>> by_state(num_states());
        for (size_t i = 0; i < state_of_frame.size(); ++i) {
            const int state = state_of_frame[i];
            if (state >= 0 && state < num_states()) by_state[state].push_back(static_cast<int>(i));
        }

        std::vector<float> new_means(static_cast<size_t>(num_states()) * mixtures * kFeatureDim);
        std::vector<float> new_inv(static_cast<size_t>(num_states()) * mixtures * kFeatureDim);
        std::vector<float> new_const(static_cast<size_t>(num_states()) * mixtures);
        std::vector<float> new_weight(static_cast<size_t>(num_states()) * mixtures);

        std::vector<double> mean(kFeatureDim), variance(kFeatureDim);
        std::vector<double> posterior(mixtures);
        std::vector<double> occupancy(mixtures);
        std::vector<double> sum(static_cast<size_t>(mixtures) * kFeatureDim);
        std::vector<double> sum_squares(static_cast<size_t>(mixtures) * kFeatureDim);

        for (int s = 0; s < num_states(); ++s) {
            const std::vector<int>& indices = by_state[s];
            const size_t base = static_cast<size_t>(s) * mixtures;

            // Seed every component from the state's single Gaussian, then
            // perturb so EM has something to separate.
            for (int k = 0; k < mixtures; ++k) {
                const float shift = (k % 2 == 0 ? 1.0f : -1.0f) * 0.2f * (1 + k / 2);
                for (int d = 0; d < kFeatureDim; ++d) {
                    const float sigma =
                        1.0f / std::sqrt(inv_variances_[static_cast<size_t>(s) * kFeatureDim + d]);
                    new_means[(base + k) * kFeatureDim + d] =
                        means_[static_cast<size_t>(s) * kFeatureDim + d] +
                        (d % 3 == 0 ? shift * sigma : 0.0f);
                    new_inv[(base + k) * kFeatureDim + d] =
                        inv_variances_[static_cast<size_t>(s) * kFeatureDim + d];
                }
                new_weight[base + k] = static_cast<float>(std::log(1.0 / mixtures));
                new_const[base + k] = log_constants_[s];
            }

            // Too few frames to estimate a mixture: keep the copies as-is.
            if (indices.size() < static_cast<size_t>(mixtures) * 20) continue;

            for (int iteration = 0; iteration < em_iterations; ++iteration) {
                std::fill(occupancy.begin(), occupancy.end(), 0.0);
                std::fill(sum.begin(), sum.end(), 0.0);
                std::fill(sum_squares.begin(), sum_squares.end(), 0.0);

                for (int index : indices) {
                    const float* row = frames.data() + static_cast<size_t>(index) * kFeatureDim;

                    // E step: component posteriors via log-sum-exp.
                    double best = -1e30;
                    for (int k = 0; k < mixtures; ++k) {
                        const float* mu = &new_means[(base + k) * kFeatureDim];
                        const float* inv = &new_inv[(base + k) * kFeatureDim];
                        double acc = 0.0;
                        for (int d = 0; d < kFeatureDim; ++d) {
                            const double e = row[d] - mu[d];
                            acc += e * e * inv[d];
                        }
                        posterior[k] = new_weight[base + k] + new_const[base + k] - 0.5 * acc;
                        if (posterior[k] > best) best = posterior[k];
                    }
                    double total = 0.0;
                    for (int k = 0; k < mixtures; ++k) {
                        posterior[k] = std::exp(posterior[k] - best);
                        total += posterior[k];
                    }
                    for (int k = 0; k < mixtures; ++k) {
                        const double weight = posterior[k] / total;
                        if (weight < 1e-6) continue;
                        occupancy[k] += weight;
                        double* s_acc = &sum[static_cast<size_t>(k) * kFeatureDim];
                        double* q_acc = &sum_squares[static_cast<size_t>(k) * kFeatureDim];
                        for (int d = 0; d < kFeatureDim; ++d) {
                            s_acc[d] += weight * row[d];
                            q_acc[d] += weight * row[d] * row[d];
                        }
                    }
                }

                // M step.
                double total_occupancy = 0.0;
                for (int k = 0; k < mixtures; ++k) total_occupancy += occupancy[k];
                if (total_occupancy <= 0.0) break;

                for (int k = 0; k < mixtures; ++k) {
                    if (occupancy[k] < 5.0) {
                        // Starved component: park it with a tiny weight rather
                        // than letting its variance collapse onto one frame.
                        new_weight[base + k] = static_cast<float>(std::log(1e-5));
                        continue;
                    }
                    double log_det = 0.0;
                    for (int d = 0; d < kFeatureDim; ++d) {
                        const double m = sum[static_cast<size_t>(k) * kFeatureDim + d] / occupancy[k];
                        double v = sum_squares[static_cast<size_t>(k) * kFeatureDim + d] /
                                       occupancy[k] - m * m;
                        if (v < variance_floor) v = variance_floor;
                        new_means[(base + k) * kFeatureDim + d] = static_cast<float>(m);
                        new_inv[(base + k) * kFeatureDim + d] = static_cast<float>(1.0 / v);
                        log_det += std::log(v);
                    }
                    constexpr double kLog2Pi = 1.8378770664093453;
                    new_const[base + k] =
                        static_cast<float>(-0.5 * (kFeatureDim * kLog2Pi + log_det));
                    new_weight[base + k] =
                        static_cast<float>(std::log(occupancy[k] / total_occupancy));
                }
            }
        }

        means_.swap(new_means);
        inv_variances_.swap(new_inv);
        log_constants_.swap(new_const);
        log_weights_.swap(new_weight);
    }

    // ---- scoring --------------------------------------------------------

    // log p(x | state): a single Gaussian, or the log-sum-exp over mixture
    // components when the model was trained with more than one.
    float log_likelihood(int state, const float* features) const {
        const size_t base = static_cast<size_t>(state) * mixtures_;
        float best = -1e30f;
        float component[kMaxMixtures];

        for (int k = 0; k < mixtures_; ++k) {
            const float* mean = &means_[(base + k) * kFeatureDim];
            const float* inv_var = &inv_variances_[(base + k) * kFeatureDim];
            float acc0 = 0.0f, acc1 = 0.0f;
            int d = 0;
            for (; d + 2 <= kFeatureDim; d += 2) {
                const float e0 = features[d] - mean[d];
                const float e1 = features[d + 1] - mean[d + 1];
                acc0 += e0 * e0 * inv_var[d];
                acc1 += e1 * e1 * inv_var[d + 1];
            }
            float acc = acc0 + acc1;
            for (; d < kFeatureDim; ++d) {
                const float e = features[d] - mean[d];
                acc += e * e * inv_var[d];
            }
            float value = log_constants_[base + k] - 0.5f * acc;
            if (mixtures_ > 1) value += log_weights_[base + k];
            component[k] = value;
            if (value > best) best = value;
        }
        if (mixtures_ == 1) return best;

        float sum = 0.0f;
        for (int k = 0; k < mixtures_; ++k) sum += std::exp(component[k] - best);
        return best + std::log(sum);
    }

    int mixtures() const { return mixtures_; }

    // Score every state for one frame into `out` (size num_states()).
    void score_frame(const float* features, std::vector<float>& out) const {
        out.resize(num_states());
        for (int s = 0; s < num_states(); ++s) out[s] = log_likelihood(s, features);
    }

    float log_self_loop(int phone) const { return log_self_loop_[phone]; }
    float log_exit(int phone) const { return log_exit_[phone]; }
    float occupancy(int state) const { return state_occupancy_[state]; }

    // ---- persistence ----------------------------------------------------

    bool save(const std::string& path) const {
        FILE* file = std::fopen(path.c_str(), "wb");
        if (!file) return false;
        const int32_t magic = 0x4D414D32;  // "MAM2" (mixture-capable)
        const int32_t states = num_states();
        const int32_t dim = kFeatureDim;
        const int32_t mixtures = mixtures_;
        std::fwrite(&magic, sizeof(magic), 1, file);
        std::fwrite(&states, sizeof(states), 1, file);
        std::fwrite(&dim, sizeof(dim), 1, file);
        std::fwrite(&mixtures, sizeof(mixtures), 1, file);
        std::fwrite(means_.data(), sizeof(float), means_.size(), file);
        std::fwrite(inv_variances_.data(), sizeof(float), inv_variances_.size(), file);
        std::fwrite(log_constants_.data(), sizeof(float), log_constants_.size(), file);
        std::fwrite(log_weights_.data(), sizeof(float), log_weights_.size(), file);
        std::fwrite(state_occupancy_.data(), sizeof(float), state_occupancy_.size(), file);
        std::fwrite(log_self_loop_.data(), sizeof(float), log_self_loop_.size(), file);
        std::fwrite(log_exit_.data(), sizeof(float), log_exit_.size(), file);
        std::fclose(file);
        return true;
    }

    bool load(const std::string& path) {
        FILE* file = std::fopen(path.c_str(), "rb");
        if (!file) return false;
        int32_t magic = 0, states = 0, dim = 0, mixtures = 1;
        bool ok = std::fread(&magic, sizeof(magic), 1, file) == 1 &&
                  std::fread(&states, sizeof(states), 1, file) == 1 &&
                  std::fread(&dim, sizeof(dim), 1, file) == 1 &&
                  std::fread(&mixtures, sizeof(mixtures), 1, file) == 1;
        if (!ok || magic != 0x4D414D32 || states != num_states() || dim != kFeatureDim ||
            mixtures < 1 || mixtures > kMaxMixtures) {
            std::fclose(file);
            return false;
        }
        resize();
        mixtures_ = mixtures;
        const size_t components = static_cast<size_t>(num_states()) * mixtures;
        means_.assign(components * kFeatureDim, 0.0f);
        inv_variances_.assign(components * kFeatureDim, 1.0f);
        log_constants_.assign(components, 0.0f);
        log_weights_.assign(components, 0.0f);
        ok = std::fread(means_.data(), sizeof(float), means_.size(), file) == means_.size() &&
             std::fread(inv_variances_.data(), sizeof(float), inv_variances_.size(), file) ==
                 inv_variances_.size() &&
             std::fread(log_constants_.data(), sizeof(float), log_constants_.size(), file) ==
                 log_constants_.size() &&
             std::fread(log_weights_.data(), sizeof(float), log_weights_.size(), file) ==
                 log_weights_.size() &&
             std::fread(state_occupancy_.data(), sizeof(float), state_occupancy_.size(), file) ==
                 state_occupancy_.size() &&
             std::fread(log_self_loop_.data(), sizeof(float), log_self_loop_.size(), file) ==
                 log_self_loop_.size() &&
             std::fread(log_exit_.data(), sizeof(float), log_exit_.size(), file) == log_exit_.size();
        std::fclose(file);
        return ok;
    }

private:
    void resize() {
        mixtures_ = 1;
        means_.assign(static_cast<size_t>(num_states()) * kFeatureDim, 0.0f);
        inv_variances_.assign(static_cast<size_t>(num_states()) * kFeatureDim, 1.0f);
        log_constants_.assign(num_states(), 0.0f);
        log_weights_.assign(num_states(), 0.0f);
        state_occupancy_.assign(num_states(), 0.0f);
        log_self_loop_.assign(num_phones(), std::log(0.6f));
        log_exit_.assign(num_phones(), std::log(0.4f));
    }

    int mixtures_ = 1;
    std::vector<float> means_;
    std::vector<float> inv_variances_;
    std::vector<float> log_constants_;
    std::vector<float> log_weights_;
    std::vector<float> state_occupancy_;
    std::vector<float> log_self_loop_;
    std::vector<float> log_exit_;

    // training accumulators
    std::vector<double> sum_;
    std::vector<double> sum_squares_;
    std::vector<double> counts_;
    std::vector<double> phone_frames_;
    std::vector<double> phone_segments_;
};

}  // namespace models
