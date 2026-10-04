// Context-dependent triphones with phonetic decision-tree state tying
// (Path 3 - "senones").
//
// The problem this solves: a monophone gives /k/ one distribution, but /k/
// before /iy/ ("keep") and before /uw/ ("cool") have F2 loci hundreds of Hz
// apart. Frames from both are averaged into one Gaussian, which is why the
// monophone system's errors are 75 % substitutions.
//
// The naive fix does not work: 40^3 triphones x 3 states is ~192,000 states,
// far more than 5 hours of audio can estimate. The classical answer is to build
// one decision tree per (centre phone, state position) and split contexts with
// binary phonetic questions ("is the right context a front vowel?"), pooling
// every context that lands in the same leaf into a single tied state. Leaves
// with too little data are never created, so every senone is backed by real
// frames, and unseen triphones still reach a leaf by answering the questions -
// which is what makes the model usable on words it never saw in training.
//
// Splitting criterion (standard HTK/Kaldi form): for a diagonal Gaussian fitted
// to a pooled set of frames, the log-likelihood of that set is
//
//     L = -0.5 * N * sum_d (log var_d + 1 + log 2pi)
//
// so a split is worth making when L(yes) + L(no) - L(parent) is largest. Only
// counts, sums and sums of squares are needed, which are additive over contexts.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include <filter/mfcc.hpp>
#include "phone_set.hpp"

namespace models {

// Sufficient statistics for one context (left, centre, right, state).
struct ContextStats {
    int left = 0;
    int right = 0;
    double count = 0.0;
    std::vector<double> sum;
    std::vector<double> sum_squares;

    ContextStats() : sum(kFeatureDim, 0.0), sum_squares(kFeatureDim, 0.0) {}

    void add(const float* row) {
        for (int d = 0; d < kFeatureDim; ++d) {
            sum[d] += row[d];
            sum_squares[d] += static_cast<double>(row[d]) * row[d];
        }
        count += 1.0;
    }
};

// Phonetic questions: each is a set of phones. A context answers "yes" when the
// relevant neighbour is in the set. Broad classes come first so early splits
// are linguistically meaningful, then every phone as a singleton question.
inline const std::vector<std::pair<std::string, std::vector<std::string>>>& question_sets() {
    static const std::vector<std::pair<std::string, std::vector<std::string>>> questions = {
        {"silence", {"SIL"}},
        {"vowel", {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW",
                   "OY", "UH", "UW"}},
        {"front_vowel", {"IY", "IH", "EH", "AE", "EY"}},
        {"back_vowel", {"UW", "UH", "OW", "AO", "AA"}},
        {"central_vowel", {"AH", "ER"}},
        {"high_vowel", {"IY", "IH", "UW", "UH"}},
        {"low_vowel", {"AE", "AA", "AO", "AW"}},
        {"rounded", {"UW", "UH", "OW", "AO", "OY", "W"}},
        {"diphthong", {"AY", "EY", "OY", "AW", "OW"}},
        {"consonant", {"B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG",
                       "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"}},
        {"stop", {"B", "D", "G", "P", "T", "K"}},
        {"voiced_stop", {"B", "D", "G"}},
        {"unvoiced_stop", {"P", "T", "K"}},
        {"fricative", {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}},
        {"sibilant", {"S", "Z", "SH", "ZH", "CH", "JH"}},
        {"voiced_fricative", {"V", "DH", "Z", "ZH"}},
        {"affricate", {"CH", "JH"}},
        {"nasal", {"M", "N", "NG"}},
        {"liquid", {"L", "R"}},
        {"glide", {"W", "Y"}},
        {"approximant", {"L", "R", "W", "Y"}},
        {"labial", {"B", "P", "M", "F", "V", "W"}},
        {"alveolar", {"D", "T", "N", "S", "Z", "L", "R"}},
        {"velar", {"K", "G", "NG", "W"}},
        {"dental", {"TH", "DH"}},
        {"palatal", {"SH", "ZH", "CH", "JH", "Y"}},
        {"voiced", {"B", "D", "G", "V", "DH", "Z", "ZH", "JH", "M", "N", "NG", "L", "R", "W",
                    "Y"}},
    };
    return questions;
}

// Bitmask over the phone inventory, so a question is answered with one AND.
struct Question {
    std::string name;
    uint64_t mask = 0;
    bool on_left = true;

    bool answer(const ContextStats& context) const {
        const int phone = on_left ? context.left : context.right;
        return phone >= 0 && (mask >> phone) & 1ull;
    }
};

inline std::vector<Question> build_questions() {
    std::vector<Question> questions;
    for (int side = 0; side < 2; ++side) {
        for (const auto& entry : question_sets()) {
            Question question;
            question.name = (side == 0 ? "L:" : "R:") + entry.first;
            question.on_left = side == 0;
            for (const std::string& phone : entry.second) {
                const int id = phone_id(phone);
                if (id >= 0) question.mask |= 1ull << id;
            }
            if (question.mask) questions.push_back(question);
        }
        // Singleton questions let the tree isolate one specific context.
        for (int p = 0; p < num_phones(); ++p) {
            Question question;
            question.name = (side == 0 ? "L=" : "R=") + phone_names()[p];
            question.on_left = side == 0;
            question.mask = 1ull << p;
            questions.push_back(question);
        }
    }
    return questions;
}

// One decision tree node. Leaves carry a senone id.
struct TreeNode {
    int question = -1;   // index into the question list; -1 for a leaf
    int yes = -1;
    int no = -1;
    int senone = -1;
};

// Trees for every (centre phone, state position) pair, plus the senone table.
class TriphoneTree {
public:
    // ---- training -------------------------------------------------------

    // `contexts[(phone, state)]` holds the statistics of every observed
    // context. Splitting stops when a split would gain less than
    // `min_gain` or leave a leaf with fewer than `min_occupancy` frames.
    void build(const std::vector<std::vector<ContextStats>>& contexts, double min_gain = 1200.0,
               double min_occupancy = 600.0, int max_senones = 2200) {
        questions_ = build_questions();
        nodes_.clear();
        roots_.assign(contexts.size(), -1);
        num_senones_ = 0;

        // Grow every tree greedily, best-gain-first across all trees, so the
        // senone budget goes where it buys the most likelihood.
        struct Pending {
            int node;
            std::vector<const ContextStats*> members;
        };
        std::vector<Pending> frontier;

        for (size_t key = 0; key < contexts.size(); ++key) {
            if (contexts[key].empty()) continue;
            Pending pending;
            pending.node = static_cast<int>(nodes_.size());
            nodes_.push_back(TreeNode{});
            roots_[key] = pending.node;
            for (const ContextStats& stats : contexts[key]) {
                if (stats.count > 0.0) pending.members.push_back(&stats);
            }
            if (!pending.members.empty()) frontier.push_back(std::move(pending));
        }

        while (!frontier.empty() && num_senones_ + static_cast<int>(frontier.size()) < max_senones) {
            // Find the best split available anywhere in the frontier.
            double best_gain = min_gain;
            size_t best_index = frontier.size();
            int best_question = -1;

            for (size_t i = 0; i < frontier.size(); ++i) {
                int question = -1;
                const double gain = best_split(frontier[i].members, min_occupancy, &question);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_index = i;
                    best_question = question;
                }
            }
            if (best_index == frontier.size()) break;  // nothing worth splitting

            Pending parent = std::move(frontier[best_index]);
            frontier.erase(frontier.begin() + best_index);

            Pending yes_branch, no_branch;
            for (const ContextStats* stats : parent.members) {
                if (questions_[best_question].answer(*stats)) {
                    yes_branch.members.push_back(stats);
                } else {
                    no_branch.members.push_back(stats);
                }
            }

            yes_branch.node = static_cast<int>(nodes_.size());
            nodes_.push_back(TreeNode{});
            no_branch.node = static_cast<int>(nodes_.size());
            nodes_.push_back(TreeNode{});

            nodes_[parent.node].question = best_question;
            nodes_[parent.node].yes = yes_branch.node;
            nodes_[parent.node].no = no_branch.node;

            frontier.push_back(std::move(yes_branch));
            frontier.push_back(std::move(no_branch));
        }

        // Everything left in the frontier becomes a leaf.
        for (Pending& pending : frontier) {
            nodes_[pending.node].senone = num_senones_++;
        }
        // Trees that never entered the frontier (no data) still need a leaf.
        for (TreeNode& node : nodes_) {
            if (node.question < 0 && node.senone < 0) node.senone = num_senones_++;
        }
    }

    // ---- lookup ---------------------------------------------------------

    int senone(int left, int centre, int state, int right) const {
        const int key = tree_key(centre, state);
        if (key < 0 || key >= static_cast<int>(roots_.size()) || roots_[key] < 0) return 0;

        ContextStats probe;
        probe.left = left;
        probe.right = right;

        int node = roots_[key];
        while (nodes_[node].question >= 0) {
            node = questions_[nodes_[node].question].answer(probe) ? nodes_[node].yes
                                                                   : nodes_[node].no;
        }
        return nodes_[node].senone;
    }

    int num_senones() const { return num_senones_; }
    static int tree_key(int phone, int state) { return phone * kNumStatesPerPhone + state; }
    static int num_tree_keys() { return num_phones() * kNumStatesPerPhone; }

    // ---- persistence ----------------------------------------------------

    bool save(const std::string& path) const {
        FILE* file = std::fopen(path.c_str(), "wb");
        if (!file) return false;
        const int32_t magic = 0x54524931;  // "TRI1"
        const int32_t nodes = static_cast<int32_t>(nodes_.size());
        const int32_t roots = static_cast<int32_t>(roots_.size());
        const int32_t senones = num_senones_;
        std::fwrite(&magic, sizeof(magic), 1, file);
        std::fwrite(&nodes, sizeof(nodes), 1, file);
        std::fwrite(&roots, sizeof(roots), 1, file);
        std::fwrite(&senones, sizeof(senones), 1, file);
        std::fwrite(nodes_.data(), sizeof(TreeNode), nodes_.size(), file);
        std::fwrite(roots_.data(), sizeof(int), roots_.size(), file);
        std::fclose(file);
        return true;
    }

    bool load(const std::string& path) {
        FILE* file = std::fopen(path.c_str(), "rb");
        if (!file) return false;
        int32_t magic = 0, nodes = 0, roots = 0, senones = 0;
        if (std::fread(&magic, sizeof(magic), 1, file) != 1 ||
            std::fread(&nodes, sizeof(nodes), 1, file) != 1 ||
            std::fread(&roots, sizeof(roots), 1, file) != 1 ||
            std::fread(&senones, sizeof(senones), 1, file) != 1 || magic != 0x54524931) {
            std::fclose(file);
            return false;
        }
        nodes_.resize(nodes);
        roots_.resize(roots);
        const bool ok =
            std::fread(nodes_.data(), sizeof(TreeNode), nodes_.size(), file) == nodes_.size() &&
            std::fread(roots_.data(), sizeof(int), roots_.size(), file) == roots_.size();
        std::fclose(file);
        num_senones_ = senones;
        questions_ = build_questions();
        return ok;
    }

private:
    // Log-likelihood of a pooled set under a single diagonal Gaussian.
    static double pooled_likelihood(double count, const std::vector<double>& sum,
                                    const std::vector<double>& sum_squares) {
        if (count < 1.0) return 0.0;
        constexpr double kLog2Pi = 1.8378770664093453;
        double log_var = 0.0;
        for (int d = 0; d < kFeatureDim; ++d) {
            const double mean = sum[d] / count;
            double variance = sum_squares[d] / count - mean * mean;
            if (variance < 0.01) variance = 0.01;
            log_var += std::log(variance);
        }
        return -0.5 * count * (log_var + kFeatureDim * (1.0 + kLog2Pi));
    }

    static void accumulate(const std::vector<const ContextStats*>& members, double& count,
                           std::vector<double>& sum, std::vector<double>& sum_squares) {
        count = 0.0;
        sum.assign(kFeatureDim, 0.0);
        sum_squares.assign(kFeatureDim, 0.0);
        for (const ContextStats* stats : members) {
            count += stats->count;
            for (int d = 0; d < kFeatureDim; ++d) {
                sum[d] += stats->sum[d];
                sum_squares[d] += stats->sum_squares[d];
            }
        }
    }

    double best_split(const std::vector<const ContextStats*>& members, double min_occupancy,
                      int* out_question) const {
        if (members.size() < 2) return 0.0;

        double parent_count;
        std::vector<double> parent_sum, parent_squares;
        accumulate(members, parent_count, parent_sum, parent_squares);
        if (parent_count < 2.0 * min_occupancy) return 0.0;
        const double parent_likelihood =
            pooled_likelihood(parent_count, parent_sum, parent_squares);

        double best_gain = 0.0;
        std::vector<double> yes_sum(kFeatureDim), yes_squares(kFeatureDim);

        for (size_t q = 0; q < questions_.size(); ++q) {
            double yes_count = 0.0;
            std::fill(yes_sum.begin(), yes_sum.end(), 0.0);
            std::fill(yes_squares.begin(), yes_squares.end(), 0.0);

            for (const ContextStats* stats : members) {
                if (!questions_[q].answer(*stats)) continue;
                yes_count += stats->count;
                for (int d = 0; d < kFeatureDim; ++d) {
                    yes_sum[d] += stats->sum[d];
                    yes_squares[d] += stats->sum_squares[d];
                }
            }
            const double no_count = parent_count - yes_count;
            if (yes_count < min_occupancy || no_count < min_occupancy) continue;

            std::vector<double> no_sum(kFeatureDim), no_squares(kFeatureDim);
            for (int d = 0; d < kFeatureDim; ++d) {
                no_sum[d] = parent_sum[d] - yes_sum[d];
                no_squares[d] = parent_squares[d] - yes_squares[d];
            }

            const double gain = pooled_likelihood(yes_count, yes_sum, yes_squares) +
                                pooled_likelihood(no_count, no_sum, no_squares) -
                                parent_likelihood;
            if (gain > best_gain) {
                best_gain = gain;
                *out_question = static_cast<int>(q);
            }
        }
        return best_gain;
    }

    std::vector<Question> questions_;
    std::vector<TreeNode> nodes_;
    std::vector<int> roots_;
    int num_senones_ = 0;
};

}  // namespace models
