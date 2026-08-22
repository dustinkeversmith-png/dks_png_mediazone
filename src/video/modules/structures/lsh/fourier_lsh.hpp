#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

namespace vision {

class FourierLSH {
public:
    int bits = 16;
    std::vector<std::vector<float>> planes;
    std::unordered_map<uint32_t, std::vector<int>> buckets;
    std::vector<std::vector<float>> stored;

    explicit FourierLSH(int n_bits = 16, int dim = 16, uint32_t seed = 7) : bits(n_bits) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        planes.assign(static_cast<size_t>(n_bits), std::vector<float>(static_cast<size_t>(dim)));
        for (auto& p : planes) {
            for (float& v : p) {
                v = dist(rng);
            }
        }
    }

    uint32_t hash(const std::vector<float>& v) const {
        uint32_t h = 0;
        for (int b = 0; b < bits; ++b) {
            const auto& p = planes[static_cast<size_t>(b)];
            double dot = 0.0;
            const size_t n = std::min(p.size(), v.size());
            for (size_t i = 0; i < n; ++i) {
                dot += static_cast<double>(p[i]) * v[i];
            }
            if (dot >= 0.0) {
                h |= (1u << static_cast<uint32_t>(b));
            }
        }
        return h;
    }

    void insert(int id, const std::vector<float>& v) {
        if (id >= static_cast<int>(stored.size())) {
            stored.resize(static_cast<size_t>(id) + 1);
        }
        stored[static_cast<size_t>(id)] = v;
        buckets[hash(v)].push_back(id);
    }

    std::vector<int> query(const std::vector<float>& v, int k = 5) const {
        std::vector<int> cand = buckets.count(hash(v)) ? buckets.at(hash(v)) : std::vector<int>{};
        // probe nearby Hamming balls of radius 1
        const uint32_t h = hash(v);
        for (int b = 0; b < bits; ++b) {
            const uint32_t h2 = h ^ (1u << static_cast<uint32_t>(b));
            auto it = buckets.find(h2);
            if (it != buckets.end()) {
                cand.insert(cand.end(), it->second.begin(), it->second.end());
            }
        }
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

        std::vector<std::pair<float, int>> scored;
        for (int id : cand) {
            if (id < 0 || id >= static_cast<int>(stored.size()) || stored[static_cast<size_t>(id)].empty()) {
                continue;
            }
            scored.push_back({l2(v, stored[static_cast<size_t>(id)]), id});
        }
        std::sort(scored.begin(), scored.end());
        std::vector<int> out;
        for (int i = 0; i < static_cast<int>(scored.size()) && i < k; ++i) {
            out.push_back(scored[static_cast<size_t>(i)].second);
        }
        return out;
    }

    static std::vector<int> exact_nn(const std::vector<std::vector<float>>& db, const std::vector<float>& q, int k) {
        std::vector<std::pair<float, int>> scored;
        for (int i = 0; i < static_cast<int>(db.size()); ++i) {
            if (db[static_cast<size_t>(i)].empty()) {
                continue;
            }
            scored.push_back({l2(q, db[static_cast<size_t>(i)]), i});
        }
        std::sort(scored.begin(), scored.end());
        std::vector<int> out;
        for (int i = 0; i < static_cast<int>(scored.size()) && i < k; ++i) {
            out.push_back(scored[static_cast<size_t>(i)].second);
        }
        return out;
    }

    static float l2(const std::vector<float>& a, const std::vector<float>& b) {
        const size_t n = std::min(a.size(), b.size());
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(a[i] - b[i]);
            s += d * d;
        }
        return static_cast<float>(std::sqrt(s));
    }
};

}  // namespace vision
