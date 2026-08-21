#pragma once

#include "../../math/vision_types.hpp"

#include <functional>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace vision {

template <typename T>
class VPTree {
public:
    using Distance = std::function<float(const T&, const T&)>;

    void build(std::vector<T> items, Distance distance) {
        data_ = std::move(items);
        dist_ = std::move(distance);
        nodes_.clear();
        if (data_.empty()) {
            root_ = -1;
            return;
        }
        std::vector<int> idx(data_.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            idx[i] = static_cast<int>(i);
        }
        std::mt19937 rng(13);
        root_ = build_node(idx, rng);
    }

    int nearest(const T& query, float* out_dist = nullptr) const {
        if (root_ < 0) {
            return -1;
        }
        int best = -1;
        float best_d = std::numeric_limits<float>::max();
        search(root_, query, best, best_d);
        if (out_dist) {
            *out_dist = best_d;
        }
        return best;
    }

    const T& at(int i) const { return data_[static_cast<size_t>(i)]; }
    size_t size() const { return data_.size(); }

    static float chamfer(const std::vector<Vec2>& a, const std::vector<Vec2>& b) {
        if (a.empty() || b.empty()) {
            return 1.0e9f;
        }
        auto one_way = [](const std::vector<Vec2>& u, const std::vector<Vec2>& v) {
            double s = 0.0;
            for (const auto& p : u) {
                float best = 1.0e12f;
                for (const auto& q : v) {
                    best = std::min(best, dist2(p, q));
                }
                s += std::sqrt(static_cast<double>(best));
            }
            return static_cast<float>(s / static_cast<double>(u.size()));
        };
        return 0.5f * (one_way(a, b) + one_way(b, a));
    }

    static float hausdorff(const std::vector<Vec2>& a, const std::vector<Vec2>& b) {
        if (a.empty() || b.empty()) {
            return 1.0e9f;
        }
        auto directed = [](const std::vector<Vec2>& u, const std::vector<Vec2>& v) {
            float worst = 0.0f;
            for (const auto& p : u) {
                float best = 1.0e12f;
                for (const auto& q : v) {
                    best = std::min(best, dist(p, q));
                }
                worst = std::max(worst, best);
            }
            return worst;
        };
        return std::max(directed(a, b), directed(b, a));
    }

private:
    struct Node {
        int index = -1;
        float threshold = 0.0f;
        int left = -1;
        int right = -1;
    };

    std::vector<T> data_;
    std::vector<Node> nodes_;
    Distance dist_;
    int root_ = -1;

    int build_node(std::vector<int>& idx, std::mt19937& rng) {
        if (idx.empty()) {
            return -1;
        }
        Node node;
        std::uniform_int_distribution<int> pick(0, static_cast<int>(idx.size()) - 1);
        const int swap_i = pick(rng);
        std::swap(idx[0], idx[static_cast<size_t>(swap_i)]);
        node.index = idx[0];
        if (idx.size() == 1) {
            nodes_.push_back(node);
            return static_cast<int>(nodes_.size() - 1);
        }
        std::vector<std::pair<float, int>> dists;
        dists.reserve(idx.size() - 1);
        for (size_t i = 1; i < idx.size(); ++i) {
            dists.push_back({dist_(data_[static_cast<size_t>(node.index)], data_[static_cast<size_t>(idx[i])]), idx[i]});
        }
        std::sort(dists.begin(), dists.end());
        const size_t mid = dists.size() / 2;
        node.threshold = dists[mid].first;
        std::vector<int> left, right;
        for (size_t i = 0; i < dists.size(); ++i) {
            if (i < mid) {
                left.push_back(dists[i].second);
            } else {
                right.push_back(dists[i].second);
            }
        }
        nodes_.push_back(node);
        const int self = static_cast<int>(nodes_.size() - 1);
        const int L = build_node(left, rng);
        const int R = build_node(right, rng);
        nodes_[static_cast<size_t>(self)].left = L;
        nodes_[static_cast<size_t>(self)].right = R;
        return self;
    }

    void search(int ni, const T& q, int& best, float& best_d) const {
        if (ni < 0) {
            return;
        }
        const Node& node = nodes_[static_cast<size_t>(ni)];
        const float d = dist_(q, data_[static_cast<size_t>(node.index)]);
        if (d < best_d) {
            best_d = d;
            best = node.index;
        }
        if (node.left < 0 && node.right < 0) {
            return;
        }
        if (d < node.threshold) {
            search(node.left, q, best, best_d);
            if (d + best_d >= node.threshold) {
                search(node.right, q, best, best_d);
            }
        } else {
            search(node.right, q, best, best_d);
            if (d - best_d <= node.threshold) {
                search(node.left, q, best, best_d);
            }
        }
    }
};

}  // namespace vision
