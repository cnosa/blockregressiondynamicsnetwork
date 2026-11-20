#include "blockdynamicnetwork/partition.hpp"

#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <utility>

namespace bdn {

Partition::Partition(std::vector<int> labels)
    : z_(std::move(labels)) {

    if (z_.empty()) {
        throw std::invalid_argument("Partition cannot be empty.");
    }

    for (int x : z_) {
        if (x < 0) {
            throw std::invalid_argument(
                "Community labels must be non-negative."
            );
        }
    }

    canonicalize();
}

std::size_t Partition::size() const noexcept {
    return z_.size();
}

int Partition::label(std::size_t node) const {
    if (node >= z_.size()) {
        throw std::out_of_range(
            "Node index outside partition bounds."
        );
    }

    return z_[node];
}

const std::vector<int>& Partition::labels() const noexcept {
    return z_;
}

int Partition::number_of_communities() const noexcept {
    return static_cast<int>(
        *std::max_element(z_.begin(), z_.end()) + 1
    );
}

std::vector<std::size_t> Partition::community_sizes() const {
    std::vector<std::size_t> sizes(
        static_cast<std::size_t>(number_of_communities()),
        0
    );

    for (int k : z_) {
        ++sizes[static_cast<std::size_t>(k)];
    }

    return sizes;
}

void Partition::canonicalize() {
    std::unordered_map<int, int> mapping;
    int next = 0;

    for (int& k : z_) {
        auto it = mapping.find(k);

        if (it == mapping.end()) {
            it = mapping.emplace(k, next++).first;
        }

        k = it->second;
    }
}

}