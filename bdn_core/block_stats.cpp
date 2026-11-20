
#include "blockdynamicnetwork/block_stats.hpp"

#include <stdexcept>

namespace bdn {

BlockStats::BlockStats(
    const Graph& graph,
    const Partition& partition
) {
    if (graph.size() != partition.size()) {
        throw std::invalid_argument(
            "Graph and partition must have the same number of nodes."
        );
    }

    compute(graph, partition);
}

int BlockStats::number_of_communities() const noexcept {
    return static_cast<int>(sizes_.size());
}

const std::vector<std::size_t>&
BlockStats::community_sizes() const noexcept {
    return sizes_;
}

std::size_t BlockStats::edge_count(int r, int s) const {
    if (r < 0 || s < 0 ||
        r >= number_of_communities() ||
        s >= number_of_communities()) {
        throw std::out_of_range(
            "Community index outside bounds."
        );
    }

    return edges_[r][s];
}


const std::vector<std::vector<std::size_t>>&
BlockStats::edge_counts() const noexcept {
    return edges_;
}

std::size_t BlockStats::possible_edge_count(int r, int s) const {
    if (r < 0 || s < 0 ||
        r >= number_of_communities() ||
        s >= number_of_communities()) {
        throw std::out_of_range("Community index outside bounds.");
    }

    if (r != s) {
        return sizes_[r] * sizes_[s];
    }

    return sizes_[r] * (sizes_[r] - 1) / 2;
}

std::size_t BlockStats::non_edge_count(int r, int s) const {
    return possible_edge_count(r, s) - edge_count(r, s);
}

void BlockStats::compute(
    const Graph& graph,
    const Partition& partition
) {
    const int K = partition.number_of_communities();

    sizes_.assign(K, 0);

    edges_.assign(
        K,
        std::vector<std::size_t>(K, 0)
    );

    // 1. Community sizes
    for (std::size_t i = 0; i < partition.size(); ++i) {
        int r = partition.label(i);
        ++sizes_[r];
    }

    // 2. Count undirected edges between communities
    for (std::size_t u = 0; u < graph.size(); ++u) {
        int r = partition.label(u);

        for (std::size_t v : graph.neighbors(u)) {

            // Count each undirected edge only once
            if (u < v) {
                int s = partition.label(v);

                ++edges_[r][s];
                if (r != s) {
                    ++edges_[s][r];
                }
            }
        }
    }
}

}
