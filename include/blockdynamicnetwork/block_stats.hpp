
#pragma once

#include "blockdynamicnetwork/graph.hpp"
#include "blockdynamicnetwork/partition.hpp"

#include <vector>
#include <cstddef>

namespace bdn {

class BlockStats {
public:
    BlockStats(const Graph& graph, const Partition& partition);

    int number_of_communities() const noexcept;

    const std::vector<std::size_t>& community_sizes() const noexcept;

    std::size_t edge_count(int r, int s) const;

    std::size_t possible_edge_count(int r, int s) const;
    std::size_t non_edge_count(int r, int s) const;

    const std::vector<std::vector<std::size_t>>&
    edge_counts() const noexcept;

private:
    std::vector<std::size_t> sizes_;
    std::vector<std::vector<std::size_t>> edges_;

    void compute(const Graph& graph, const Partition& partition);
};

}
