
#pragma once

#include "blockdynamicnetwork/graph.hpp"
#include "blockdynamicnetwork/partition.hpp"
#include "blockdynamicnetwork/block_stats.hpp"

#include <vector>

namespace bdn {

class SBMModel {
public:
    SBMModel(const Graph& graph, const Partition& partition);

    double log_likelihood(
        const Partition& partition,
        const std::vector<std::vector<double>>& theta
    ) const;

    double log_collapsed_likelihood(
        const Partition& partition,
        double a = 1.0,
        double b = 1.0
    ) const;

private:
    const Graph& graph_;
};

}
