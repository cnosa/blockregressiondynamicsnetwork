
#include "blockdynamicnetwork/sbm.hpp"

#include <cmath>
#include <stdexcept>

namespace bdn {

SBMModel::SBMModel(const Graph& graph, const Partition& partition)
    : graph_(graph) {

    if (graph.size() != partition.size()) {
        throw std::invalid_argument(
            "Graph and partition must have the same size."
        );
    }
}

double SBMModel::log_likelihood(
    const Partition& partition,
    const std::vector<std::vector<double>>& theta
) const {

    if (partition.size() != graph_.size()) {
        throw std::invalid_argument(
            "Partition size must match graph size."
        );
    }

    BlockStats stats(graph_, partition);

    const int K = stats.number_of_communities();

    if (static_cast<int>(theta.size()) != K) {
        throw std::invalid_argument(
            "Theta has incorrect dimensions."
        );
    }

    for (const auto& row : theta) {
        if (static_cast<int>(row.size()) != K) {
            throw std::invalid_argument(
                "Theta must be a square K x K matrix."
            );
        }
    }

    double result = 0.0;

    for (int r = 0; r < K; ++r) {
        for (int s = r; s < K; ++s) {

            const double p = theta[r][s];

            if (p <= 0.0 || p >= 1.0) {
                throw std::invalid_argument(
                    "Theta probabilities must be strictly between 0 and 1."
                );
            }

            const double e = static_cast<double>(
                stats.edge_count(r, s)
            );

            const double f = static_cast<double>(
                stats.non_edge_count(r, s)
            );

            result += e * std::log(p)
                    + f * std::log1p(-p);
        }
    }

    return result;
}


double SBMModel::log_collapsed_likelihood(
    const Partition& partition,
    double a,
    double b
) const {

    if (partition.size() != graph_.size()) {
        throw std::invalid_argument(
            "Partition size must match graph size."
        );
    }

    if (a <= 0.0 || b <= 0.0) {
        throw std::invalid_argument(
            "Beta prior parameters must be positive."
        );
    }

    BlockStats stats(graph_, partition);

    const int K = stats.number_of_communities();

    double result = 0.0;

    // log Beta(a,b)
    const double log_beta_prior =
        std::lgamma(a) +
        std::lgamma(b) -
        std::lgamma(a + b);

    // Sum over block pairs r <= s
    for (int r = 0; r < K; ++r) {
        for (int s = r; s < K; ++s) {

            const double e = static_cast<double>(
                stats.edge_count(r, s)
            );

            const double f = static_cast<double>(
                stats.non_edge_count(r, s)
            );

            result +=
                std::lgamma(e + a) +
                std::lgamma(f + b) -
                std::lgamma(e + f + a + b) -
                log_beta_prior;
        }
    }

    return result;
}


}
