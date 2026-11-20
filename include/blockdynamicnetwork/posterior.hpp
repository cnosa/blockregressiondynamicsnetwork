#ifndef BDN_POSTERIOR_HPP
#define BDN_POSTERIOR_HPP

#include "blockdynamicnetwork/graph.hpp"
#include "blockdynamicnetwork/partition.hpp"
#include "blockdynamicnetwork/partition_prior.hpp"

namespace bdn {

class PosteriorModel {
public:

    // Constructor
    PosteriorModel(
        const Graph& graph,
        const PartitionPrior& prior,
        double a = 1.0,
        double b = 1.0
    );

    // Collapsed log-likelihood
    double log_likelihood(
        const Partition& partition
    ) const;

    // Log-prior of the partition
    double log_prior(
        const Partition& partition
    ) const;

    // Log-posterior (up to a constant)
    double log_probability(
        const Partition& partition
    ) const;

    // Beta prior parameters
    double a() const;
    double b() const;

private:

    const Graph& graph_;
    const PartitionPrior& prior_;

    double a_;
    double b_;
};

} // namespace bdn

#endif