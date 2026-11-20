#ifndef BDN_GIBBS_SAMPLER_HPP
#define BDN_GIBBS_SAMPLER_HPP

#include "blockdynamicnetwork/partition.hpp"
#include "blockdynamicnetwork/posterior.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <optional>

namespace bdn {

class GibbsSampler {
public:

    GibbsSampler(
        const PosteriorModel& posterior,
        std::size_t iterations,
        std::size_t burn_in = 0,
        std::size_t thinning = 1,
        std::uint64_t seed = 123
    );

    // Run the sampler from an initial partition
    void run(const Partition& initial_partition);

    // Access stored samples
    const std::vector<Partition>& samples() const;

    // Number of communities at each saved iteration
    const std::vector<std::size_t>& number_of_communities() const;

    // Final partition
    const Partition& final_partition() const;

    // Total iterations
    std::size_t iterations() const;

private:

    const PosteriorModel& posterior_;

    std::size_t iterations_;
    std::size_t burn_in_;
    std::size_t thinning_;

    std::mt19937_64 rng_;

    std::vector<Partition> samples_;
    std::vector<std::size_t> community_counts_;

    std::optional<Partition> final_partition_;

    // Sample from log-weights
    std::size_t sample_from_log_weights(
        const std::vector<double>& log_weights
    );

};

} // namespace bdn

#endif