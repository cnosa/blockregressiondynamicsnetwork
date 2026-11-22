#ifndef BDN_HYBRID_SAMPLER_HPP
#define BDN_HYBRID_SAMPLER_HPP

#include "blockdynamicnetwork/partition.hpp"
#include "blockdynamicnetwork/posterior.hpp"
#include "blockdynamicnetwork/gibbs_sampler.hpp"
#include "blockdynamicnetwork/split_merge_sampler.hpp"
#include "blockdynamicnetwork/hybrid_sampler_results.hpp" 

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <optional>

namespace bdn {

class HybridSampler {
public:

    HybridSampler(
        const PosteriorModel& posterior,
        std::size_t iterations,
        std::size_t burn_in = 0,
        std::size_t thinning = 1,
        std::uint64_t seed = 123,
        double gibbs_probability = 0.75
    );

    // Run the hybrid sampler from an initial partition
    void run(const Partition& initial_partition);

    // Access stored posterior samples
    const std::vector<Partition>& samples() const;

    // Number of communities at each saved iteration
    const std::vector<std::size_t>& number_of_communities() const;

    // Final partition
    const Partition& final_partition() const;

    // Total iterations
    std::size_t iterations() const;

    // Number of iterations assigned to each sampler
    std::size_t gibbs_iterations() const;
    std::size_t split_merge_iterations() const;

    // Gibbs selection probability
    double gibbs_probability() const;

    // Hybrid sampler results // NEW
    const HybridSamplerResults& results() const; // NEW

private:

    const PosteriorModel& posterior_;

    std::size_t iterations_;
    std::size_t burn_in_;
    std::size_t thinning_;

    double gibbs_probability_;

    std::mt19937_64 rng_;

    std::vector<Partition> samples_;
    std::vector<std::size_t> community_counts_;

    std::optional<Partition> final_partition_;

    std::size_t gibbs_iterations_;
    std::size_t split_merge_iterations_;

    HybridSamplerResults results_; // NEW

    // Randomly decide which sampler to execute
    bool select_gibbs();

};

} // namespace bdn

#endif