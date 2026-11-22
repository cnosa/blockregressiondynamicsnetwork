#ifndef BDN_SPLIT_MERGE_SAMPLER_HPP
#define BDN_SPLIT_MERGE_SAMPLER_HPP

#include "blockdynamicnetwork/partition.hpp"
#include "blockdynamicnetwork/posterior.hpp"
#include "blockdynamicnetwork/split_merge_results.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <optional>
#include <utility>

namespace bdn {

class SplitMergeSampler {
public:
    SplitMergeSampler(
        const PosteriorModel& posterior,
        std::size_t iterations,
        std::size_t burn_in = 0,
        std::size_t thinning = 1,
        std::uint64_t seed = 123,
        bool verbose = true
    );

    void run(const Partition& initial_partition);

    const std::vector<Partition>& samples() const;
    const std::vector<std::size_t>& number_of_communities() const;

    const Partition& final_partition() const;

    std::size_t iterations() const;
    std::size_t accepted_splits() const;
    std::size_t accepted_merges() const;

    const SplitMergeResults& results() const;

private:
    const PosteriorModel& posterior_;

    std::size_t iterations_;
    std::size_t burn_in_;
    std::size_t thinning_;

    std::mt19937_64 rng_;

    bool verbose_;

    std::vector<Partition> samples_;
    std::vector<std::size_t> community_counts_;

    std::optional<Partition> final_partition_;

    std::size_t accepted_splits_;
    std::size_t accepted_merges_;

    SplitMergeResults results_;

    std::pair<std::size_t, std::size_t> select_pivots(
        std::size_t n
    );

    double log_sum_exp(
        double a,
        double b
    );

    bool accept_move(double log_ratio);

    std::pair<Partition, double> propose_split(
        const Partition& current,
        std::size_t i,
        std::size_t j
    );

    Partition propose_merge(
        const Partition& current,
        std::size_t i,
        std::size_t j
    );

    double log_split_probability(
        const Partition& merged,
        const Partition& target_split,
        std::size_t i,
        std::size_t j
    );
};

} // namespace bdn

#endif