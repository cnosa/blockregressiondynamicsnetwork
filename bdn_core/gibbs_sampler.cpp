#include "blockdynamicnetwork/gibbs_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <optional>

namespace bdn {

GibbsSampler::GibbsSampler(
    const PosteriorModel& posterior,
    std::size_t iterations,
    std::size_t burn_in,
    std::size_t thinning,
    std::uint64_t seed
)
    : posterior_(posterior),
      iterations_(iterations),
      burn_in_(burn_in),
      thinning_(thinning),
      rng_(seed),
      samples_(),
      community_counts_(),
      final_partition_(std::nullopt)
{
    if (iterations_ == 0) {
        throw std::invalid_argument(
            "GibbsSampler requires iterations > 0."
        );
    }

    if (burn_in_ >= iterations_) {
        throw std::invalid_argument(
            "Burn-in must be smaller than iterations."
        );
    }

    if (thinning_ == 0) {
        throw std::invalid_argument(
            "Thinning must be at least 1."
        );
    }
}


// Sample from log-weights
std::size_t GibbsSampler::sample_from_log_weights(
    const std::vector<double>& log_weights
) {
    const double max_log_weight =
        *std::max_element(
            log_weights.begin(),
            log_weights.end()
        );

    if (!std::isfinite(max_log_weight)) {
        throw std::runtime_error(
            "All Gibbs candidate probabilities are zero."
        );
    }

    std::vector<double> weights;
    weights.reserve(log_weights.size());

    double total = 0.0;

    for (double value : log_weights) {

        const double weight = std::isfinite(value)
            ? std::exp(value - max_log_weight)
            : 0.0;

        weights.push_back(weight);
        total += weight;
    }

    std::uniform_real_distribution<double> uniform(
        0.0,
        total
    );

    const double draw = uniform(rng_);

    double cumulative = 0.0;

    for (std::size_t i = 0; i < weights.size(); ++i) {

        cumulative += weights[i];

        if (draw < cumulative) {
            return i;
        }
    }

    return weights.size() - 1;
}

// Run Gibbs sampler
void GibbsSampler::run(
    const Partition& initial_partition
) {
    samples_.clear();
    community_counts_.clear();
    final_partition_.reset();

    std::vector<int> z = initial_partition.labels();

    const std::size_t n = z.size();

    for (std::size_t iteration = 0;
         iteration < iterations_;
         ++iteration) {

        // Random permutation of nodes
        std::vector<std::size_t> order(n);

        for (std::size_t i = 0; i < n; ++i) {
            order[i] = i;
        }

        std::shuffle(
            order.begin(),
            order.end(),
            rng_
        );

        // Update each node
        for (std::size_t node : order) {

            // Remove current node temporarily
            std::vector<int> base_z = z;

            base_z[node] = -1;

            // Identify existing communities
            std::vector<int> existing;

            for (int label : base_z) {

                if (label >= 0 &&
                    std::find(
                        existing.begin(),
                        existing.end(),
                        label
                    ) == existing.end()) {

                    existing.push_back(label);
                }
            }

            // Candidate assignments:
            // existing communities + one new community
            const std::size_t K = existing.size();

            std::vector<double> log_weights;
            log_weights.reserve(K + 1);

            // Existing communities
            for (int community : existing) {

                std::vector<int> candidate_z = base_z;
                candidate_z[node] = community;

                Partition candidate(candidate_z);

                log_weights.push_back(
                    posterior_.log_probability(candidate)
                );
            }

            // New community
            std::vector<int> candidate_z = base_z;

            int new_label = 0;

            if (!existing.empty()) {
                new_label =
                    *std::max_element(
                        existing.begin(),
                        existing.end()
                    ) + 1;
            }

            candidate_z[node] = new_label;

            Partition new_candidate(candidate_z);

            log_weights.push_back(
                posterior_.log_probability(new_candidate)
            );

            // Sample new assignment
            const std::size_t selected =
                sample_from_log_weights(log_weights);

            if (selected < K) {
                z[node] = existing[selected];
            }
            else {
                z[node] = new_label;
            }
        }

        // Canonicalize partition labels
        Partition current_partition(z);

        z = current_partition.labels();

        // Save after burn-in and thinning
        if (iteration >= burn_in_ &&
            ((iteration - burn_in_) % thinning_ == 0)) {

            samples_.push_back(current_partition);

            community_counts_.push_back(
                current_partition.number_of_communities()
            );
        }
    }

    final_partition_.emplace(z);
}


// Access stored samples
const std::vector<Partition>& GibbsSampler::samples() const {
    return samples_;
}


// Number of communities in each saved sample
const std::vector<std::size_t>&
GibbsSampler::number_of_communities() const {
    return community_counts_;
}


// Final partition
const Partition& GibbsSampler::final_partition() const {

    if (!final_partition_.has_value()) {
        throw std::runtime_error(
            "GibbsSampler has not been run yet."
        );
    }

    return final_partition_.value();
}


// Total iterations
std::size_t GibbsSampler::iterations() const {
    return iterations_;
}

} // namespace bdn