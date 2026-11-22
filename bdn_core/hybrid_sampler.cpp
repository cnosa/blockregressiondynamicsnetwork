#include "blockdynamicnetwork/hybrid_sampler.hpp"
#include "blockdynamicnetwork/mcmc_diagnostics.hpp"

#include <stdexcept>
#include <random>
#include <cstdint>
#include <numeric>
#include <vector>
#include <algorithm>

namespace bdn {

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

HybridSampler::HybridSampler(
    const PosteriorModel& posterior,
    std::size_t iterations,
    std::size_t burn_in,
    std::size_t thinning,
    std::uint64_t seed,
    double gibbs_probability
)
    : posterior_(posterior),
      iterations_(iterations),
      burn_in_(burn_in),
      thinning_(thinning),
      gibbs_probability_(gibbs_probability),
      rng_(seed),
      samples_(),
      community_counts_(),
      final_partition_(std::nullopt),
      gibbs_iterations_(0),
      split_merge_iterations_(0),
      results_()
{
    if (iterations_ == 0) {
        throw std::invalid_argument(
            "HybridSampler requires iterations > 0."
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

    if (gibbs_probability_ < 0.0 ||
        gibbs_probability_ > 1.0) {

        throw std::invalid_argument(
            "Gibbs probability must be between 0 and 1."
        );
    }
}


// ------------------------------------------------------------
// Select sampler
// ------------------------------------------------------------
//
// Returns true  -> Gibbs
// Returns false -> Split-Merge
//
// ------------------------------------------------------------

bool HybridSampler::select_gibbs() {

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    return uniform(rng_) < gibbs_probability_;
}


// ------------------------------------------------------------
// Run Hybrid Sampler
// ------------------------------------------------------------

void HybridSampler::run(
    const Partition& initial_partition
) {

    // --------------------------------------------------------
    // Reset previous run
    // --------------------------------------------------------

    samples_.clear();
    community_counts_.clear();

    final_partition_.reset();

    gibbs_iterations_ = 0;
    split_merge_iterations_ = 0;

    results_ = HybridSamplerResults{};

    results_.k_trace.reserve(iterations_);
    results_.log_posterior_trace.reserve(iterations_);
    results_.sampler_selection_trace.reserve(iterations_);

    results_.split_attempt_trace.reserve(iterations_);
    results_.merge_attempt_trace.reserve(iterations_);

    results_.split_acceptance_trace.reserve(iterations_);
    results_.merge_acceptance_trace.reserve(iterations_);


    // --------------------------------------------------------
    // Initial state
    // --------------------------------------------------------

    Partition current = initial_partition;

    const std::size_t n = current.size();

    if (n < 2) {
        throw std::invalid_argument(
            "HybridSampler requires at least 2 nodes."
        );
    }

    if (posterior_.graph().size() != n) {
        throw std::invalid_argument(
            "Partition size does not match the posterior graph size."
        );
    }


    // --------------------------------------------------------
    // Main hybrid iterations
    // --------------------------------------------------------

    for (std::size_t iteration = 0;
         iteration < iterations_;
         ++iteration) {

        // ----------------------------------------------------
        // Initialize iteration statistics
        // ----------------------------------------------------

        int selected_sampler = 0;

        int split_attempted = 0;
        int merge_attempted = 0;

        int split_accepted = 0;
        int merge_accepted = 0;


        // ----------------------------------------------------
        // Select and execute sampler
        // ----------------------------------------------------

        if (select_gibbs()) {

            // =================================================
            // GIBBS SAMPLER
            // =================================================

            ++gibbs_iterations_;

            const std::uint64_t sampler_seed = rng_();

            GibbsSampler sampler(
                posterior_,
                1,              // One iteration
                0,              // No burn-in
                1,              // No thinning
                sampler_seed
            );

            sampler.run(current);

            // Update current partition
            current = sampler.final_partition();

            selected_sampler = 0;

        }
        else {

            // =================================================
            // SPLIT-MERGE SAMPLER
            // =================================================

            ++split_merge_iterations_;

            const std::uint64_t sampler_seed = rng_();

            SplitMergeSampler sampler(
                posterior_,
                1,              // One iteration
                0,              // No burn-in
                1,              // No thinning
                sampler_seed,
                false
            );

            sampler.run(current);

            // Update current partition
            current = sampler.final_partition();

            selected_sampler = 1;


            // -------------------------------------------------
            // Extract Split-Merge statistics safely
            // -------------------------------------------------

            const auto& sm_results = sampler.results();

            if (!sm_results.split_attempt_trace.empty()) {
                split_attempted =
                    sm_results.split_attempt_trace.front();
            }

            if (!sm_results.merge_attempt_trace.empty()) {
                merge_attempted =
                    sm_results.merge_attempt_trace.front();
            }

            if (!sm_results.split_acceptance_trace.empty()) {
                split_accepted =
                    sm_results.split_acceptance_trace.front();
            }

            if (!sm_results.merge_acceptance_trace.empty()) {
                merge_accepted =
                    sm_results.merge_acceptance_trace.front();
            }
        }


        // --------------------------------------------------------
        // Store iteration traces
        // --------------------------------------------------------

        results_.sampler_selection_trace.push_back(
            selected_sampler
        );

        results_.k_trace.push_back(
            static_cast<int>(
                current.number_of_communities()
            )
        );

        results_.log_posterior_trace.push_back(
            posterior_.log_probability(current)
        );

        results_.split_attempt_trace.push_back(
            split_attempted
        );

        results_.merge_attempt_trace.push_back(
            merge_attempted
        );

        results_.split_acceptance_trace.push_back(
            split_accepted
        );

        results_.merge_acceptance_trace.push_back(
            merge_accepted
        );


        // --------------------------------------------------------
        // Save posterior samples
        // --------------------------------------------------------

        if (iteration >= burn_in_ &&
            ((iteration - burn_in_) % thinning_ == 0)) {

            samples_.push_back(current);

            community_counts_.push_back(
                current.number_of_communities()
            );

            results_.samples.push_back(
                current.labels()
            );
        }
    }


    // ------------------------------------------------------------
    // Store final partition
    // ------------------------------------------------------------

    final_partition_.emplace(current);

    results_.iterations = iterations_;
    results_.burn_in = burn_in_;
    results_.thinning = thinning_;

    results_.gibbs_iterations = gibbs_iterations_;
    results_.split_merge_iterations = split_merge_iterations_;

    results_.saved_samples = samples_.size();
    results_.gibbs_probability = gibbs_probability_;

    results_.final_partition = current.labels();


    // ------------------------------------------------------------
    // Split-Merge acceptance rates
    // ------------------------------------------------------------

    const int total_split_attempts =
        std::accumulate(
            results_.split_attempt_trace.begin(),
            results_.split_attempt_trace.end(),
            0
        );

    const int total_merge_attempts =
        std::accumulate(
            results_.merge_attempt_trace.begin(),
            results_.merge_attempt_trace.end(),
            0
        );

    const int total_split_acceptances =
        std::accumulate(
            results_.split_acceptance_trace.begin(),
            results_.split_acceptance_trace.end(),
            0
        );

    const int total_merge_acceptances =
        std::accumulate(
            results_.merge_acceptance_trace.begin(),
            results_.merge_acceptance_trace.end(),
            0
        );


    // ------------------------------------------------------------
    // Calculate acceptance rates
    // ------------------------------------------------------------

    results_.split_acceptance_rate =
        (total_split_attempts > 0)
        ? static_cast<double>(total_split_acceptances)
            / total_split_attempts
        : 0.0;

    results_.merge_acceptance_rate =
        (total_merge_attempts > 0)
        ? static_cast<double>(total_merge_acceptances)
            / total_merge_attempts
        : 0.0;


    // ------------------------------------------------------------
    // Calculate ESS
    // ------------------------------------------------------------

    std::vector<double> k_trace_post_burn;
    std::vector<double> log_posterior_post_burn;

    k_trace_post_burn.reserve(
        iterations_ - burn_in_
    );

    log_posterior_post_burn.reserve(
        iterations_ - burn_in_
    );


    for (std::size_t iteration = burn_in_;
         iteration < iterations_;
         ++iteration) {

        k_trace_post_burn.push_back(
            static_cast<double>(
                results_.k_trace[iteration]
            )
        );

        log_posterior_post_burn.push_back(
            results_.log_posterior_trace[iteration]
        );
    }


    // ------------------------------------------------------------
    // Effective Sample Size
    // ------------------------------------------------------------

    results_.ess_k =
        effective_sample_size(
            k_trace_post_burn
        );

    results_.ess_log_posterior =
        effective_sample_size(
            log_posterior_post_burn
        );
}


// ------------------------------------------------------------
// Access stored samples
// ------------------------------------------------------------

const std::vector<Partition>&
HybridSampler::samples() const {

    return samples_;
}


// ------------------------------------------------------------
// Number of communities in each saved sample
// ------------------------------------------------------------

const std::vector<std::size_t>&
HybridSampler::number_of_communities() const {

    return community_counts_;
}


// ------------------------------------------------------------
// Final partition
// ------------------------------------------------------------

const Partition&
HybridSampler::final_partition() const {

    if (!final_partition_.has_value()) {
        throw std::runtime_error(
            "HybridSampler has not been run yet."
        );
    }

    return final_partition_.value();
}


// ------------------------------------------------------------
// Total iterations
// ------------------------------------------------------------

std::size_t
HybridSampler::iterations() const {

    return iterations_;
}


// ------------------------------------------------------------
// Number of Gibbs iterations
// ------------------------------------------------------------

std::size_t
HybridSampler::gibbs_iterations() const {

    return gibbs_iterations_;
}


// ------------------------------------------------------------
// Number of Split-Merge iterations
// ------------------------------------------------------------

std::size_t
HybridSampler::split_merge_iterations() const {

    return split_merge_iterations_;
}


// ------------------------------------------------------------
// Gibbs probability
// ------------------------------------------------------------

double
HybridSampler::gibbs_probability() const {

    return gibbs_probability_;
}


// ------------------------------------------------------------
// Results
// ------------------------------------------------------------

const HybridSamplerResults&
HybridSampler::results() const {

    return results_;
}

} // namespace bdn