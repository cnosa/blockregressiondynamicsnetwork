#pragma once

#include <vector>
#include <cstddef>

namespace bdn {

struct HybridSamplerResults {

    // --------------------------------------------------------
    // Saved partition samples
    // --------------------------------------------------------

    std::vector<std::vector<int>> samples;


    // --------------------------------------------------------
    // Full MCMC traces (one entry per iteration)
    // --------------------------------------------------------

    std::vector<int> k_trace;

    std::vector<double> log_posterior_trace;


    // --------------------------------------------------------
    // Sampler selection indicators
    // 0 = Gibbs
    // 1 = Split-Merge
    // --------------------------------------------------------

    std::vector<int> sampler_selection_trace;


    // --------------------------------------------------------
    // Iteration counts for each sampler
    // --------------------------------------------------------

    std::size_t gibbs_iterations = 0;

    std::size_t split_merge_iterations = 0;


    // --------------------------------------------------------
    // Acceptance information from Split-Merge
    // --------------------------------------------------------

    std::vector<int> split_attempt_trace;

    std::vector<int> merge_attempt_trace;

    std::vector<int> split_acceptance_trace;

    std::vector<int> merge_acceptance_trace;


    // --------------------------------------------------------
    // Acceptance rates
    // --------------------------------------------------------

    double split_acceptance_rate = 0.0;

    double merge_acceptance_rate = 0.0;


    // --------------------------------------------------------
    // Effective sample sizes
    // --------------------------------------------------------

    double ess_k = 0.0;

    double ess_log_posterior = 0.0;


    // --------------------------------------------------------
    // Run information
    // --------------------------------------------------------

    std::size_t iterations = 0;

    std::size_t burn_in = 0;

    std::size_t thinning = 1;

    std::size_t saved_samples = 0;


    // --------------------------------------------------------
    // Hybrid configuration
    // --------------------------------------------------------

    double gibbs_probability = 0.75;


    // --------------------------------------------------------
    // Final partition
    // --------------------------------------------------------

    std::vector<int> final_partition;

};

} // namespace bdn