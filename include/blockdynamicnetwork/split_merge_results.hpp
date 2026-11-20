#pragma once

#include <vector>
#include <cstddef>

namespace bdn {

struct SplitMergeResults {

    // Saved partition samples
    std::vector<std::vector<int>> samples;

    // Full MCMC traces (one entry per iteration)
    std::vector<int> k_trace;
    std::vector<double> log_posterior_trace;

    // Move attempt indicators (0 = not attempted, 1 = attempted)
    std::vector<int> split_attempt_trace;
    std::vector<int> merge_attempt_trace;

    // Move acceptance indicators (0 = rejected, 1 = accepted)
    std::vector<int> split_acceptance_trace;
    std::vector<int> merge_acceptance_trace;

    // Acceptance rates
    double split_acceptance_rate = 0.0;
    double merge_acceptance_rate = 0.0;

    // Effective sample sizes
    double ess_k = 0.0;
    double ess_log_posterior = 0.0;

    // Run information
    std::size_t iterations = 0;
    std::size_t burn_in = 0;
    std::size_t thinning = 1;
    std::size_t saved_samples = 0;

    // Final partition
    std::vector<int> final_partition;
};

} // namespace bdn