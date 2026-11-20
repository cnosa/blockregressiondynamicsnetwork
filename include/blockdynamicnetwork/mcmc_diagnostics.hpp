#pragma once

#include <vector>
#include <cstddef>

namespace bdn {

// Estimate the effective sample size of an MCMC trace.
double effective_sample_size(
    const std::vector<double>& trace,
    std::size_t max_lag = 1000
);

} // namespace bdn