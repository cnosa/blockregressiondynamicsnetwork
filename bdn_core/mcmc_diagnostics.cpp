#include "blockdynamicnetwork/mcmc_diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

namespace bdn {

double effective_sample_size(
    const std::vector<double>& trace,
    std::size_t max_lag
) {
    const std::size_t n = trace.size();

    if (n < 2)
        return static_cast<double>(n);

    // Compute the sample mean.
    const double mean =
        std::accumulate(trace.begin(), trace.end(), 0.0) /
        static_cast<double>(n);

    // Compute variance.
    double variance = 0.0;

    for (double x : trace) {
        variance += (x - mean) * (x - mean);
    }

    variance /= static_cast<double>(n);

    // A constant trace contains no autocorrelation information.
    if (variance <= std::numeric_limits<double>::epsilon())
        return static_cast<double>(n);

    max_lag = std::min(max_lag, n - 1);

    double rho_sum = 0.0;
    double previous_pair_sum =
        std::numeric_limits<double>::infinity();

    // Compute autocorrelation in consecutive lag pairs.
    for (std::size_t lag = 1; lag <= max_lag; lag += 2) {

        double pair_sum = 0.0;

        for (std::size_t offset = 0; offset < 2; ++offset) {

            const std::size_t current_lag = lag + offset;

            if (current_lag > max_lag)
                break;

            double covariance = 0.0;

            for (std::size_t t = 0;
                 t < n - current_lag;
                 ++t) {

                covariance +=
                    (trace[t] - mean) *
                    (trace[t + current_lag] - mean);
            }

            covariance /= static_cast<double>(n);

            const double rho = covariance / variance;

            pair_sum += rho;
        }

        // Stop at the first non-positive pair.
        if (pair_sum <= 0.0)
            break;

        // Enforce a monotonically decreasing sequence of pair sums.
        pair_sum = std::min(pair_sum, previous_pair_sum);

        rho_sum += pair_sum;
        previous_pair_sum = pair_sum;
    }

    // Estimate effective sample size.
    const double ess =
        static_cast<double>(n) /
        (1.0 + 2.0 * rho_sum);

    return std::clamp(
        ess,
        1.0,
        static_cast<double>(n)
    );
}

} // namespace bdn