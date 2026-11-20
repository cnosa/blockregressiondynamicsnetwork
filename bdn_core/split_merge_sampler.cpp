#include "blockdynamicnetwork/split_merge_sampler.hpp"
#include "blockdynamicnetwork/mcmc_diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>

namespace bdn {

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

SplitMergeSampler::SplitMergeSampler(
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
      final_partition_(std::nullopt),
      accepted_splits_(0),
      accepted_merges_(0),
      results_()
{
    if (iterations_ == 0)
        throw std::invalid_argument("Iterations must be > 0.");

    if (burn_in_ >= iterations_)
        throw std::invalid_argument(
            "Burn-in must be smaller than iterations."
        );

    if (thinning_ == 0)
        throw std::invalid_argument("Thinning must be >= 1.");
}

// ------------------------------------------------------------
// Select two distinct nodes uniformly
// ------------------------------------------------------------

std::pair<std::size_t, std::size_t>
SplitMergeSampler::select_pivots(std::size_t n) {

    if (n < 2)
        throw std::invalid_argument(
            "SplitMergeSampler requires at least 2 nodes."
        );

    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    std::size_t i = dist(rng_);
    std::size_t j = dist(rng_);

    while (j == i)
        j = dist(rng_);

    return {i, j};
}

// ------------------------------------------------------------
// Log-sum-exp for two values
// ------------------------------------------------------------

double SplitMergeSampler::log_sum_exp(double a, double b) {

    if (a == -std::numeric_limits<double>::infinity())
        return b;

    if (b == -std::numeric_limits<double>::infinity())
        return a;

    double m = std::max(a, b);

    return m + std::log(
        std::exp(a - m) + std::exp(b - m)
    );
}

// ------------------------------------------------------------
// Metropolis-Hastings acceptance
// ------------------------------------------------------------

bool SplitMergeSampler::accept_move(double log_ratio) {

    if (std::isnan(log_ratio))
        return false;

    if (log_ratio >= 0.0)
        return true;

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    double u = uniform(rng_);

    if (u == 0.0)
        return true;

    return std::log(u) < log_ratio;
}

// ------------------------------------------------------------
// Split proposal using Restricted Gibbs Scan
// ------------------------------------------------------------

std::pair<Partition, double>
SplitMergeSampler::propose_split(
    const Partition& current,
    std::size_t i,
    std::size_t j
) {

    std::vector<int> z = current.labels();

    int community = z[i];

    std::vector<std::size_t> members;

    for (std::size_t v = 0; v < z.size(); ++v) {
        if (z[v] == community)
            members.push_back(v);
    }

    // Create two new labels.
    int max_label = *std::max_element(z.begin(), z.end());

    int label_a = max_label + 1;
    int label_b = max_label + 2;

    z[i] = label_a;
    z[j] = label_b;

    // Initially assign remaining nodes to group A.
    for (auto v : members) {
        if (v != i && v != j)
            z[v] = label_a;
    }

    // Use a deterministic scan order.
    std::vector<std::size_t> scan;

    for (auto v : members) {
        if (v != i && v != j)
            scan.push_back(v);
    }

    double log_q = 0.0;

    for (std::size_t v : scan) {

        // Candidate A.
        std::vector<int> z_a = z;
        z_a[v] = label_a;

        Partition p_a(z_a);

        double log_w_a = posterior_.log_probability(p_a);

        // Candidate B.
        std::vector<int> z_b = z;
        z_b[v] = label_b;

        Partition p_b(z_b);

        double log_w_b = posterior_.log_probability(p_b);

        double log_norm = log_sum_exp(log_w_a, log_w_b);

        // Restricted Gibbs probabilities.
        double log_prob_a = log_w_a - log_norm;
        double log_prob_b = log_w_b - log_norm;

        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        double u = uniform(rng_);

        if (std::log(u) < log_prob_a) {
            z[v] = label_a;
            log_q += log_prob_a;
        }
        else {
            z[v] = label_b;
            log_q += log_prob_b;
        }
    }

    Partition proposed(z);

    return {proposed, log_q};
}

// ------------------------------------------------------------
// Merge proposal
// ------------------------------------------------------------

Partition SplitMergeSampler::propose_merge(
    const Partition& current,
    std::size_t i,
    std::size_t j
) {

    std::vector<int> z = current.labels();

    int label_i = z[i];
    int label_j = z[j];

    for (std::size_t v = 0; v < z.size(); ++v) {
        if (z[v] == label_j)
            z[v] = label_i;
    }

    return Partition(z);
}

// ------------------------------------------------------------
// Probability of reverse split under RGS
// ------------------------------------------------------------

double SplitMergeSampler::log_split_probability(
    const Partition& merged,
    const Partition& target_split,
    std::size_t i,
    std::size_t j
) {

    std::vector<int> z = merged.labels();
    std::vector<int> target = target_split.labels();

    int merged_label = z[i];
    int target_a = target[i];
    int target_b = target[j];

    if (target_a == target_b) {
        throw std::invalid_argument(
            "Target partition must separate the pivot nodes."
        );
    }

    std::vector<std::size_t> members;

    for (std::size_t v = 0; v < z.size(); ++v) {
        if (z[v] == merged_label)
            members.push_back(v);
    }

    // Assign pivots to separate groups.
    z[i] = target_a;
    z[j] = target_b;

    // Initially place other nodes into group A.
    for (auto v : members) {
        if (v != i && v != j)
            z[v] = target_a;
    }

    // Use the same deterministic scan order.
    std::vector<std::size_t> scan;

    for (auto v : members) {
        if (v != i && v != j)
            scan.push_back(v);
    }

    double log_q = 0.0;

    for (std::size_t v : scan) {

        // Probability of assigning to A.
        std::vector<int> z_a = z;
        z_a[v] = target_a;

        Partition p_a(z_a);

        double log_w_a = posterior_.log_probability(p_a);

        // Probability of assigning to B.
        std::vector<int> z_b = z;
        z_b[v] = target_b;

        Partition p_b(z_b);

        double log_w_b = posterior_.log_probability(p_b);

        double log_norm = log_sum_exp(log_w_a, log_w_b);

        int target_label = target[v];

        if (target_label == target_a) {

            log_q += log_w_a - log_norm;
            z[v] = target_a;

        }
        else if (target_label == target_b) {

            log_q += log_w_b - log_norm;
            z[v] = target_b;

        }
        else {

            throw std::invalid_argument(
                "Target partition is incompatible with the proposed split."
            );
        }
    }

    return log_q;
}

// ------------------------------------------------------------
// Run Split-Merge MCMC
// ------------------------------------------------------------

void SplitMergeSampler::run(
    const Partition& initial_partition
) {

    // Reset previous run data.
    samples_.clear();
    community_counts_.clear();

    accepted_splits_ = 0;
    accepted_merges_ = 0;

    results_ = SplitMergeResults{};

    Partition current = initial_partition;

    const std::size_t n = current.size();

    if (n < 2)
        throw std::invalid_argument(
            "SplitMergeSampler requires at least 2 nodes."
        );

    // Reserve memory for full traces.
    results_.k_trace.reserve(iterations_);
    results_.log_posterior_trace.reserve(iterations_);

    results_.split_attempt_trace.reserve(iterations_);
    results_.merge_attempt_trace.reserve(iterations_);

    results_.split_acceptance_trace.reserve(iterations_);
    results_.merge_acceptance_trace.reserve(iterations_);

    // --------------------------------------------------------
    // MCMC iterations
    // --------------------------------------------------------

    for (std::size_t iteration = 0;
         iteration < iterations_;
         ++iteration) {

        int split_attempted = 0;
        int merge_attempted = 0;

        int split_accepted = 0;
        int merge_accepted = 0;

        auto pivots = select_pivots(n);

        std::size_t i = pivots.first;
        std::size_t j = pivots.second;

        int label_i = current.label(i);
        int label_j = current.label(j);

        double log_current =
            posterior_.log_probability(current);

        // ----------------------------------------------------
        // SPLIT
        // ----------------------------------------------------

        if (label_i == label_j) {

            split_attempted = 1;

            auto proposal = propose_split(current, i, j);

            Partition proposed = proposal.first;
            double log_q_split = proposal.second;

            double log_proposed =
                posterior_.log_probability(proposed);

            // Reverse merge is deterministic given the pivot pair.
            double log_ratio =
                log_proposed - log_current - log_q_split;

            if (accept_move(log_ratio)) {

                current = proposed;

                ++accepted_splits_;
                split_accepted = 1;
            }
        }

        // ----------------------------------------------------
        // MERGE
        // ----------------------------------------------------

        else {

            merge_attempted = 1;

            Partition proposed =
                propose_merge(current, i, j);

            double log_proposed =
                posterior_.log_probability(proposed);

            // Reverse split probability.
            double log_q_reverse =
                log_split_probability(
                    proposed,
                    current,
                    i,
                    j
                );

            double log_ratio =
                log_proposed - log_current + log_q_reverse;

            if (accept_move(log_ratio)) {

                current = proposed;

                ++accepted_merges_;
                merge_accepted = 1;
            }
        }

        // ----------------------------------------------------
        // Record full traces
        // ----------------------------------------------------

        const int current_k =
            static_cast<int>(current.number_of_communities());

        const double current_log_posterior =
            posterior_.log_probability(current);

        results_.k_trace.push_back(current_k);

        results_.log_posterior_trace.push_back(
            current_log_posterior
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

        // ----------------------------------------------------
        // Save posterior samples
        // ----------------------------------------------------

        if (iteration >= burn_in_ &&
            ((iteration - burn_in_) % thinning_ == 0)) {

            samples_.push_back(current);

            community_counts_.push_back(
                current.number_of_communities()
            );

            results_.samples.push_back(current.labels());
        }
    }

    // --------------------------------------------------------
    // Acceptance rates
    // --------------------------------------------------------

    const double split_attempts = std::accumulate(
        results_.split_attempt_trace.begin(),
        results_.split_attempt_trace.end(),
        0.0
    );

    const double merge_attempts = std::accumulate(
        results_.merge_attempt_trace.begin(),
        results_.merge_attempt_trace.end(),
        0.0
    );

    const double split_accepts = std::accumulate(
        results_.split_acceptance_trace.begin(),
        results_.split_acceptance_trace.end(),
        0.0
    );

    const double merge_accepts = std::accumulate(
        results_.merge_acceptance_trace.begin(),
        results_.merge_acceptance_trace.end(),
        0.0
    );

    results_.split_acceptance_rate =
        split_attempts > 0.0
        ? split_accepts / split_attempts
        : 0.0;

    results_.merge_acceptance_rate =
        merge_attempts > 0.0
        ? merge_accepts / merge_attempts
        : 0.0;

    // --------------------------------------------------------
    // Effective sample size (post burn-in, no thinning)
    // --------------------------------------------------------

    std::vector<double> k_post_burnin;
    std::vector<double> log_posterior_post_burnin;

    k_post_burnin.reserve(iterations_ - burn_in_);
    log_posterior_post_burnin.reserve(iterations_ - burn_in_);

    for (std::size_t t = burn_in_; t < iterations_; ++t) {

        k_post_burnin.push_back(
            static_cast<double>(results_.k_trace[t])
        );

        log_posterior_post_burnin.push_back(
            results_.log_posterior_trace[t]
        );
    }

    results_.ess_k =
        effective_sample_size(k_post_burnin);

    results_.ess_log_posterior =
        effective_sample_size(log_posterior_post_burnin);

    // --------------------------------------------------------
    // Run metadata
    // --------------------------------------------------------

    results_.iterations = iterations_;
    results_.burn_in = burn_in_;
    results_.thinning = thinning_;
    results_.saved_samples = results_.samples.size();

    results_.final_partition = current.labels();

    final_partition_.emplace(current);
}

// ------------------------------------------------------------
// Accessors
// ------------------------------------------------------------

const std::vector<Partition>&
SplitMergeSampler::samples() const {
    return samples_;
}

const std::vector<std::size_t>&
SplitMergeSampler::number_of_communities() const {
    return community_counts_;
}

const Partition&
SplitMergeSampler::final_partition() const {

    if (!final_partition_.has_value()) {
        throw std::runtime_error(
            "SplitMergeSampler has not been run yet."
        );
    }

    return final_partition_.value();
}

std::size_t
SplitMergeSampler::iterations() const {
    return iterations_;
}

std::size_t
SplitMergeSampler::accepted_splits() const {
    return accepted_splits_;
}

std::size_t
SplitMergeSampler::accepted_merges() const {
    return accepted_merges_;
}

const SplitMergeResults&
SplitMergeSampler::results() const {
    return results_;
}

} // namespace bdn