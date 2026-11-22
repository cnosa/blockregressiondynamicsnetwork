#include "blockdynamicnetwork/split_merge_sampler.hpp"
#include "blockdynamicnetwork/mcmc_diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>
#include <chrono>
#include <iostream>

namespace bdn {

namespace {

// ------------------------------------------------------------
// Numerically stable functions
// ------------------------------------------------------------

// log(1 + exp(x)), avoiding overflow.
double softplus(double x) {
    if (x > 0.0)
        return x + std::log1p(std::exp(-x));

    return std::log1p(std::exp(x));
}

// Stable logistic function.
double logistic(double x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double e = std::exp(x);
    return e / (1.0 + e);
}

// log(sigmoid(x))
double log_sigmoid(double x) {
    return -softplus(-x);
}

// log(1 - sigmoid(x))
double log_one_minus_sigmoid(double x) {
    return -softplus(x);
}

// ------------------------------------------------------------
// Structural affinity between a node and a pivot
// ------------------------------------------------------------
//
// score(v, pivot) = A(v,pivot)
//                 + lambda * common_neighbors(v,pivot)
//
// Common neighbors are restricted to the community being split.
//
// ------------------------------------------------------------

double structural_score(
    const Graph& graph,
    std::size_t v,
    std::size_t pivot,
    const std::vector<bool>& in_community,
    double lambda
) {
    double direct_connection =
        graph.has_edge(v, pivot) ? 1.0 : 0.0;

    double common_neighbors = 0.0;

    const auto& neighbors_v = graph.neighbors(v);

    for (std::size_t u : neighbors_v) {

        if (!in_community[u])
            continue;

        if (graph.has_edge(u, pivot))
            common_neighbors += 1.0;
    }

    return direct_connection + lambda * common_neighbors;
}

// ------------------------------------------------------------
// Log proposal probabilities for structural Split
// ------------------------------------------------------------
//
// x = beta * (score_i - score_j)
//
// log P(A) = log sigmoid(x)
// log P(B) = log sigmoid(-x)
//
// ------------------------------------------------------------

struct SplitProbabilities {
    double probability_a;
    double log_probability_a;
    double log_probability_b;
};

SplitProbabilities structural_split_probabilities(
    const Graph& graph,
    std::size_t v,
    std::size_t i,
    std::size_t j,
    const std::vector<bool>& in_community,
    double lambda,
    double beta
) {
    double score_i = structural_score(
        graph,
        v,
        i,
        in_community,
        lambda
    );

    double score_j = structural_score(
        graph,
        v,
        j,
        in_community,
        lambda
    );

    double x = beta * (score_i - score_j);

    SplitProbabilities result;

    result.probability_a = logistic(x);
    result.log_probability_a = log_sigmoid(x);
    result.log_probability_b = log_one_minus_sigmoid(x);

    return result;
}

} // anonymous namespace


// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

SplitMergeSampler::SplitMergeSampler(
    const PosteriorModel& posterior,
    std::size_t iterations,
    std::size_t burn_in,
    std::size_t thinning,
    std::uint64_t seed,
    bool verbose
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
      results_(),
      verbose_(verbose)
{
    if (iterations_ == 0)
        throw std::invalid_argument(
            "Iterations must be > 0."
        );

    if (burn_in_ >= iterations_)
        throw std::invalid_argument(
            "Burn-in must be smaller than iterations."
        );

    if (thinning_ == 0)
        throw std::invalid_argument(
            "Thinning must be >= 1."
        );
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
// Structural Split proposal
// ------------------------------------------------------------
//
// The proposal no longer uses Restricted Gibbs Scan.
//
// All non-pivot nodes are assigned independently using structural
// affinities to the two pivots.
//
// lambda = 1
// beta  = 1
//
// ------------------------------------------------------------

std::pair<Partition, double>
SplitMergeSampler::propose_split(
    const Partition& current,
    std::size_t i,
    std::size_t j
) {

    const Graph& graph = posterior_.graph();

    std::vector<int> z = current.labels();

    if (i >= z.size() || j >= z.size() || i == j)
        throw std::invalid_argument(
            "Invalid pivot indices."
        );

    int community = z[i];

    if (z[i] != z[j])
        throw std::invalid_argument(
            "Split pivots must belong to the same community."
        );

    // Nodes belonging to the community being split.
    std::vector<std::size_t> members;

    std::vector<bool> in_community(z.size(), false);

    for (std::size_t v = 0; v < z.size(); ++v) {

        if (z[v] == community) {

            members.push_back(v);
            in_community[v] = true;
        }
    }

    // Create two new labels.
    int max_label = *std::max_element(z.begin(), z.end());

    int label_a = max_label + 1;
    int label_b = max_label + 2;

    // Fix the pivots in different groups.
    z[i] = label_a;
    z[j] = label_b;

    // --------------------------------------------------------
    // Structural proposal parameters
    // --------------------------------------------------------

    constexpr double lambda = 1.0;
    constexpr double beta = 1.0;

    double log_q = 0.0;

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // --------------------------------------------------------
    // Independent assignment of all non-pivot nodes
    // --------------------------------------------------------

    for (std::size_t v : members) {

        if (v == i || v == j)
            continue;

        SplitProbabilities probs =
            structural_split_probabilities(
                graph,
                v,
                i,
                j,
                in_community,
                lambda,
                beta
            );

        double u = uniform(rng_);

        if (u < probs.probability_a) {

            z[v] = label_a;

            log_q += probs.log_probability_a;

        }
        else {

            z[v] = label_b;

            log_q += probs.log_probability_b;
        }
    }

    Partition proposed(z);

    return {proposed, log_q};
}


// ------------------------------------------------------------
// Merge proposal
// ------------------------------------------------------------
//
// Deterministic conditional on the selected pivots.
// Community j is merged into community i.
// ------------------------------------------------------------

Partition SplitMergeSampler::propose_merge(
    const Partition& current,
    std::size_t i,
    std::size_t j
) {

    std::vector<int> z = current.labels();

    int label_i = z[i];
    int label_j = z[j];

    if (label_i == label_j)
        throw std::invalid_argument(
            "Merge pivots must belong to different communities."
        );

    for (std::size_t v = 0; v < z.size(); ++v) {

        if (z[v] == label_j)
            z[v] = label_i;
    }

    return Partition(z);
}


// ------------------------------------------------------------
// Probability of reverse structural Split
// ------------------------------------------------------------
//
// Given a merged partition and the original split target,
// calculate the probability of reconstructing that split.
//
// This uses exactly the same structural proposal as propose_split.
//
// ------------------------------------------------------------

double SplitMergeSampler::log_split_probability(
    const Partition& merged,
    const Partition& target_split,
    std::size_t i,
    std::size_t j
) {

    const Graph& graph = posterior_.graph();

    std::vector<int> z = merged.labels();
    std::vector<int> target = target_split.labels();

    if (i >= z.size() || j >= z.size() || i == j)
        throw std::invalid_argument(
            "Invalid pivot indices."
        );

    int merged_label = z[i];

    int target_a = target[i];
    int target_b = target[j];

    if (target_a == target_b)
        throw std::invalid_argument(
            "Target partition must separate the pivot nodes."
        );

    if (z[i] != z[j])
        throw std::invalid_argument(
            "Reverse Split pivots must belong to the same merged community."
        );

    // Identify the merged community.
    std::vector<std::size_t> members;

    std::vector<bool> in_community(z.size(), false);

    for (std::size_t v = 0; v < z.size(); ++v) {

        if (z[v] == merged_label) {

            members.push_back(v);
            in_community[v] = true;
        }
    }

    // --------------------------------------------------------
    // Structural proposal parameters
    // Must match propose_split().
    // --------------------------------------------------------

    constexpr double lambda = 1.0;
    constexpr double beta = 1.0;

    double log_q = 0.0;

    // --------------------------------------------------------
    // Reconstruct probability of the original split
    // --------------------------------------------------------

    for (std::size_t v : members) {

        if (v == i || v == j)
            continue;

        SplitProbabilities probs =
            structural_split_probabilities(
                graph,
                v,
                i,
                j,
                in_community,
                lambda,
                beta
            );

        // Determine the side of the original split.
        //
        // We compare community membership relative to the pivots,
        // not raw numeric labels, since Partition may normalize labels.
        // ----------------------------------------------------

        if (target[v] == target_a) {

            log_q += probs.log_probability_a;

        }
        else if (target[v] == target_b) {

            log_q += probs.log_probability_b;

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

    // Verify that the posterior graph and partition are compatible.
    if (posterior_.graph().size() != n)
        throw std::invalid_argument(
            "Partition size does not match the posterior graph size."
        );

    // Reserve memory for traces.
    results_.k_trace.reserve(iterations_);
    results_.log_posterior_trace.reserve(iterations_);

    results_.split_attempt_trace.reserve(iterations_);
    results_.merge_attempt_trace.reserve(iterations_);

    results_.split_acceptance_trace.reserve(iterations_);
    results_.merge_acceptance_trace.reserve(iterations_);

    // --------------------------------------------------------
    // MCMC iterations
    // --------------------------------------------------------

    auto start_time = std::chrono::steady_clock::now();

    const std::size_t progress_interval = 5000;

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

            // Reverse Merge is deterministic given the pivots.
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

            // Probability of the reverse structural Split.
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


        // --------------------------------------------------------
        // Progress reporting
        // --------------------------------------------------------

        if (verbose_ &&
            ((iteration + 1) % progress_interval == 0 ||
            iteration + 1 == iterations_)) {

            auto current_time = std::chrono::steady_clock::now();

            double elapsed_seconds =
                std::chrono::duration<double>(
                    current_time - start_time
                ).count();

            double progress =
                100.0 * static_cast<double>(iteration + 1)
                / static_cast<double>(iterations_);

            std::cout << "\r"
                    << "Iteracion: " << iteration + 1
                    << " / " << iterations_
                    << " (" << progress << "%)"
                    << " | Tiempo: " << elapsed_seconds << " s"
                    << std::flush;
        }
    }

    if (verbose_) {
        std::cout << std::endl;
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