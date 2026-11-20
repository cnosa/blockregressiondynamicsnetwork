// cmake -S . -B build
// cmake --build build
// .\build\Debug\bdn_test.exe

#include "blockdynamicnetwork/graph.hpp"
#include "blockdynamicnetwork/partition.hpp"
#include "blockdynamicnetwork/block_stats.hpp"
#include "blockdynamicnetwork/sbm.hpp"
#include "blockdynamicnetwork/partition_prior.hpp"
#include "blockdynamicnetwork/posterior.hpp"
#include "blockdynamicnetwork/gibbs_sampler.hpp"
#include "blockdynamicnetwork/split_merge_sampler.hpp"

#include <iostream>
#include <vector>
#include <cstddef>
#include <map>
#include <string>
#include <functional>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <numeric>

std::string partition_key(const bdn::Partition& p) {
    std::string key;

    for (int label : p.labels()) {
        key += std::to_string(label) + ",";
    }

    return key;
}

std::vector<bdn::Partition> generate_all_partitions(
    std::size_t n
) {
    std::vector<bdn::Partition> partitions;
    std::vector<int> z(n, 0);

    std::function<void(std::size_t, int)> generate =
        [&](std::size_t pos, int max_label) {

        if (pos == n) {
            partitions.emplace_back(z);
            return;
        }

        for (int label = 0; label <= max_label + 1; ++label) {
            z[pos] = label;

            generate(
                pos + 1,
                std::max(max_label, label)
            );
        }
    };

    if (n == 0) {
        return partitions;
    }

    z[0] = 0;
    generate(1, 0);

    return partitions;
}

int main() {

    // 1. Create graph
    bdn::Graph g(4);

    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);

    // 2. Define partition
    bdn::Partition z({0, 0, 1, 1});

    // Partition prior: Dirichlet Process
    bdn::PartitionPrior prior(1.0);

    double log_prior = prior.log_probability(z);

    std::cout << "\nPartition prior (DP):\n";
    std::cout << "Alpha: " << prior.alpha() << "\n";
    std::cout << "Log prior: " << log_prior << "\n";

    // Partition prior: Pitman-Yor
    bdn::PartitionPrior prior_py(
        bdn::PriorType::PY,
        1.0,  // alpha
        0.2    // sigma
    );

    double log_prior_py = prior_py.log_probability(z);

    std::cout << "\nPartition prior (PY):\n";
    std::cout << "Alpha: " << prior_py.alpha() << "\n";
    std::cout << "Sigma: " << prior_py.sigma() << "\n";
    std::cout << "Log prior: " << log_prior_py << "\n";

    // Partition prior: Dirichlet-Multinomial
    bdn::PartitionPrior prior_dm(
        bdn::PriorType::DM,
        0.5,  // beta
        static_cast<std::size_t>(3)      // H_total
    );

    double log_prior_dm = prior_dm.log_probability(z);

    std::cout << "\nPartition prior (DM):\n";
    std::cout << "Beta: " << prior_dm.beta() << "\n";
    std::cout << "H_total: " << prior_dm.H_total() << "\n";
    std::cout << "Log prior: " << log_prior_dm << "\n";

    // Partition prior: Gnedin
    bdn::PartitionPrior prior_gn(
        bdn::PriorType::GN,
        0.5  // gamma
    );

    double log_prior_gn = prior_gn.log_probability(z);

    std::cout << "\nPartition prior (GN):\n";
    std::cout << "Gamma: " << prior_gn.gamma() << "\n";
    std::cout << "Log prior: " << log_prior_gn << "\n";


    // Partition prior: Mixture of Finite Mixtures
    std::vector<double> probabilities = {
        0.2, 0.3, 0.5
    };

    bdn::PartitionPrior prior_mfm(
        bdn::PriorType::MFM,
        1.0,           // rho
        probabilities   // P(M = 1), P(M = 2), P(M = 3)
    );

    double log_prior_mfm = prior_mfm.log_probability(z);

    std::cout << "\nPartition prior (MFM):\n";
    std::cout << "Rho: " << prior_mfm.rho() << "\n";
    std::cout << "Log prior: " << log_prior_mfm << "\n";

    // Partition prior: PCPP
    bdn::PartitionPrior prior_pcpp(
        bdn::PriorType::PCPP,
        1.0,       // theta
        0.5,        // omega
        1e-10       // pcpp_tol
    );

    double log_prior_pcpp =
        prior_pcpp.log_probability(z);

    std::cout << "\nPartition prior (PCPP):\n";
    std::cout << "Theta: " << prior_pcpp.theta() << "\n";
    std::cout << "Omega: " << prior_pcpp.omega() << "\n";
    std::cout << "Tolerance: " << prior_pcpp.pcpp_tol() << "\n";
    std::cout << "Log prior: " << log_prior_pcpp << "\n";

    // 3. Block statistics
    bdn::BlockStats stats(g, z);

    std::cout << "Community sizes: ";
    for (auto n : stats.community_sizes()) {
        std::cout << n << " ";
    }

    std::cout << "\n\nBlock statistics:\n";

    for (int r = 0; r < 2; ++r) {
        for (int s = r; s < 2; ++s) {
            std::cout << "(" << r << "," << s << ")"
                      << " edges=" << stats.edge_count(r, s)
                      << " possible=" << stats.possible_edge_count(r, s)
                      << " non-edges=" << stats.non_edge_count(r, s)
                      << "\n";
        }
    }

        // 4. SBM model
    bdn::SBMModel model(g, z);

    // 5. Ordinary likelihood with fixed Theta
    std::vector<std::vector<double>> theta = {
        {0.8, 0.2},
        {0.2, 0.7}
    };

    double ll = model.log_likelihood(z, theta);

    // 6. Collapsed likelihood with Beta(1,1)
    double collapsed = model.log_collapsed_likelihood(z);

    // 7. Collapsed likelihood with Beta(2,2)
    double collapsed_22 =
        model.log_collapsed_likelihood(z, 2.0, 2.0);

    std::cout << "\nLog-likelihood: "
              << ll << "\n";

    std::cout << "Collapsed log-likelihood Beta(1,1): "
              << collapsed << "\n";

    std::cout << "Collapsed log-likelihood Beta(2,2): "
              << collapsed_22 << "\n";


    // 8. Posterior model using DP prior

    bdn::PosteriorModel posterior(
        g,
        prior
    );

    double posterior_ll = posterior.log_likelihood(z);
    double posterior_lp = posterior.log_prior(z);
    double log_posterior = posterior.log_probability(z);

    std::cout << "\nPosterior model (DP):\n";

    std::cout << "Log-likelihood: "
              << posterior_ll << "\n";

    std::cout << "Log-prior: "
              << posterior_lp << "\n";

    std::cout << "Log-posterior: "
              << log_posterior << "\n";


    // 9. Exact posterior validation

    auto all_partitions = generate_all_partitions(4);

    std::vector<double> log_probs;
    double max_log_prob = -std::numeric_limits<double>::infinity();

    for (const auto& p : all_partitions) {
        double lp = posterior.log_probability(p);

        log_probs.push_back(lp);

        if (lp > max_log_prob) {
            max_log_prob = lp;
        }
    }

    // Normalize exact posterior probabilities
    std::vector<double> exact_probabilities;
    double total = 0.0;

    for (double lp : log_probs) {
        double w = std::exp(lp - max_log_prob);

        exact_probabilities.push_back(w);
        total += w;
    }

    for (double& p : exact_probabilities) {
        p /= total;
    }

    std::cout << "\nExact posterior:\n";

    for (std::size_t i = 0; i < all_partitions.size(); ++i) {
        std::cout << partition_key(all_partitions[i])
                << " -> "
                << exact_probabilities[i]
                << "\n";
    }

    // 10. Run Gibbs for validation

    bdn::GibbsSampler sampler_validation(
        posterior,
        10000,  // iterations
        1000,   // burn-in
        10,      // thinning
        123      // seed
    );

    sampler_validation.run(z);

    // Count empirical partition frequencies
    std::map<std::string, std::size_t> counts;

    for (const auto& sample : sampler_validation.samples()) {
        counts[partition_key(sample)]++;
    }

    // Compare exact and empirical probabilities
    std::cout << "\nValidation: exact vs Gibbs\n";

    for (std::size_t i = 0; i < all_partitions.size(); ++i) {

        std::string key = partition_key(all_partitions[i]);

        double empirical = 0.0;

        if (counts.find(key) != counts.end()) {
            empirical =
                static_cast<double>(counts[key]) /
                sampler_validation.samples().size();
        }

        std::cout << key
                << " | Exact: " << exact_probabilities[i]
                << " | Gibbs: " << empirical
                << " | Difference: "
                << std::abs(exact_probabilities[i] - empirical)
                << "\n";
    }

    std::cout << "\nTotal saved samples: "
            << sampler_validation.samples().size()
            << "\n";

    // Split-Merge sampler
    bdn::SplitMergeSampler sm_sampler(
        posterior,
        100000,  // iterations
        10000,   // burn-in
        10,      // thinning
        123      // seed
    );

    sm_sampler.run(z);

    // Count empirical partition frequencies
    std::map<std::string, std::size_t> sm_counts;

    for (const auto& sample : sm_sampler.samples()) {
        sm_counts[partition_key(sample)]++;
    }

    // Compare exact vs empirical probabilities
    std::cout << "\nValidation: Exact vs Split-Merge\n";

    for (std::size_t i = 0; i < all_partitions.size(); ++i) {

        std::string key = partition_key(all_partitions[i]);

        double empirical = 0.0;

        if (sm_counts.find(key) != sm_counts.end()) {
            empirical =
                static_cast<double>(sm_counts[key]) /
                sm_sampler.samples().size();
        }

        std::cout << key
                << " | Exact: " << exact_probabilities[i]
                << " | Split-Merge: " << empirical
                << " | Difference: "
                << std::abs(exact_probabilities[i] - empirical)
                << "\n";
    }

    std::cout << "\nTotal saved samples: "
            << sm_sampler.samples().size()
            << "\n";

    std::cout << "Accepted splits: "
            << sm_sampler.accepted_splits()
            << "\n";

    std::cout << "Accepted merges: "
            << sm_sampler.accepted_merges()
            << "\n";


    // 11. Split-Merge diagnostics and results

    const bdn::SplitMergeResults& sm_results = sm_sampler.results();

    // General run information
    std::cout << "\n===== Split-Merge Results =====\n";

    std::cout << "Iterations: "
            << sm_results.iterations << "\n";

    std::cout << "Burn-in: "
            << sm_results.burn_in << "\n";

    std::cout << "Thinning: "
            << sm_results.thinning << "\n";

    std::cout << "Saved samples: "
            << sm_results.saved_samples << "\n";

    // Acceptance rates
    std::cout << "\n===== Acceptance Rates =====\n";

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Split acceptance rate: "
            << sm_results.split_acceptance_rate << "\n";

    std::cout << "Merge acceptance rate: "
            << sm_results.merge_acceptance_rate << "\n";

    // Effective sample sizes
    std::cout << "\n===== Effective Sample Size (ESS) =====\n";

    std::cout << "ESS (number of communities): "
            << sm_results.ess_k << "\n";

    std::cout << "ESS (log-posterior): "
            << sm_results.ess_log_posterior << "\n";

    // Trace sizes
    std::cout << "\n===== Trace Information =====\n";

    std::cout << "K trace length: "
            << sm_results.k_trace.size() << "\n";

    std::cout << "Log-posterior trace length: "
            << sm_results.log_posterior_trace.size() << "\n";

    std::cout << "Split attempts: "
            << std::accumulate(
                sm_results.split_attempt_trace.begin(),
                sm_results.split_attempt_trace.end(),
                0
            ) << "\n";

    std::cout << "Merge attempts: "
            << std::accumulate(
                sm_results.merge_attempt_trace.begin(),
                sm_results.merge_attempt_trace.end(),
                0
            ) << "\n";

    return 0;
}