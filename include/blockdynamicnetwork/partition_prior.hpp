
#ifndef BLOCKDYNAMICNETWORK_PARTITION_PRIOR_HPP
#define BLOCKDYNAMICNETWORK_PARTITION_PRIOR_HPP

#include "partition.hpp"

#include <cstddef>
#include <vector>

namespace bdn {

enum class PriorType {
    DP,
    PY,
    DM,
    GN,
    MFM,
    PCPP
};

class PartitionPrior {
public:
    explicit PartitionPrior(double alpha = 1.0);

    // Constructor for DP and PY
    PartitionPrior(
        PriorType type,
        double alpha,
        double sigma
    );

    // Constructor for DM
    PartitionPrior(
        PriorType type,
        double beta,
        std::size_t H_total
    );

    // Constructor for GN
    PartitionPrior(
        PriorType type,
        double gamma
    );

    // Constructor for MFM
    PartitionPrior(
        PriorType type,
        double rho,
        const std::vector<double>& p_k
    );

    // Constructor for PCPP
    PartitionPrior(
        PriorType type,
        double theta,
        double omega,
        double pcpp_tol
    );
    
    double log_probability(const Partition& partition) const;

    double alpha() const;
    double sigma() const;
    double beta() const;
    double gamma() const;
    double rho() const;
    double theta() const;
    double omega() const;
    double pcpp_tol() const;
    std::size_t H_total() const;
    PriorType type() const;

private:
    double log_V(std::size_t n, std::size_t k) const;

    double log_pcpp_conditional(
        const std::vector<std::size_t>& sizes,
        std::size_t m
    ) const;

    double log_pcpp_eppf(
        const std::vector<std::size_t>& sizes
    ) const;

    static double log_sum_exp(
        const std::vector<double>& values
    );

    PriorType type_;
    double alpha_;
    double sigma_;
    double beta_;
    double gamma_;
    double rho_;
    double theta_;
    double omega_;
    double pcpp_tol_;
    std::size_t H_total_;

    std::vector<double> p_k_;
};

} // namespace bdn

#endif
