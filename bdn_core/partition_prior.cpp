
#include "blockdynamicnetwork/partition_prior.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace bdn {

// ============================================================
// Constructor: DP
// ============================================================

PartitionPrior::PartitionPrior(double alpha)
    : type_(PriorType::DP),
      alpha_(alpha),
      sigma_(0.0),
      beta_(0.0),
      gamma_(0.0),
      rho_(0.0),
      theta_(0.0),
      omega_(0.0),
      pcpp_tol_(1e-10),
      H_total_(0),
      p_k_()
{
    if (!std::isfinite(alpha_) || alpha_ <= 0.0) {
        throw std::invalid_argument(
            "DP requires alpha to be positive and finite."
        );
    }
}

// ============================================================
// Constructor: DP / PY
// ============================================================

PartitionPrior::PartitionPrior(
    PriorType type,
    double alpha,
    double sigma
)
    : type_(type),
      alpha_(alpha),
      sigma_(sigma),
      beta_(0.0),
      gamma_(0.0),
      rho_(0.0),
      theta_(0.0),
      omega_(0.0),
      pcpp_tol_(1e-10),
      H_total_(0),
      p_k_()
{
    if (!std::isfinite(alpha_)) {
        throw std::invalid_argument(
            "Alpha must be finite."
        );
    }

    if (type_ == PriorType::DP) {

        if (alpha_ <= 0.0) {
            throw std::invalid_argument(
                "DP requires alpha > 0."
            );
        }

        sigma_ = 0.0;
    }
    else if (type_ == PriorType::PY) {

        if (!std::isfinite(sigma_) ||
            sigma_ < 0.0 ||
            sigma_ >= 1.0) {
            throw std::invalid_argument(
                "PY requires 0 <= sigma < 1."
            );
        }

        if (alpha_ <= -sigma_) {
            throw std::invalid_argument(
                "PY requires alpha > -sigma."
            );
        }
    }
    else {
        throw std::invalid_argument(
            "Invalid prior type for this constructor."
        );
    }
}

// ============================================================
// Constructor: DM
// ============================================================

PartitionPrior::PartitionPrior(
    PriorType type,
    double beta,
    std::size_t H_total
)
    : type_(type),
      alpha_(0.0),
      sigma_(0.0),
      beta_(beta),
      gamma_(0.0),
      rho_(0.0),
      theta_(0.0),
      omega_(0.0),
      pcpp_tol_(1e-10),
      H_total_(H_total),
      p_k_()
{
    if (type_ != PriorType::DM) {
        throw std::invalid_argument(
            "This constructor is only for DM."
        );
    }

    if (!std::isfinite(beta_) || beta_ <= 0.0) {
        throw std::invalid_argument(
            "DM requires beta > 0 and finite."
        );
    }

    if (H_total_ < 1) {
        throw std::invalid_argument(
            "DM requires H_total >= 1."
        );
    }
}

// ============================================================
// Constructor: GN
// ============================================================

PartitionPrior::PartitionPrior(
    PriorType type,
    double gamma
)
    : type_(type),
      alpha_(0.0),
      sigma_(0.0),
      beta_(0.0),
      gamma_(gamma),
      rho_(0.0),
      theta_(0.0),
      omega_(0.0),
      pcpp_tol_(1e-10),
      H_total_(0),
      p_k_()
{
    if (type_ != PriorType::GN) {
        throw std::invalid_argument(
            "This constructor is only for GN."
        );
    }

    if (!std::isfinite(gamma_) ||
        gamma_ <= 0.0 ||
        gamma_ >= 1.0) {
        throw std::invalid_argument(
            "GN requires 0 < gamma < 1."
        );
    }
}

// ============================================================
// Constructor: MFM
// ============================================================

PartitionPrior::PartitionPrior(
    PriorType type,
    double rho,
    const std::vector<double>& p_k
)
    : type_(type),
      alpha_(0.0),
      sigma_(0.0),
      beta_(0.0),
      gamma_(0.0),
      rho_(rho),
      theta_(0.0),
      omega_(0.0),
      pcpp_tol_(1e-10),
      H_total_(0),
      p_k_(p_k)
{
    if (type_ != PriorType::MFM) {
        throw std::invalid_argument(
            "This constructor is only for MFM."
        );
    }

    if (!std::isfinite(rho_) || rho_ <= 0.0) {
        throw std::invalid_argument(
            "MFM requires rho > 0 and finite."
        );
    }

    if (p_k_.empty()) {
        throw std::invalid_argument(
            "MFM requires a non-empty p_k vector."
        );
    }

    double sum = 0.0;

    for (double p : p_k_) {

        if (!std::isfinite(p) || p < 0.0) {
            throw std::invalid_argument(
                "p_k must contain finite non-negative probabilities."
            );
        }

        sum += p;
    }

    if (sum <= 0.0) {
        throw std::invalid_argument(
            "The sum of p_k must be positive."
        );
    }

    // Normalize p_k
    for (double& p : p_k_) {
        p /= sum;
    }
}

// ============================================================
// Constructor: PCPP
// ============================================================

PartitionPrior::PartitionPrior(
    PriorType type,
    double theta,
    double omega,
    double pcpp_tol
)
    : type_(type),
      alpha_(0.0),
      sigma_(0.0),
      beta_(0.0),
      gamma_(0.0),
      rho_(0.0),
      theta_(theta),
      omega_(omega),
      pcpp_tol_(pcpp_tol),
      H_total_(0),
      p_k_()
{
    if (type_ != PriorType::PCPP) {
        throw std::invalid_argument(
            "This constructor is only for PCPP."
        );
    }

    if (!std::isfinite(theta_) || theta_ <= 0.0) {
        throw std::invalid_argument(
            "PCPP requires theta > 0 and finite."
        );
    }

    if (!std::isfinite(omega_) ||
        omega_ < 0.0 ||
        omega_ > 1.0) {
        throw std::invalid_argument(
            "PCPP requires 0 <= omega <= 1."
        );
    }

    if (!std::isfinite(pcpp_tol_) ||
        pcpp_tol_ <= 0.0 ||
        pcpp_tol_ >= 1.0) {
        throw std::invalid_argument(
            "PCPP requires 0 < pcpp_tol < 1."
        );
    }

    if (theta_ > 700.0) {
        throw std::invalid_argument(
            "PCPP currently requires theta <= 700."
        );
    }
}

// ============================================================
// MFM: log_V
// ============================================================

double PartitionPrior::log_V(
    std::size_t n,
    std::size_t k
) const {

    if (type_ != PriorType::MFM) {
        throw std::invalid_argument(
            "log_V is only available for MFM."
        );
    }

    if (k < 1 || n < k) {
        return -INFINITY;
    }

    const std::size_t M = p_k_.size();

    std::vector<double> log_terms;

    for (std::size_t m = k; m <= M; ++m) {

        const double probability = p_k_[m - 1];

        if (probability <= 0.0) {
            continue;
        }

        const double m_double = static_cast<double>(m);
        const double k_double = static_cast<double>(k);
        const double n_double = static_cast<double>(n);

        const double term =
            std::lgamma(m_double + 1.0)
            - std::lgamma(m_double - k_double + 1.0)
            + std::lgamma(m_double * rho_)
            - std::lgamma(m_double * rho_ + n_double)
            + std::log(probability);

        log_terms.push_back(term);
    }

    if (log_terms.empty()) {
        return -INFINITY;
    }

    // Log-Sum-Exp
    const double max_term =
        *std::max_element(
            log_terms.begin(),
            log_terms.end()
        );

    double sum_exp = 0.0;

    for (double term : log_terms) {
        sum_exp += std::exp(term - max_term);
    }

    return max_term + std::log(sum_exp);
}

// ============================================================
// PCPP: log_pcpp_conditional
// ============================================================

double PartitionPrior::log_pcpp_conditional(
    const std::vector<std::size_t>& sizes,
    std::size_t m
) const {

    const std::size_t n =
        std::accumulate(
            sizes.begin(),
            sizes.end(),
            std::size_t{0}
        );

    const std::size_t K = sizes.size();

    if (m < K) {
        return -INFINITY;
    }

    // Caso M = 1
    if (m == 1) {
        return (K == 1) ? 0.0 : -INFINITY;
    }

    // Caso omega = 0: pesos uniformes
    if (omega_ == 0.0) {

        return
            std::lgamma(static_cast<double>(m) + 1.0)
            - std::lgamma(
                static_cast<double>(m - K) + 1.0
            )
            - static_cast<double>(n)
              * std::log(static_cast<double>(m));
    }

    // Parámetro de la aproximación Dirichlet
    const double m_double = static_cast<double>(m);

    const double alpha_m =
        m_double / (omega_ * omega_)
        - 1.0 / m_double;

    double result =
        std::lgamma(m_double + 1.0)
        - std::lgamma(
            static_cast<double>(m - K) + 1.0
        )
        + std::lgamma(m_double * alpha_m)
        - std::lgamma(
            m_double * alpha_m + static_cast<double>(n)
        );

    for (const auto size : sizes) {

        result +=
            std::lgamma(
                static_cast<double>(size) + alpha_m
            )
            - std::lgamma(alpha_m);
    }

    return result;
}

// ============================================================
// PCPP: log_sum_exp
// ============================================================

double PartitionPrior::log_sum_exp(
    const std::vector<double>& values
) {

    if (values.empty()) {
        return -INFINITY;
    }

    const double max_value =
        *std::max_element(
            values.begin(),
            values.end()
        );

    if (!std::isfinite(max_value)) {
        return max_value;
    }

    double sum = 0.0;

    for (double value : values) {
        sum += std::exp(value - max_value);
    }

    return max_value + std::log(sum);
}

// ============================================================
// PCPP: log_pcpp_eppf
// ============================================================

double PartitionPrior::log_pcpp_eppf(
    const std::vector<std::size_t>& sizes
) const {

    const std::size_t n =
        std::accumulate(
            sizes.begin(),
            sizes.end(),
            std::size_t{0}
        );

    const std::size_t K = sizes.size();

    // Truncamiento de M = 1 + Poisson(theta)
    const double target = 1.0 - pcpp_tol_;

    double poisson_probability = std::exp(-theta_);
    double cumulative = poisson_probability;

    std::size_t q = 0;

    while (cumulative < target) {

        ++q;

        poisson_probability *=
            theta_ / static_cast<double>(q);

        cumulative += poisson_probability;

        if (q > 100000) {
            throw std::runtime_error(
                "PCPP Poisson truncation failed."
            );
        }
    }

    // M = 1 + Poisson(theta)
    const std::size_t L = std::max(K, q + 1);

    std::vector<double> log_terms;

    for (std::size_t m = K; m <= L; ++m) {

        // log P(M = m)
        const double log_p_m =
            -theta_
            + static_cast<double>(m - 1)
              * std::log(theta_)
            - std::lgamma(static_cast<double>(m));

        const double log_conditional =
            log_pcpp_conditional(sizes, m);

        log_terms.push_back(
            log_p_m + log_conditional
        );
    }

    return log_sum_exp(log_terms);
}

// ============================================================
// Partition log probability
// ============================================================

double PartitionPrior::log_probability(
    const Partition& partition
) const {

    const std::size_t n = partition.size();
    const auto sizes = partition.community_sizes();
    const std::size_t K = sizes.size();

    // --------------------------------------------------------
    // Dirichlet Process
    // --------------------------------------------------------

    if (type_ == PriorType::DP) {

        double result =
            std::lgamma(alpha_)
            - std::lgamma(alpha_ + static_cast<double>(n))
            + static_cast<double>(K) * std::log(alpha_);

        for (const auto size : sizes) {
            result += std::lgamma(
                static_cast<double>(size)
            );
        }

        return result;
    }

    // --------------------------------------------------------
    // Pitman-Yor Process
    // --------------------------------------------------------

    if (type_ == PriorType::PY) {

        double log_product = 0.0;

        for (std::size_t j = 1; j < K; ++j) {

            const double term =
                alpha_ + static_cast<double>(j) * sigma_;

            if (term <= 0.0) {
                return -INFINITY;
            }

            log_product += std::log(term);
        }

        double result =
            log_product
            + std::lgamma(alpha_ + 1.0)
            - std::lgamma(alpha_ + static_cast<double>(n));

        for (const auto size : sizes) {

            result +=
                std::lgamma(
                    static_cast<double>(size) - sigma_
                )
                - std::lgamma(1.0 - sigma_);
        }

        return result;
    }

    // --------------------------------------------------------
    // Dirichlet-Multinomial
    // --------------------------------------------------------

    if (type_ == PriorType::DM) {

        if (K > H_total_) {
            return -INFINITY;
        }

        const double H = static_cast<double>(H_total_);
        const double K_double = static_cast<double>(K);
        const double n_double = static_cast<double>(n);

        double result =
            std::lgamma(H + 1.0)
            - std::lgamma(H - K_double + 1.0)
            + std::lgamma(H * beta_)
            - std::lgamma(n_double + H * beta_);

        for (const auto size : sizes) {

            result +=
                std::lgamma(
                    static_cast<double>(size) + beta_
                )
                - std::lgamma(beta_);
        }

        return result;
    }

    // --------------------------------------------------------
    // Gnedin
    // --------------------------------------------------------

    if (type_ == PriorType::GN) {

        const double K_double = static_cast<double>(K);
        const double n_double = static_cast<double>(n);

        double result =
            std::lgamma(K_double)
            + std::lgamma(K_double - gamma_)
            - std::lgamma(1.0 - gamma_)
            + std::lgamma(n_double - K_double + gamma_)
            - std::lgamma(gamma_)
            - std::lgamma(n_double)
            - std::lgamma(n_double + gamma_)
            + std::lgamma(1.0 + gamma_);

        for (const auto size : sizes) {

            result += std::lgamma(
                static_cast<double>(size) + 1.0
            );
        }

        return result;
    }

    // --------------------------------------------------------
    // Mixture of Finite Mixtures
    // --------------------------------------------------------

    if (type_ == PriorType::MFM) {

        const double log_v = log_V(n, K);

        if (!std::isfinite(log_v)) {
            return -INFINITY;
        }

        double result = log_v;

        for (const auto size : sizes) {

            result +=
                std::lgamma(
                    static_cast<double>(size) + rho_
                )
                - std::lgamma(rho_);
        }

        return result;
    }

    // --------------------------------------------------------
    // PCPP
    // --------------------------------------------------------

    if (type_ == PriorType::PCPP) {

        return log_pcpp_eppf(sizes);
    }

    // --------------------------------------------------------
    // Unknown prior
    // --------------------------------------------------------

    throw std::invalid_argument(
        "Unknown partition prior type."
    );
}

// ============================================================
// Getters
// ============================================================

double PartitionPrior::alpha() const {
    return alpha_;
}

double PartitionPrior::sigma() const {
    return sigma_;
}

double PartitionPrior::beta() const {
    return beta_;
}

double PartitionPrior::gamma() const {
    return gamma_;
}

double PartitionPrior::rho() const {
    return rho_;
}

double PartitionPrior::theta() const {
    return theta_;
}

double PartitionPrior::omega() const {
    return omega_;
}

double PartitionPrior::pcpp_tol() const {
    return pcpp_tol_;
}

std::size_t PartitionPrior::H_total() const {
    return H_total_;
}

PriorType PartitionPrior::type() const {
    return type_;
}

} // namespace bdn
