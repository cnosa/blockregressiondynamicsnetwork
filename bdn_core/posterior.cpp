#include "blockdynamicnetwork/posterior.hpp"
#include "blockdynamicnetwork/sbm.hpp"

#include <cmath>
#include <stdexcept>

namespace bdn {

// Constructor
PosteriorModel::PosteriorModel(
    const Graph& graph,
    const PartitionPrior& prior,
    double a,
    double b
)
    : graph_(graph),
      prior_(prior),
      a_(a),
      b_(b)
{
    if (!std::isfinite(a_) || a_ <= 0.0) {
        throw std::invalid_argument(
            "Beta prior requires a > 0 and finite."
        );
    }

    if (!std::isfinite(b_) || b_ <= 0.0) {
        throw std::invalid_argument(
            "Beta prior requires b > 0 and finite."
        );
    }
}

// Collapsed log-likelihood
double PosteriorModel::log_likelihood(
    const Partition& partition
) const {

    SBMModel model(graph_, partition);

    return model.log_collapsed_likelihood(
        partition,
        a_,
        b_
    );
}

// Log-prior
double PosteriorModel::log_prior(
    const Partition& partition
) const {

    return prior_.log_probability(partition);
}

// Log-posterior (up to a constant)
double PosteriorModel::log_probability(
    const Partition& partition
) const {

    return log_likelihood(partition)
         + log_prior(partition);
}

// Beta parameters
double PosteriorModel::a() const {
    return a_;
}

double PosteriorModel::b() const {
    return b_;
}

} // namespace bdn