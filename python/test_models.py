import sys
import math

sys.path.append(r"build_py\Debug")

import blockdynamicnetworks as bdn

# ==================================================
# 1. Crear grafo
# ==================================================

G = bdn.Graph(4)

G.add_edge(0, 1)
G.add_edge(1, 2)
G.add_edge(2, 3)

# ==================================================
# 2. Crear partición
# ==================================================

Z = bdn.Partition([0, 0, 1, 1])

# ==================================================
# 3. SBM
# ==================================================

sbm = bdn.SBMModel(G, Z)

log_likelihood = sbm.log_collapsed_likelihood(Z)

print("Log-likelihood:", log_likelihood)

# ==================================================
# 4. Prior DP
# ==================================================

prior = bdn.PartitionPrior(
    bdn.PriorType.DP,
    1.0,
    0.0
)

log_prior = prior.log_probability(Z)

print("Log-prior:", log_prior)

# ==================================================
# 5. Modelo posterior
# ==================================================

posterior = bdn.PosteriorModel(
    G,
    prior,
    1.0,
    1.0
)

log_post = posterior.log_probability(Z)

print("Log-posterior:", log_post)
print("Log-likelihood posterior:", posterior.log_likelihood(Z))
print("Log-prior posterior:", posterior.log_prior(Z))

# ==================================================
# 6. Comprobaciones
# ==================================================

assert math.isfinite(log_likelihood)
assert math.isfinite(log_prior)
assert math.isfinite(log_post)

assert math.isclose(
    log_post,
    posterior.log_likelihood(Z) + posterior.log_prior(Z),
    rel_tol=1e-9
)

print("\nTodas las pruebas estadísticas pasaron.")