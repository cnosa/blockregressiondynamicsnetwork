import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(r"build_py\Debug")

import blockdynamicnetworks as bdn


n = 300
G = bdn.Graph(n)

rng = np.random.default_rng(123)

# Tres comunidades de 100 nodos
community_1 = range(0, 100)
community_2 = range(100, 200)
community_3 = range(200, 300)

communities = [community_1, community_2, community_3]

# Probabilidades de conexión
p_in = 0.12
p_out = 0.015


for i in range(n):
    for j in range(i + 1, n):

        # Identificar si pertenecen a la misma comunidad
        same_community = (i // 100 == j // 100)

        # Probabilidad de conexión
        p = p_in if same_community else p_out

        # Generar conexión
        if rng.random() < p:
            G.add_edge(i, j)

# ============================================================
# 3. PARTICIÓN INICIAL
# ============================================================

Z = bdn.Partition(
    [0]*100 + [1]*100 + [2]*100
)

# ============================================================
# 4. PRIOR
# ============================================================

prior = bdn.PartitionPrior(
    bdn.PriorType.DP,
    1.0,
    0.0
)

# ============================================================
# 5. MODELO POSTERIOR
# ============================================================

posterior = bdn.PosteriorModel(
    G,
    prior,
    1.0,
    1.0
)

# ============================================================
# 6. GIBBS SAMPLER
# ============================================================

sampler = bdn.GibbsSampler(
    posterior,
    10000,  # Iteraciones
    2000,   # Burn-in
    20,     # Thinning
    123     # Semilla
)

# ============================================================
# 7. EJECUTAR
# ============================================================

sampler.run(Z)

# ============================================================
# 8. RESULTADOS
# ============================================================

print("\n--- RESULTADOS GIBBS ---")

print("Número de muestras:", len(sampler.samples()))

print("Partición final:")
print(sampler.final_partition().labels())

print("\nNúmero de comunidades:")
print(sampler.number_of_communities())

print("\nNúmero de iteraciones:")
print(sampler.iterations())