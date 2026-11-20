import sys

sys.path.append(r"build_py\Debug")

import blockdynamicnetworks as bdn

# 1. Crear grafo
G = bdn.Graph(6)

G.add_edge(0, 1)
G.add_edge(0, 2)
G.add_edge(1, 2)

G.add_edge(3, 4)
G.add_edge(3, 5)
G.add_edge(4, 5)

# 2. Partición inicial
Z = bdn.Partition([0, 0, 0, 1, 1, 1])

# 3. Prior
prior = bdn.PartitionPrior(
    bdn.PriorType.DP,
    1.0,
    0.0
)

# 4. Modelo posterior
posterior = bdn.PosteriorModel(
    G,
    prior,
    1.0,
    1.0
)

# 5. Crear Gibbs sampler
sampler = bdn.GibbsSampler(
    posterior,
    10000,  # iteraciones
    2000,   # burn-in
    20,     # thinning
    123     # semilla
)

# 6. Ejecutar
sampler.run(Z)

# 7. Resultados
print("\n--- RESULTADOS GIBBS ---")

print("Número de muestras:", len(sampler.samples()))
print("Partición final:", sampler.final_partition().labels())

print("\nNúmero de comunidades:")
print(sampler.number_of_communities())

print("\nNúmero de iteraciones:", sampler.iterations())