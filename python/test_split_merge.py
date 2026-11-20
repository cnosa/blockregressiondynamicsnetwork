import sys

sys.path.append(r"build_py\Debug")

import blockdynamicnetworks as bdn

# 1. Crear un grafo pequeño
G = bdn.Graph(6)

G.add_edge(0, 1)
G.add_edge(0, 2)
G.add_edge(1, 2)

G.add_edge(3, 4)
G.add_edge(3, 5)
G.add_edge(4, 5)

# 2. Definir partición inicial
Z = bdn.Partition([0, 0, 0, 1, 1, 1])

# 3. Definir prior
prior = bdn.PartitionPrior(
    bdn.PriorType.DP,
    1.0,
    0.0
)

# 4. Construir modelo posterior
posterior = bdn.PosteriorModel(
    G,
    prior,
    1.0,
    1.0
)

# 5. Crear sampler
sampler = bdn.SplitMergeSampler(
    posterior,
    10000,  # iteraciones
    2000,    # burn-in
    20,      # thinning
    123     # semilla
)

# 6. Ejecutar
sampler.run(Z)

# 7. Recuperar resultados
results = sampler.results()

print("\n--- RESULTADOS ---")

print("Número de muestras:", len(sampler.samples()))
print("Partición final:", sampler.final_partition().labels())
print("Número de comunidades:", sampler.number_of_communities())

print("\nTasa aceptación splits:", sampler.accepted_splits())
print("Tasa aceptación merges:", sampler.accepted_merges())

print("\nESS log-posterior:", results.ess_log_posterior)
print("ESS número de comunidades:", results.ess_k)



print("\n--- TRAZAS ---")

print("Muestras almacenadas:", results.saved_samples)
print("Iteraciones:", results.iterations)
print("Burn-in:", results.burn_in)
print("Thinning:", results.thinning)

print("Longitud traza K:", len(results.k_trace))
print("Longitud traza log-posterior:", len(results.log_posterior_trace))

print("\n--- TASAS DE ACEPTACIÓN ---")

print("Split acceptance rate:", results.split_acceptance_rate)
print("Merge acceptance rate:", results.merge_acceptance_rate)

print("\n--- DIAGNÓSTICOS ---")

print("ESS K:", results.ess_k)
print("ESS log-posterior:", results.ess_log_posterior)