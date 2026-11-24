import numpy as np
import networkx as nx
import blockdynamicnetworks as bdn

# ============================================================
# 1. CONFIGURACIÓN DEL SBM
# ============================================================

n = 270                 # Número de nodos (elegido por el usuario)
K_true = 3              # Número real de comunidades
p_in = 0.50             # Probabilidad de conexión dentro de comunidad
p_out = 0.20            # Probabilidad de conexión entre comunidades

ITERATIONS = 55_000
BURN_IN = 5_000
THINNING = 10
SEED = 123


# ============================================================
# 2. GENERAR LA ESTRUCTURA COMUNITARIA
# ============================================================

# Distribuir los n nodos entre las K comunidades
sizes = [n // K_true] * K_true

# Distribuir los nodos restantes
for i in range(n % K_true):
    sizes[i] += 1

# Matriz de probabilidades de conexión
probability_matrix = np.full(
    (K_true, K_true),
    p_out
)

np.fill_diagonal(
    probability_matrix,
    p_in
)

# Generar el SBM
G_real = nx.stochastic_block_model(
    sizes,
    probability_matrix,
    seed=SEED
)

# Etiquetas verdaderas de comunidad
z_true = np.array(G_real.graph["partition"])


# ============================================================
# 3. CONVERTIR NETWORKX A BLOCKDYNAMICNETWORKS
# ============================================================

nodes = list(G_real.nodes())

node_to_index = {
    node: i for i, node in enumerate(nodes)
}

graph = bdn.Graph(n)

for u, v in G_real.edges():

    i = node_to_index[u]
    j = node_to_index[v]

    if i != j:
        graph.add_edge(i, j)


# ============================================================
# 4. PARTICIÓN INICIAL
# ============================================================

# Crear una partición aleatoria inicial de K_true comunidades
rng = np.random.default_rng(SEED)

z0 = rng.integers(
    0,
    K_true,
    size=n
)

partition = bdn.Partition(z0.tolist())


# ============================================================
# 5. CREAR EL PRIOR
# ============================================================

prior = bdn.PartitionPrior(1.0)


# ============================================================
# 6. CREAR EL MODELO POSTERIOR
# ============================================================

posterior = bdn.PosteriorModel(
    graph,
    prior
)


# ============================================================
# 7. CREAR HYBRID SAMPLER
# ============================================================

sampler = bdn.HybridSampler(
    posterior,
    iterations=ITERATIONS,
    burn_in=BURN_IN,
    thinning=THINNING,
    seed=SEED,
    gibbs_probability=0.05
)


# ============================================================
# 8. EJECUTAR
# ============================================================

sampler.run(partition)


# ============================================================
# 9. CONSULTAR RESULTADOS
# ============================================================

results = sampler.results()

print("=" * 50)
print("SBM SIMULADO")
print("=" * 50)

print("Número de nodos:", graph.size())
print("Número de aristas:", graph.number_of_edges())
print("Comunidades reales:", K_true)
print("Tamaños reales:", sizes)

print("\nINFERENCIA")

print("Número de muestras:", len(sampler.samples()))
print("Iteraciones Gibbs:", sampler.gibbs_iterations())
print("Iteraciones Split-Merge:", sampler.split_merge_iterations())

print("\nDIAGNÓSTICOS")

print("ESS K:", results.ess_k)
print("ESS log-posterior:", results.ess_log_posterior)

print("Tasa aceptación Split:",
      results.split_acceptance_rate)

print("Tasa aceptación Merge:",
      results.merge_acceptance_rate)

print("\nPartición final:")
print(sampler.final_partition().labels())

results = sampler.results()

print("Split attempts:", np.sum(results.split_attempt_trace))
print("Split accepted:", np.sum(results.split_acceptance_trace))

print("Merge attempts:", np.sum(results.merge_attempt_trace))
print("Merge accepted:", np.sum(results.merge_acceptance_trace))

print("Split acceptance rate:", results.split_acceptance_rate)
print("Merge acceptance rate:", results.merge_acceptance_rate)

z_final = np.array(sampler.final_partition().labels())

unique, counts = np.unique(z_final, return_counts=True)

print("Comunidades finales:", len(unique))
print("Tamaños:", dict(zip(unique, counts)))

from sklearn.metrics import adjusted_rand_score
import numpy as np

z_true = np.repeat(np.arange(K_true), sizes)

z_estimated = np.array(
    sampler.final_partition().labels()
)

ari = adjusted_rand_score(z_true, z_estimated)

print("Adjusted Rand Index (ARI):", ari)



import numpy as np
import matplotlib.pyplot as plt

# Obtener resultados
results = sampler.results()

log_posterior = np.array(results.log_posterior_trace)

# Eje de iteraciones
iterations = np.arange(len(log_posterior))

# ============================================================
# VISUALIZACIÓN DE LA CADENA DE LOG-POSTERIOR
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    iterations,
    log_posterior,
    linewidth=0.8
)

plt.xlabel("Iteración")
plt.ylabel("Log-posterior")
plt.title("Cadena MCMC de la log-posterior")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.plot(
    iterations[BURN_IN:],
    log_posterior[BURN_IN:],
    linewidth=0.8,
    label="Log-posterior"
)


plt.xlabel("Iteración")
plt.ylabel("Log-posterior")
plt.title("Cadena MCMC de la log-posterior")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt

results = sampler.results()

k_trace = np.array(results.k_trace)
iterations = np.arange(len(k_trace))

plt.figure(figsize=(12, 5))

plt.plot(
    iterations,
    k_trace,
    linewidth=0.8
)

plt.axhline(
    K_true,
    linestyle="--",
    label=f"K verdadero = {K_true}"
)

plt.xlabel("Muestra")
plt.ylabel("Número de comunidades (K)")
plt.title("Traza MCMC del número de comunidades")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()




k_post = k_trace[BURN_IN:]

iterations_post = np.arange(
    BURN_IN,
    len(k_trace)
)

plt.figure(figsize=(12, 5))

plt.plot(
    iterations_post,
    k_post,
    linewidth=0.8
)

plt.axhline(
    K_true,
    linestyle="--",
    label=f"K verdadero = {K_true}"
)

plt.xlabel("Muestra")
plt.ylabel("Número de comunidades (K)")
plt.title("Traza de K después del burn-in")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()