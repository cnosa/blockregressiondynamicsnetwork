# colombian_politics.py
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import time
from tqdm import tqdm
import blockdynamicnetworks as bdn
import community as community_louvain  
ARCHIVO_EXCEL = r"C:\Users\carlo\OneDrive\Documents\blockdynamicnetworks\python\others\data\raw_data_colombian_politics.xlsx"
HOJA = 1
df = pd.read_excel(ARCHIVO_EXCEL, sheet_name=HOJA)
columnas_excluir = ["NOMBRE", "EDAD", "GENERO", "PARTIDO_POLITICO", "NIVEL_DE_ESTUDIOS", "UNIVERSIDAD", "CARRERA", "PREDECESORES", "GOBERNACION","FLAG_MINISTERIO", "FLAG_VIVO"]
columnas_cargos = []
for col in df.columns:
    if col in columnas_excluir:
        continue
    valores = set(df[col].dropna().unique())
    if valores.issubset({0, 1}):
        columnas_cargos.append(col)
B = nx.Graph()
for _, row in df.iterrows():
    nombre = row["NOMBRE"]
    B.add_node(nombre, bipartite="persona", tipo="persona",
        edad=row.get("EDAD"),
        genero=row.get("GENERO"),
        partido=row.get("PARTIDO_POLITICO"),
        nivel_estudios=row.get("NIVEL_DE_ESTUDIOS"),
        universidad=row.get("UNIVERSIDAD"),
        carrera=row.get("CARRERA"),
        predecesores=row.get("PREDECESORES"),
        gobernacion=row.get("GOBERNACION"),
        flag_ministerio=row.get("FLAG_MINISTERIO"),
        flag_vivo=row.get("FLAG_VIVO"))
for cargo in columnas_cargos:
    B.add_node(cargo, bipartite="cargo", tipo="cargo")
for _, row in df.iterrows():
    nombre = row["NOMBRE"]
    for cargo in columnas_cargos:
        if row[cargo] == 1:
            B.add_edge(nombre, cargo, relacion="ocupa")
print("Bipartitie network")
print("Nodes:", B.number_of_nodes())
print("Edges:", B.number_of_edges())
personas = {n for n, d in B.nodes(data=True) if d["tipo"] == "persona"}
cargos = {n for n, d in B.nodes(data=True) if d["tipo"] == "cargo"}
print("Number or persons:", len(personas))
print("Number of employees:", len(cargos))
from networkx.algorithms import bipartite
G_personas = bipartite.weighted_projected_graph(B, personas)
G_real = nx.Graph()
G_real.add_nodes_from(G_personas.nodes(data=True))
for u, v, data in G_personas.edges(data=True):
    if u == v:
        continue
    G_real.add_edge(u, v, weight=1)
print("Colombian politics network")
print("Nodes:", G_real.number_of_nodes())
print("Edges:", G_real.number_of_edges())
print("Directed:", G_real.is_directed())
print("Autoedges:", nx.number_of_selfloops(G_real))
print("Density:", nx.density(G_real))
partition_louvain = community_louvain.best_partition(G_real,weight="weight",random_state=123)
labels_louvain = [partition_louvain[node] for node in G_real.nodes()]
z0 = bdn.Partition(labels_louvain)
print("Initial partition - Louvain")
print("Number of communities:", z0.number_of_communities())
print("Sizes of communities:", z0.community_sizes())
def networkx_to_bdn(G_nx):
    nodes = list(G_nx.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    G_bdn = bdn.Graph(len(nodes))
    for u, v in G_nx.edges():
        i = node_to_index[u]
        j = node_to_index[v]
        if i != j:
            G_bdn.add_edge(i, j)
    return G_bdn, nodes, node_to_index
G_bdn, nodes, node_to_index = networkx_to_bdn(G_real)
print("Converted to blockdynamicnetworks package")
adjacency = nx.to_numpy_array(G_real, nodelist=nodes, dtype=int, weight=None)
prior_DP = bdn.PartitionPrior(alpha=1.918466168283546)
prior_PY = bdn.PartitionPrior(type=bdn.PriorType.PY,alpha=1.291061303823689,sigma=0.01)
prior_DM = bdn.PartitionPrior(type=bdn.PriorType.DM, beta=0.08253466813107523, H_total=int(30))
prior_GN = bdn.PartitionPrior(type=bdn.PriorType.GN, gamma=0.6)
prior_MFM = bdn.PartitionPrior(type=bdn.PriorType.MFM, rho=0.24597707294209856, p_k=np.ones(30) / 30)
prior_PCPP = bdn.PartitionPrior(type=bdn.PriorType.PCPP,theta=9,omega=0.5,pcpp_tol=0.01)
priors = {"DP": {"prior": prior_DP, "parameters": {"alpha": 1.918466168283546}},
    "PY": {"prior": prior_PY, "parameters": {"alpha": 1.291061303823689, "sigma": 0.01}},
    "DM": {"prior": prior_DM, "parameters": {"beta": 0.08253466813107523, "H": 30}},
    "GN": {"prior": prior_GN, "parameters": {"gamma": 0.6}},
    "MFM": {"prior": prior_MFM, "parameters": {"rho": 0.24597707294209856, "p_k": np.ones(30) / 30}},
    "PCPP": {"prior": prior_PCPP, "parameters": {"theta": 9,"omega": 0.5,"pcpp_tol": 0.01}}}

ITERATIONS = 25000
BURN_IN = 5000
THINNING = 4
SEED = 123
n = G_bdn.size()
m = G_bdn.number_of_edges()
adjacency = nx.to_numpy_array(G_real, nodelist=nodes, dtype=int, weight=None)
node_attributes = {node: G_real.nodes[node].copy() for node in nodes}
degree = np.array([G_real.degree(node) for node in nodes])
all_results = {}




print("\n========== PRIOR DP ==========", flush=True)
start_time = time.time()

model_DP = bdn.PosteriorModel(G_bdn, prior_DP, 1.0, 1.0)

sampler_DP = bdn.HybridSampler(
    model_DP,
    iterations=ITERATIONS,
    burn_in=BURN_IN,
    thinning=THINNING,
    seed=SEED,
    gibbs_probability=0.125
)

sampler_DP.run(z0)

samples_DP = sampler_DP.samples()
k_trace_DP = sampler_DP.number_of_communities()
z_final_DP = sampler_DP.final_partition()
results_DP = sampler_DP.results()

elapsed_DP = time.time() - start_time

all_results["politica_colombiana_DP"] = {
    "network": "politica_colombiana",
    "prior": "DP",
    "prior_parameters": {"alpha": 1.918466168283546},
    "n": n,
    "m": m,
    "density": nx.density(G_real),
    "degree": degree,
    "nodes": nodes,
    "node_to_index": node_to_index,
    "node_attributes": node_attributes,
    "adjacency": adjacency,
    "final_partition": np.array(z_final_DP.labels()),
    "K_final": z_final_DP.number_of_communities(),
    "samples": np.array([s.labels() for s in samples_DP]),
    "K_trace": np.array(k_trace_DP),
    "log_posterior": np.array(results_DP.log_posterior_trace),
    "sampler_selection_trace": np.array(results_DP.sampler_selection_trace),
    "saved_samples": len(samples_DP),
    "gibbs_iterations": sampler_DP.gibbs_iterations(),
    "split_merge_iterations": sampler_DP.split_merge_iterations(),
    "gibbs_probability": sampler_DP.gibbs_probability(),
    "split_attempts": int(np.sum(results_DP.split_attempt_trace)),
    "merge_attempts": int(np.sum(results_DP.merge_attempt_trace)),
    "accepted_splits": int(np.sum(results_DP.split_acceptance_trace)),
    "accepted_merges": int(np.sum(results_DP.merge_acceptance_trace)),
    "split_acceptance_rate": results_DP.split_acceptance_rate,
    "merge_acceptance_rate": results_DP.merge_acceptance_rate,
    "ESS_K": results_DP.ess_k,
    "ESS_log_posterior": results_DP.ess_log_posterior,
    "iterations": ITERATIONS,
    "burn_in": BURN_IN,
    "thinning": THINNING,
    "seed": SEED,
    "a": 1.0,
    "b": 1.0,
    "runtime_seconds": elapsed_DP,
    "status": "success"
}

print(
    f"DP | K={z_final_DP.number_of_communities()} | "
    f"Gibbs={sampler_DP.gibbs_iterations()} | "
    f"Split-Merge={sampler_DP.split_merge_iterations()} | "
    f"ESS K={results_DP.ess_k:.2f} | "
    f"ESS log-posterior={results_DP.ess_log_posterior:.2f} | "
    f"Time={elapsed_DP:.2f}s",
    flush=True
)


print("\n========== PRIOR PY ==========", flush=True)
start_time = time.time()

model_PY = bdn.PosteriorModel(G_bdn, prior_PY, 1.0, 1.0)

sampler_PY = bdn.HybridSampler(
    model_PY,
    iterations=ITERATIONS,
    burn_in=BURN_IN,
    thinning=THINNING,
    seed=SEED,
    gibbs_probability=0.125
)

sampler_PY.run(z0)

samples_PY = sampler_PY.samples()
k_trace_PY = sampler_PY.number_of_communities()
z_final_PY = sampler_PY.final_partition()
results_PY = sampler_PY.results()

elapsed_PY = time.time() - start_time

all_results["politica_colombiana_PY"] = {
    "network": "politica_colombiana",
    "prior": "PY",
    "prior_parameters": {
        "alpha": 1.291061303823689,
        "sigma": 0.01
    },
    "n": n,
    "m": m,
    "density": nx.density(G_real),
    "degree": degree,
    "nodes": nodes,
    "node_to_index": node_to_index,
    "node_attributes": node_attributes,
    "adjacency": adjacency,
    "final_partition": np.array(z_final_PY.labels()),
    "K_final": z_final_PY.number_of_communities(),
    "samples": np.array([s.labels() for s in samples_PY]),
    "K_trace": np.array(k_trace_PY),
    "log_posterior": np.array(results_PY.log_posterior_trace),
    "sampler_selection_trace": np.array(results_PY.sampler_selection_trace),
    "saved_samples": len(samples_PY),
    "gibbs_iterations": sampler_PY.gibbs_iterations(),
    "split_merge_iterations": sampler_PY.split_merge_iterations(),
    "gibbs_probability": sampler_PY.gibbs_probability(),
    "split_attempts": int(np.sum(results_PY.split_attempt_trace)),
    "merge_attempts": int(np.sum(results_PY.merge_attempt_trace)),
    "accepted_splits": int(np.sum(results_PY.split_acceptance_trace)),
    "accepted_merges": int(np.sum(results_PY.merge_acceptance_trace)),
    "split_acceptance_rate": results_PY.split_acceptance_rate,
    "merge_acceptance_rate": results_PY.merge_acceptance_rate,
    "ESS_K": results_PY.ess_k,
    "ESS_log_posterior": results_PY.ess_log_posterior,
    "iterations": ITERATIONS,
    "burn_in": BURN_IN,
    "thinning": THINNING,
    "seed": SEED,
    "a": 1.0,
    "b": 1.0,
    "runtime_seconds": elapsed_PY,
    "status": "success"
}

print(
    f"PY | K={z_final_PY.number_of_communities()} | "
    f"Gibbs={sampler_PY.gibbs_iterations()} | "
    f"Split-Merge={sampler_PY.split_merge_iterations()} | "
    f"ESS K={results_PY.ess_k:.2f} | "
    f"ESS log-posterior={results_PY.ess_log_posterior:.2f} | "
    f"Time={elapsed_PY:.2f}s",
    flush=True
)



print("\n========== PRIOR DM ==========", flush=True)
start_time = time.time()

model_DM = bdn.PosteriorModel(G_bdn, prior_DM, 1.0, 1.0)

sampler_DM = bdn.HybridSampler(
    model_DM,
    iterations=ITERATIONS,
    burn_in=BURN_IN,
    thinning=THINNING,
    seed=SEED,
    gibbs_probability=0.125
)

sampler_DM.run(z0)

samples_DM = sampler_DM.samples()
k_trace_DM = sampler_DM.number_of_communities()
z_final_DM = sampler_DM.final_partition()
results_DM = sampler_DM.results()

elapsed_DM = time.time() - start_time

all_results["politica_colombiana_DM"] = {
    "network": "politica_colombiana",
    "prior": "DM",
    "prior_parameters": {
        "beta": 0.08253466813107523,
        "H": 30
    },
    "n": n,
    "m": m,
    "density": nx.density(G_real),
    "degree": degree,
    "nodes": nodes,
    "node_to_index": node_to_index,
    "node_attributes": node_attributes,
    "adjacency": adjacency,
    "final_partition": np.array(z_final_DM.labels()),
    "K_final": z_final_DM.number_of_communities(),
    "samples": np.array([s.labels() for s in samples_DM]),
    "K_trace": np.array(k_trace_DM),
    "log_posterior": np.array(results_DM.log_posterior_trace),
    "sampler_selection_trace": np.array(results_DM.sampler_selection_trace),
    "saved_samples": len(samples_DM),
    "gibbs_iterations": sampler_DM.gibbs_iterations(),
    "split_merge_iterations": sampler_DM.split_merge_iterations(),
    "gibbs_probability": sampler_DM.gibbs_probability(),
    "split_attempts": int(np.sum(results_DM.split_attempt_trace)),
    "merge_attempts": int(np.sum(results_DM.merge_attempt_trace)),
    "accepted_splits": int(np.sum(results_DM.split_acceptance_trace)),
    "accepted_merges": int(np.sum(results_DM.merge_acceptance_trace)),
    "split_acceptance_rate": results_DM.split_acceptance_rate,
    "merge_acceptance_rate": results_DM.merge_acceptance_rate,
    "ESS_K": results_DM.ess_k,
    "ESS_log_posterior": results_DM.ess_log_posterior,
    "iterations": ITERATIONS,
    "burn_in": BURN_IN,
    "thinning": THINNING,
    "seed": SEED,
    "a": 1.0,
    "b": 1.0,
    "runtime_seconds": elapsed_DM,
    "status": "success"
}

print(
    f"DM | K={z_final_DM.number_of_communities()} | "
    f"Gibbs={sampler_DM.gibbs_iterations()} | "
    f"Split-Merge={sampler_DM.split_merge_iterations()} | "
    f"ESS K={results_DM.ess_k:.2f} | "
    f"ESS log-posterior={results_DM.ess_log_posterior:.2f} | "
    f"Time={elapsed_DM:.2f}s",
    flush=True
)



print("\n========== PRIOR GN ==========", flush=True)
start_time = time.time()

model_GN = bdn.PosteriorModel(G_bdn, prior_GN, 1.0, 1.0)

sampler_GN = bdn.HybridSampler(
    model_GN,
    iterations=ITERATIONS,
    burn_in=BURN_IN,
    thinning=THINNING,
    seed=SEED,
    gibbs_probability=0.125
)

sampler_GN.run(z0)

samples_GN = sampler_GN.samples()
k_trace_GN = sampler_GN.number_of_communities()
z_final_GN = sampler_GN.final_partition()
results_GN = sampler_GN.results()

elapsed_GN = time.time() - start_time

all_results["politica_colombiana_GN"] = {
    "network": "politica_colombiana",
    "prior": "GN",
    "prior_parameters": {"gamma": 0.6},
    "n": n,
    "m": m,
    "density": nx.density(G_real),
    "degree": degree,
    "nodes": nodes,
    "node_to_index": node_to_index,
    "node_attributes": node_attributes,
    "adjacency": adjacency,
    "final_partition": np.array(z_final_GN.labels()),
    "K_final": z_final_GN.number_of_communities(),
    "samples": np.array([s.labels() for s in samples_GN]),
    "K_trace": np.array(k_trace_GN),
    "log_posterior": np.array(results_GN.log_posterior_trace),
    "sampler_selection_trace": np.array(results_GN.sampler_selection_trace),
    "saved_samples": len(samples_GN),
    "gibbs_iterations": sampler_GN.gibbs_iterations(),
    "split_merge_iterations": sampler_GN.split_merge_iterations(),
    "gibbs_probability": sampler_GN.gibbs_probability(),
    "split_attempts": int(np.sum(results_GN.split_attempt_trace)),
    "merge_attempts": int(np.sum(results_GN.merge_attempt_trace)),
    "accepted_splits": int(np.sum(results_GN.split_acceptance_trace)),
    "accepted_merges": int(np.sum(results_GN.merge_acceptance_trace)),
    "split_acceptance_rate": results_GN.split_acceptance_rate,
    "merge_acceptance_rate": results_GN.merge_acceptance_rate,
    "ESS_K": results_GN.ess_k,
    "ESS_log_posterior": results_GN.ess_log_posterior,
    "iterations": ITERATIONS,
    "burn_in": BURN_IN,
    "thinning": THINNING,
    "seed": SEED,
    "a": 1.0,
    "b": 1.0,
    "runtime_seconds": elapsed_GN,
    "status": "success"
}

print(
    f"GN | K={z_final_GN.number_of_communities()} | "
    f"Gibbs={sampler_GN.gibbs_iterations()} | "
    f"Split-Merge={sampler_GN.split_merge_iterations()} | "
    f"ESS K={results_GN.ess_k:.2f} | "
    f"ESS log-posterior={results_GN.ess_log_posterior:.2f} | "
    f"Time={elapsed_GN:.2f}s",
    flush=True
)



print("\n========== PRIOR MFM ==========", flush=True)
start_time = time.time()

model_MFM = bdn.PosteriorModel(G_bdn, prior_MFM, 1.0, 1.0)

sampler_MFM = bdn.HybridSampler(
    model_MFM,
    iterations=ITERATIONS,
    burn_in=BURN_IN,
    thinning=THINNING,
    seed=SEED,
    gibbs_probability=0.125
)

sampler_MFM.run(z0)

samples_MFM = sampler_MFM.samples()
k_trace_MFM = sampler_MFM.number_of_communities()
z_final_MFM = sampler_MFM.final_partition()
results_MFM = sampler_MFM.results()

elapsed_MFM = time.time() - start_time

all_results["politica_colombiana_MFM"] = {
    "network": "politica_colombiana",
    "prior": "MFM",
    "prior_parameters": {
        "rho": 0.24597707294209856,
        "p_k": np.ones(30) / 30
    },
    "n": n,
    "m": m,
    "density": nx.density(G_real),
    "degree": degree,
    "nodes": nodes,
    "node_to_index": node_to_index,
    "node_attributes": node_attributes,
    "adjacency": adjacency,
    "final_partition": np.array(z_final_MFM.labels()),
    "K_final": z_final_MFM.number_of_communities(),
    "samples": np.array([s.labels() for s in samples_MFM]),
    "K_trace": np.array(k_trace_MFM),
    "log_posterior": np.array(results_MFM.log_posterior_trace),
    "sampler_selection_trace": np.array(results_MFM.sampler_selection_trace),
    "saved_samples": len(samples_MFM),
    "gibbs_iterations": sampler_MFM.gibbs_iterations(),
    "split_merge_iterations": sampler_MFM.split_merge_iterations(),
    "gibbs_probability": sampler_MFM.gibbs_probability(),
    "split_attempts": int(np.sum(results_MFM.split_attempt_trace)),
    "merge_attempts": int(np.sum(results_MFM.merge_attempt_trace)),
    "accepted_splits": int(np.sum(results_MFM.split_acceptance_trace)),
    "accepted_merges": int(np.sum(results_MFM.merge_acceptance_trace)),
    "split_acceptance_rate": results_MFM.split_acceptance_rate,
    "merge_acceptance_rate": results_MFM.merge_acceptance_rate,
    "ESS_K": results_MFM.ess_k,
    "ESS_log_posterior": results_MFM.ess_log_posterior,
    "iterations": ITERATIONS,
    "burn_in": BURN_IN,
    "thinning": THINNING,
    "seed": SEED,
    "a": 1.0,
    "b": 1.0,
    "runtime_seconds": elapsed_MFM,
    "status": "success"
}

print(
    f"MFM | K={z_final_MFM.number_of_communities()} | "
    f"Gibbs={sampler_MFM.gibbs_iterations()} | "
    f"Split-Merge={sampler_MFM.split_merge_iterations()} | "
    f"ESS K={results_MFM.ess_k:.2f} | "
    f"ESS log-posterior={results_MFM.ess_log_posterior:.2f} | "
    f"Time={elapsed_MFM:.2f}s",
    flush=True
)



print("\n========== PRIOR PCPP ==========", flush=True)
start_time = time.time()

model_PCPP = bdn.PosteriorModel(G_bdn, prior_PCPP, 1.0, 1.0)

sampler_PCPP = bdn.HybridSampler(
    model_PCPP,
    iterations=ITERATIONS,
    burn_in=BURN_IN,
    thinning=THINNING,
    seed=SEED,
    gibbs_probability=0.125
)

sampler_PCPP.run(z0)

samples_PCPP = sampler_PCPP.samples()
k_trace_PCPP = sampler_PCPP.number_of_communities()
z_final_PCPP = sampler_PCPP.final_partition()
results_PCPP = sampler_PCPP.results()

elapsed_PCPP = time.time() - start_time

all_results["politica_colombiana_PCPP"] = {
    "network": "politica_colombiana",
    "prior": "PCPP",
    "prior_parameters": {
        "theta": 9,
        "omega": 0.5,
        "pcpp_tol": 0.01
    },
    "n": n,
    "m": m,
    "density": nx.density(G_real),
    "degree": degree,
    "nodes": nodes,
    "node_to_index": node_to_index,
    "node_attributes": node_attributes,
    "adjacency": adjacency,
    "final_partition": np.array(z_final_PCPP.labels()),
    "K_final": z_final_PCPP.number_of_communities(),
    "samples": np.array([s.labels() for s in samples_PCPP]),
    "K_trace": np.array(k_trace_PCPP),
    "log_posterior": np.array(results_PCPP.log_posterior_trace),
    "sampler_selection_trace": np.array(results_PCPP.sampler_selection_trace),
    "saved_samples": len(samples_PCPP),
    "gibbs_iterations": sampler_PCPP.gibbs_iterations(),
    "split_merge_iterations": sampler_PCPP.split_merge_iterations(),
    "gibbs_probability": sampler_PCPP.gibbs_probability(),
    "split_attempts": int(np.sum(results_PCPP.split_attempt_trace)),
    "merge_attempts": int(np.sum(results_PCPP.merge_attempt_trace)),
    "accepted_splits": int(np.sum(results_PCPP.split_acceptance_trace)),
    "accepted_merges": int(np.sum(results_PCPP.merge_acceptance_trace)),
    "split_acceptance_rate": results_PCPP.split_acceptance_rate,
    "merge_acceptance_rate": results_PCPP.merge_acceptance_rate,
    "ESS_K": results_PCPP.ess_k,
    "ESS_log_posterior": results_PCPP.ess_log_posterior,
    "iterations": ITERATIONS,
    "burn_in": BURN_IN,
    "thinning": THINNING,
    "seed": SEED,
    "a": 1.0,
    "b": 1.0,
    "runtime_seconds": elapsed_PCPP,
    "status": "success"
}

print(
    f"PCPP | K={z_final_PCPP.number_of_communities()} | "
    f"Gibbs={sampler_PCPP.gibbs_iterations()} | "
    f"Split-Merge={sampler_PCPP.split_merge_iterations()} | "
    f"ESS K={results_PCPP.ess_k:.2f} | "
    f"ESS log-posterior={results_PCPP.ess_log_posterior:.2f} | "
    f"Time={elapsed_PCPP:.2f}s",
    flush=True
)

























filename = "SBM_colombian_politics_6_priors.pkl"

with open(filename, "wb") as f:
    pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

successful = sum(
    result["status"] == "success"
    for result in all_results.values()
)

print("Total runs:", 6)
print("Successful:", successful)
print("File:", filename)