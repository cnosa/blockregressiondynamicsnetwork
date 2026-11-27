import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
import time
import blockdynamicnetworks as bdn
import numpy as np
import pickle
import time
def generate_sbm(sizes, P, seed=None):
    rng = np.random.default_rng(seed)
    sizes = np.asarray(sizes, dtype=int)
    K = len(sizes)
    n = sizes.sum()
    if P.shape != (K, K):
        raise ValueError("P debe ser una matriz K x K.")
    if not np.allclose(P, P.T):
        raise ValueError("P debe ser simétrica.")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")
    z_true = np.repeat(np.arange(K), sizes)
    Y = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            p_ij = P[z_true[i], z_true[j]]
            Y[i, j] = rng.binomial(1, p_ij)
            Y[j, i] = Y[i, j]
    G = nx.from_numpy_array(Y)
    nx.set_node_attributes(G, {i: int(z_true[i]) for i in range(n)}, "block")
    return G, Y, z_true
n = 50
sizes = [10, 5, 13, 15, 7]
K = len(sizes)
assert sum(sizes) == n
def pattern_horizontal():
    P = np.full((5, 5), 0.25)
    np.fill_diagonal(P, 0.75)
    return P
def pattern_vertical():
    P = np.full((5, 5), 0.25)
    pairs = [(0, 1), (2, 3), (1, 4), (3, 4)]
    for a, b in pairs:
        P[a, b] = P[b, a] = 0.75
    np.fill_diagonal(P, 0.75)
    return P
def pattern_mixed():
    P = np.full((5, 5), 0.25)
    pairs = [(0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    for a, b in pairs:
        P[a, b] = P[b, a] = 0.75
    np.fill_diagonal(P, 0.75)
    return P
patterns = {"Horizontal": pattern_horizontal(), "Vertical": pattern_vertical(), "Mixed": pattern_mixed()}
def transform_probabilities(P_base, p_in, p_out):
    P_new = np.empty_like(P_base, dtype=float)
    for i in range(P_base.shape[0]):
        for j in range(P_base.shape[1]):
            if i == j:
                P_new[i, j] = p_in
            else:
                if P_base[i, j] > 0.25:
                    P_new[i, j] = p_in
                else:
                    P_new[i, j] = p_out
    return P_new
scenario_config = {
    "Horizontal_usual": {"pattern": "Horizontal", "p_in": 0.75, "p_out": 0.25, "seed": 101},
    "Vertical_usual": {"pattern": "Vertical","p_in": 0.75,"p_out": 0.25,"seed": 102},
    "Mixed_usual": {"pattern": "Mixed","p_in": 0.75,"p_out": 0.25,"seed": 103},
    "Horizontal_extreme": {"pattern": "Horizontal","p_in": 0.95,"p_out": 0.05,"seed": 104},
    "Vertical_extreme": {"pattern": "Vertical","p_in": 0.95,"p_out": 0.05,"seed": 105},
    "Mixed_extreme": {"pattern": "Mixed","p_in": 0.95,"p_out": 0.05,"seed": 106}
}
networks = {}
for name, config in scenario_config.items():
    P_base = patterns[config["pattern"]]
    P = transform_probabilities(P_base,config["p_in"],config["p_out"])
    G, Y, z_true = generate_sbm(sizes=sizes,P=P,seed=config["seed"])
    networks[name] = {"G": G,"Y": Y,"z_true": z_true,"P": P,"sizes": sizes,"n": n,"K_true": K,"config": config}
for name, data in networks.items():
    G = data["G"]
    print(
        f"{name}: "
        f"n={G.number_of_nodes()}, "
        f"m={G.number_of_edges()}, "
        f"K={data['K_true']}, "
        f"densidad={nx.density(G):.4f}"
    )
def networkx_to_bdn(G_nx):
    nodes = list(G_nx.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    G_bdn = bdn.Graph(len(nodes))
    for u, v in G_nx.edges():
        i = node_to_index[u]
        j = node_to_index[v]
        G_bdn.add_edge(i, j)
    return G_bdn, nodes, node_to_index


prior_DP = bdn.PartitionPrior(alpha=1.1787275669831943)
prior_PY = bdn.PartitionPrior(type=bdn.PriorType.PY,alpha=1.0161300150172323,sigma=0.01)
prior_DM = bdn.PartitionPrior(type=bdn.PriorType.DM, beta=0.04473669476337631, H_total=int(30))
prior_GN = bdn.PartitionPrior(type=bdn.PriorType.GN, gamma=0.66)
prior_MFM = bdn.PartitionPrior(type=bdn.PriorType.MFM, rho=0.1076381362769065, p_k=np.ones(30) / 30)
prior_PCPP = bdn.PartitionPrior(type=bdn.PriorType.PCPP,theta=4.0,omega=0.5,pcpp_tol=0.01)
priors = {"DP": {"prior": prior_DP, "parameters": {"alpha": 1.1787275669831943}},
    "PY": {"prior": prior_PY, "parameters": {"alpha": 1.0161300150172323, "sigma": 0.01}},
    "DM": {"prior": prior_DM, "parameters": {"beta": 0.04473669476337631, "H": 30}},
    "GN": {"prior": prior_GN, "parameters": {"gamma": 0.66}},
    "MFM": {"prior": prior_MFM, "parameters": {"rho": 0.1076381362769065, "p_k": np.ones(30) / 30}},
    "PCPP": {"prior": prior_PCPP, "parameters": {"theta": 4.0,"omega": 0.5,"pcpp_tol": 0.01}}}
ITERATIONS = 100_000
BURN_IN = 10_000
THINNING = 90
SEED = 123
A = 1.0
B = 1.0
all_results = {}
total_runs = len(networks) * len(priors)
for scenario_name, scenario_data in tqdm(networks.items()):
    G_nx = scenario_data["G"]
    G_bdn, nodes, node_to_index = networkx_to_bdn(G_nx)
    z_true = scenario_data["z_true"]
    z0 = bdn.Partition([0] * G_bdn.size())
    for prior_name, prior_info in priors.items():
        prior = prior_info["prior"]
        prior_parameters = prior_info["parameters"]
        run_name = f"{scenario_name}_{prior_name}"
        start_time = time.time()
        try:
            model = bdn.PosteriorModel(G_bdn,prior,A,B)
            sampler = bdn.HybridSampler(model,iterations=ITERATIONS,burn_in=BURN_IN,thinning=THINNING,seed=SEED,gibbs_probability=0.05)
            sampler.run(z0)
            samples = sampler.samples()
            k_trace = sampler.number_of_communities()
            z_final = sampler.final_partition()
            results = sampler.results()
            log_posterior_trace = np.array(results.log_posterior_trace)
            sample_labels = np.array([s.labels() for s in samples])
            elapsed = time.time() - start_time
            all_results[run_name] = {
                "scenario": scenario_name,
                "prior": prior_name,
                "prior_parameters": prior_parameters.copy(),
                "n": G_bdn.size(),
                "m": G_bdn.number_of_edges(),
                "density": nx.density(G_nx),
                "K_true": len(np.unique(z_true)),
                "z_true": np.asarray(z_true),
                "P": scenario_data["P"],
                "sizes": scenario_data["sizes"],
                "adjacency": scenario_data["Y"],
                "final_partition": np.array(z_final.labels()),
                "K_final": z_final.number_of_communities(),
                "samples": sample_labels,
                "K_trace": np.array(k_trace),
                "log_posterior": log_posterior_trace,
                "saved_samples": len(samples),
                "gibbs_iterations": sampler.gibbs_iterations(),
                "split_merge_iterations": sampler.split_merge_iterations(),
                "gibbs_probability": sampler.gibbs_probability(),
                "split_attempts": int(np.sum(results.split_attempt_trace)),
                "merge_attempts": int(np.sum(results.merge_attempt_trace)),
                "accepted_splits": int(np.sum(results.split_acceptance_trace)),
                "accepted_merges": int(np.sum(results.merge_acceptance_trace)),
                "split_acceptance_rate": results.split_acceptance_rate,
                "merge_acceptance_rate": results.merge_acceptance_rate,
                "ESS_K": results.ess_k,
                "ESS_log_posterior": results.ess_log_posterior,
                "burn_in": BURN_IN,
                "thinning": THINNING,
                "seed": SEED,
                "a": A,
                "b": B,
                "runtime_seconds": elapsed,
                "status": "success"
            }
            print(
                f"\n{run_name} | "
                f"K final={z_final.number_of_communities()} | "
                f"ESS K={results.ess_k:.2f} | "
                f"Time={elapsed:.2f}s"
            )
        except Exception as e:
            all_results[run_name] = {
                "scenario": scenario_name,
                "prior": prior_name,
                "prior_parameters": prior_parameters.copy(),
                "status": "error",
                "error": str(e)
            }
            print(f"\nERROR {run_name}: {e}")
filename = "SBM_simulation_36_runs.pkl"
with open(filename, "wb") as f:
    pickle.dump(all_results,f,protocol=pickle.HIGHEST_PROTOCOL)
successful = sum(result["status"] == "success" for result in all_results.values())
print("Total of runs:", total_runs)
print("Successful:", successful)
print("File:", filename)