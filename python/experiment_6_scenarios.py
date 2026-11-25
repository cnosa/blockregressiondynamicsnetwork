import numpy as np
import networkx as nx
from tqdm.auto import tqdm
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
    pairs = [(0, 1),
        (2, 3),
        (1, 4),
        (3, 4)
    ]

    for a, b in pairs:
        P[a, b] = P[b, a] = 0.75

    np.fill_diagonal(P, 0.75)

    return P


def pattern_mixed():

    P = np.full((5, 5), 0.25)

    # Conexiones fuertes selectivas
    pairs = [
        (0, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4)
    ]

    for a, b in pairs:
        P[a, b] = P[b, a] = 0.75

    np.fill_diagonal(P, 0.75)

    return P


patterns = {
    "Horizontal": pattern_horizontal(),
    "Vertical": pattern_vertical(),
    "Mixed": pattern_mixed()
}


# ============================================================
# 4. TRANSFORMAR MATRICES A PROBABILIDADES USUALES Y EXTREMAS
# ============================================================

def transform_probabilities(P_base, p_in, p_out):

    P_new = np.empty_like(P_base, dtype=float)

    for i in range(P_base.shape[0]):
        for j in range(P_base.shape[1]):

            if i == j:
                P_new[i, j] = p_in
            else:
                # Conservamos la estructura de conexiones
                # entre bloques del patrón base.
                if P_base[i, j] > 0.25:
                    P_new[i, j] = p_in
                else:
                    P_new[i, j] = p_out

    return P_new


# ============================================================
# 5. DEFINIR LOS SEIS ESCENARIOS
# ============================================================

scenario_config = {

    "Horizontal_usual": {
        "pattern": "Horizontal",
        "p_in": 0.75,
        "p_out": 0.25,
        "seed": 101
    },

    "Vertical_usual": {
        "pattern": "Vertical",
        "p_in": 0.75,
        "p_out": 0.25,
        "seed": 102
    },

    "Mixed_usual": {
        "pattern": "Mixed",
        "p_in": 0.75,
        "p_out": 0.25,
        "seed": 103
    },

    "Horizontal_extreme": {
        "pattern": "Horizontal",
        "p_in": 0.95,
        "p_out": 0.05,
        "seed": 104
    },

    "Vertical_extreme": {
        "pattern": "Vertical",
        "p_in": 0.95,
        "p_out": 0.05,
        "seed": 105
    },

    "Mixed_extreme": {
        "pattern": "Mixed",
        "p_in": 0.95,
        "p_out": 0.05,
        "seed": 106
    }
}


# ============================================================
# 6. GENERAR LAS REDES
# ============================================================

networks = {}

for name, config in scenario_config.items():

    P_base = patterns[config["pattern"]]

    P = transform_probabilities(
        P_base,
        config["p_in"],
        config["p_out"]
    )

    G, Y, z_true = generate_sbm(
        sizes=sizes,
        P=P,
        seed=config["seed"]
    )

    networks[name] = {
        "G": G,
        "Y": Y,
        "z_true": z_true,
        "P": P,
        "sizes": sizes,
        "n": n,
        "K_true": K,
        "config": config
    }


# ============================================================
# 7. RESUMEN DE LOS ESCENARIOS
# ============================================================

for name, data in networks.items():

    G = data["G"]

    print(
        f"{name}: "
        f"n={G.number_of_nodes()}, "
        f"m={G.number_of_edges()}, "
        f"K={data['K_true']}, "
        f"densidad={nx.density(G):.4f}"
    )