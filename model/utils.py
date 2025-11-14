from typing import Dict, Optional
import numpy as np
import networkx as nx

def clip_belief(x: float) -> float:
    return max(-1.0, min(1.0, x))

def mixture_beliefs(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a trimodal belief distribution: extremes and moderates.
    Modes near -0.8, 0.0, +0.8 with mixing weights (0.35, 0.30, 0.35).
    """
    rng = np.random.default_rng(seed)
    weights = np.array([0.35, 0.30, 0.35])
    choices = rng.choice([0, 1, 2], size=n, p=weights)
    vals = np.zeros(n)
    for i, c in enumerate(choices):
        if c == 0:
            vals[i] = np.clip(rng.normal(-0.8, 0.12), -1, 1)
        elif c == 1:
            vals[i] = np.clip(rng.normal(0.0, 0.25), -1, 1)
        else:
            vals[i] = np.clip(rng.normal(0.8, 0.12), -1, 1)
    return vals

def assortativity_by_belief_bins(G: nx.Graph, beliefs: Dict[int, float], bins: int = 6) -> float:
    """Approximate homophily by binning belief to categorical attribute and computing
    attribute assortativity. Returns NaN if graph too small/degenerate."""
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return float("nan")
    # Assign attribute category based on bins in [-1, 1]
    for n in G.nodes:
        b = beliefs.get(n, 0.0)
        cat = int(np.digitize(b, np.linspace(-1, 1, bins + 1)) - 1)
        G.nodes[n]["belief_cat"] = cat
        try:
            return nx.attribute_assortativity_coefficient(G, "belief_cat")
        except Exception:
            return float("nan")