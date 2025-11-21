from typing import Dict, Optional
import numpy as np
import networkx as nx

def clip_belief(x: float) -> float:
    return max(-1.0, min(1.0, x))

def mixture_beliefs(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a trimodal belief distribution: extremes and moderates.
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

def mixture_beliefs_asymmetric_shift(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Asymmetric trimodal distribution:
    - Left extremists ~ -0.8
    - Moderates ~ +0.25 (shifted toward right)
    - Right extremists ~ +0.8
    """

    rng = np.random.default_rng(seed)
    weights = np.array([0.30, 0.30, 0.40])  # mild asymmetry
    choices = rng.choice([0, 1, 2], size=n, p=weights)

    vals = np.zeros(n)
    for i, c in enumerate(choices):
        if c == 0:
            vals[i] = np.clip(rng.normal(-0.8, 0.12), -1, 1)
        elif c == 1:
            vals[i] = np.clip(rng.normal(+0.25, 0.18), -1, 1)
        else:
            vals[i] = np.clip(rng.normal(+0.8, 0.12), -1, 1)
    return vals

def mixture_beliefs_asymmetric_extremists(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Asymmetric extremist distribution:
    - 20% left extremists at -0.8
    - 30% moderates at 0
    - 50% right extremists at +0.8
    """

    rng = np.random.default_rng(seed)
    weights = np.array([0.20, 0.30, 0.50])
    choices = rng.choice([0, 1, 2], size=n, p=weights)

    vals = np.zeros(n)
    for i, c in enumerate(choices):
        if c == 0:
            vals[i] = np.clip(rng.normal(-0.8, 0.10), -1, 1)
        elif c == 1:
            vals[i] = np.clip(rng.normal(0.0, 0.20), -1, 1)
        else:
            vals[i] = np.clip(rng.normal(+0.8, 0.10), -1, 1)
    return vals

def skewed_beliefs_positive(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Right-skewed unimodal belief distribution centered around +0.3.
    Produces realistic mild bias without extreme clustering.
    """

    rng = np.random.default_rng(seed)
    raw = rng.beta(a=2.5, b=1.8, size=n)  # beta in [0,1], slightly right-skewed

    # Map [0,1] to [-1,1]
    vals = 2 * raw - 1

    # Shift the mean and compress the region (centered around 0.3)
    vals = 0.6 * vals + 0.3

    vals = np.clip(vals, -1, 1)
    return vals


def assortativity_by_belief_bins(G: nx.Graph, beliefs: Dict[int, float], bins: int = 6) -> float:
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return float("nan")

    # Assign categories first
    for n in G.nodes:
        b = beliefs.get(n, 0.0)
        cat = int(np.digitize(b, np.linspace(-1, 1, bins + 1)) - 1)
        G.nodes[n]["belief_cat"] = cat

    # Compute assortativity after all the assignments
    try:
        return nx.attribute_assortativity_coefficient(G, "belief_cat")
    except Exception:
        return float("nan")
