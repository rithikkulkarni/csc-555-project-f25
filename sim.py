""" 
Social Computing / Decentralized AI — Agent‑Based Belief Dynamics with Mesa
--------------------------------------------------------------------------
A single-file, headless (CLI) Mesa model you can run immediately to simulate
belief change in different social network/feed regimes (echo chamber, mixed,
curated). It logs summary metrics per step to CSV and can also dump per‑agent
trajectories.

Tested with Mesa >= 2.2, NetworkX >= 3.1, NumPy, Pandas.

Usage (from terminal):

  # 1) Install deps
  pip install mesa networkx numpy pandas

  # 2) Run a small mixed-community demo
  python social_belief_mesa.py --N 400 --graph mixed --steps 200 --seed 1 \
      --openness 0.55 --tolerance 0.35 --stubbornness 0.15

  # 3) Echo-chamber regime
  python social_belief_mesa.py --graph echo

  # 4) Curated-feed regime (higher beta = stronger similarity bias)
  python social_belief_mesa.py --graph curated --beta 4.0

  # 5) Save per-agent trajectories
  python social_belief_mesa.py --agent-log agent_trajectories.csv

Outputs:
  - run_metrics.csv  (one row per step with polarization, extremes, etc.)
  - agent_trajectories.csv (optional, long; id, step, belief)

You can sweep parameters via shell loops or Python.
"""
from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

# -------------------------------
# Helpers
# -------------------------------

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


# -------------------------------
# Agent
# -------------------------------
class SocialAgent(Agent):
    def __init__(
        self,
        unique_id: int,
        model: "SocialBeliefModel",
        belief: float,
        tolerance: float,
        openness: float,
        stubbornness: float,
    ):
        super().__init__(unique_id, model)
        self.belief = float(belief)
        self.tolerance = float(tolerance)  # max distance they'll consider
        self.openness = float(openness)    # weight put on peers vs self
        self.stubbornness = float(stubbornness)  # slows movement toward target

    # --- Exposure policies ---
    def sample_exposures(self) -> List[int]:
        G = self.model.G
        k = self.model.k_exposures
        regime = self.model.graph_regime

        if regime in ("echo", "mixed"):
            # Local neighborhood exposure (who your edges connect you to)
            nbrs = list(G.neighbors(self.unique_id))
            if not nbrs:
                return []
            return random.sample(nbrs, k=min(k, len(nbrs)))

        if regime == "curated":
            # Global exposure with similarity-biased sampling ("feed")
            # p(i sees j) ∝ exp(beta * similarity), similarity = 1 - |bi - bj|
            beta = self.model.beta
            all_nodes = list(G.nodes)
            # don't show self as content source
            candidates = [nid for nid in all_nodes if nid != self.unique_id]
            if not candidates:
                return []
            myb = self.belief
            sims = np.array([1.0 - abs(myb - self.model.agent_belief(nid)) for nid in candidates])
            weights = np.exp(beta * sims)
            # Slight smoothing to avoid zero-prob due to numeric underflow
            weights = weights + 1e-9
            probs = weights / weights.sum()
            k_eff = min(k, len(candidates))
            return list(np.random.choice(candidates, size=k_eff, replace=False, p=probs))

        # Fallback: no exposure
        return []

    # --- Update rule per step ---
    def step(self):
        peers = self.sample_exposures()
        if not peers:
            return
        peer_beliefs = [self.model.agent_belief(pid) for pid in peers]

        # Bounded confidence filter: consider only peers within tolerance
        close_peers = [b for b in peer_beliefs if abs(b - self.belief) <= self.tolerance]
        if not close_peers:
            return

        # DeGroot-style target: convex combo of self and mean of considered peers
        mean_peer = float(np.mean(close_peers))
        target = (1.0 - self.openness) * self.belief + self.openness * mean_peer

        # Stubbornness damps motion toward target
        new_belief = self.belief + (1.0 - self.stubbornness) * (target - self.belief)
        self.belief = clip_belief(new_belief)


# -------------------------------
# Model
# -------------------------------
class SocialBeliefModel(Model):
    def __init__(
        self,
        N: int = 500,
        graph: str = "mixed",  # echo | mixed | curated
        avg_degree: int = 10,
        steps: int = 200,
        seed: Optional[int] = None,
        openness: float = 0.5,
        tolerance: float = 0.3,
        stubbornness: float = 0.1,
        tolerance_jitter: float = 0.05,
        k_exposures: int = 8,
        beta: float = 3.0,  # similarity bias for curated feeds
    ):
        super().__init__()
        self.random.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.N = N
        self.steps = steps
        self.graph_regime = graph
        self.avg_degree = avg_degree
        self.k_exposures = k_exposures
        self.beta = beta

        # Construct graph
        self.G = self._make_graph()
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)

        # Initialize beliefs and heterogeneous tolerances
        init_beliefs = mixture_beliefs(N, seed)
        tol_vals = np.clip(np.random.normal(tolerance, tolerance_jitter, N), 0.01, 1.0)

        # Create agents and place them
        for i in range(N):
            a = SocialAgent(
                unique_id=i,
                model=self,
                belief=float(init_beliefs[i]),
                tolerance=float(tol_vals[i]),
                openness=float(openness),
                stubbornness=float(stubbornness),
            )
            self.schedule.add(a)
            self.grid.place_agent(a, i)  # map id to the same node id

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.step_count,
                "mean_belief": lambda m: float(np.mean([ag.belief for ag in m.schedule.agents])),
                "polarization_var": lambda m: float(np.var([ag.belief for ag in m.schedule.agents])),
                "share_extremes": lambda m: float(np.mean([abs(ag.belief) >= 0.9 for ag in m.schedule.agents])),
                "assortativity": lambda m: assortativity_by_belief_bins(
                    m.G, {ag.unique_id: ag.belief for ag in m.schedule.agents}
                ),
                "regime": lambda m: m.graph_regime,
            },
            agent_reporters={"belief": lambda a: a.belief},
        )

        self.step_count = 0

    # ---------- Graph builders ----------
    def _make_graph(self) -> nx.Graph:
        if self.graph_regime == "mixed":
            # Small‑world mixed network
            # Watts-Strogatz with moderate rewiring (diverse contacts)
            k = max(2, self.avg_degree - (self.avg_degree % 2))
            return nx.watts_strogatz_graph(self.N, k=k, p=0.15)

        if self.graph_regime == "echo":
            # Stochastic block model: two polarized blocks + moderates block,
            # with high intra-block and low inter-block connectivity.
            # Block sizes roughly match mixture_beliefs() weights
            sizes = [int(0.35 * self.N), int(0.30 * self.N), self.N]
            sizes[2] = self.N - sizes[0] - sizes[1]
            # Intra >> inter probabilities
            p_in = 0.12
            p_mid = 0.08  # inside moderates
            p_out = 0.01
            probs = [
                [p_in,  p_out, p_out],
                [p_out, p_mid, p_out],
                [p_out, p_out, p_in],
            ]
            G = nx.stochastic_block_model(sizes, probs, seed=None)
            # Relabel nodes to 0..N-1
            mapping = {old: i for i, old in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            return G

        if self.graph_regime == "curated":
            # Underlying social graph (who you might occasionally DM etc.)
            # but exposure comes from global, similarity-biased feed
            p = min(1.0, self.avg_degree / (self.N - 1))
            return nx.erdos_renyi_graph(self.N, p)

        raise ValueError(f"Unknown graph regime: {self.graph_regime}")

    # ---------- Utilities ----------
    def agent_belief(self, aid: int) -> float:
        # In NetworkGrid, multiple agents can occupy a node, but we place 1:1
        # So, find the agent at node id == aid
        cell_agents = self.grid.get_cell_list_contents([aid])
        if not cell_agents:
            return 0.0
        return cell_agents[0].belief

    # ---------- Simulation loop ----------
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.step_count += 1

    def run(self, steps: Optional[int] = None, agent_log_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        steps = steps if steps is not None else self.steps
        agent_rows: List[Tuple[int, int, float]] = []
        for t in range(steps):
            # Optional per-agent logging
            if agent_log_path is not None:
                for a in self.schedule.agents:
                    agent_rows.append((t, a.unique_id, a.belief))
            self.step()
        # Final collect
        self.datacollector.collect(self)
        model_df = self.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        agent_df = None
        if agent_log_path is not None:
            agent_df = pd.DataFrame(agent_rows, columns=["step", "agent_id", "belief"])
            agent_df.to_csv(agent_log_path, index=False)
        return model_df, agent_df


# -------------------------------
# CLI entry
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=500, help="Number of agents")
    ap.add_argument("--graph", type=str, default="mixed", choices=["mixed", "echo", "curated"], help="Network/feed regime")
    ap.add_argument("--avg-degree", type=int, default=10, help="Approx. average degree for base graph")
    ap.add_argument("--steps", type=int, default=200, help="Steps to simulate")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")

    ap.add_argument("--openness", type=float, default=0.5, help="Weight on peers vs self (0..1)")
    ap.add_argument("--tolerance", type=float, default=0.3, help="Max distance accepted in bounded confidence (0..1)")
    ap.add_argument("--tolerance-jitter", type=float, default=0.05, help="Stddev for individual tolerance heterogeneity")
    ap.add_argument("--stubbornness", type=float, default=0.1, help="Damping on movement toward target (0..1)")
    ap.add_argument("--k-exposures", type=int, default=8, help="Number of posts/peers exposed to per step")
    ap.add_argument("--beta", type=float, default=3.0, help="Similarity bias for curated feeds (higher = stronger)")

    ap.add_argument("--metrics-out", type=str, default="run_metrics.csv", help="Where to write model metrics CSV")
    ap.add_argument("--agent-log", type=str, default=None, help="Optional per-agent trajectory CSV path")

    args = ap.parse_args()

    model = SocialBeliefModel(
        N=args.N,
        graph=args.graph,
        avg_degree=args.avg_degree,
        steps=args.steps,
        seed=args.seed,
        openness=args.openness,
        tolerance=args.tolerance,
        tolerance_jitter=args.tolerance_jitter,
        stubbornness=args.stubbornness,
        k_exposures=args.k_exposures,
        beta=args.beta,
    )

    model_df, _ = model.run(steps=args.steps, agent_log_path=args.agent_log)
    model_df.to_csv(args.metrics_out, index=False)

    print("\nFinished! Saved:")
    print(f"  - {args.metrics_out} (per-step metrics)")
    if args.agent_log:
        print(f"  - {args.agent_log} (per-agent trajectories)")
    print("\nColumns in per-step metrics:\n", list(model_df.columns))


if __name__ == "__main__":
    main()
