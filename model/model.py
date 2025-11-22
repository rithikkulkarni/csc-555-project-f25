import random
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from mesa import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

from .agent import SocialAgent
from .utils import mixture_beliefs, mixture_beliefs_asymmetric_shift, mixture_beliefs_asymmetric_extremists, skewed_beliefs_positive, clip_belief, assortativity_by_belief_bins

from configs.control_configs import (
    BELIEF_INITIALIZATION,
    GRAPH_TYPE,
    STEPS,
    SEED,
    K_EXPOSURES,
    AVG_DEGREE,
    BETA,
    OPENNESS,
    TOLERANCE,
    TOLERANCE_JITTER,
    STUBBORNNESS,
    NUM_AGENTS,
)

def compute_weak_tie_fraction(m: "SocialBeliefModel") -> float:
    """
    Fraction of edges that are weak ties (by tie_strength).
    This is static per run but useful for sanity checks across configs.
    """
    total = 0
    weak = 0
    for u, v, data in m.G.edges(data=True):
        total += 1
        w = float(data.get("tie_strength", m.strong_tie_weight))
        if np.isclose(w, m.weak_tie_weight):
            weak += 1
    return float(weak) / total if total > 0 else float("nan")



def edge_disagreement(m: "SocialBeliefModel", use_weak: bool) -> float:
    """
    Average |belief_i - belief_j| over strong or weak edges.
    - If use_weak=True: only weak ties.
    - If use_weak=False: only strong ties.
    This shows how much disagreement sits on strong vs weak edges.
    """
    beliefs = {ag.node_id: ag.belief for ag in m.agents}
    diffs = []
    for u, v, data in m.G.edges(data=True):
        w = float(data.get("tie_strength", m.strong_tie_weight))
        is_weak = np.isclose(w, m.weak_tie_weight)
        if is_weak != use_weak:
            continue
        bu = beliefs.get(u, 0.0)
        bv = beliefs.get(v, 0.0)
        diffs.append(abs(bu - bv))
    return float(np.mean(diffs)) if diffs else float("nan")


def inter_cluster_gap(m: "SocialBeliefModel") -> float:
    """
    For echo-chamber graphs with stochastic blocks, compute the difference
    between the most extreme cluster means: max(mean_block) - min(mean_block).
    Large gap = clusters remain far apart; small gap = clusters converged.
    """
    # Only defined for echo regime
    if not hasattr(m, "block_bounds") or m.block_bounds is None:
        return float("nan")

    beliefs = {ag.node_id: ag.belief for ag in m.agents}
    bounds = m.block_bounds
    start = 0
    block_means = []
    for end in bounds:
        idxs = range(start, end)
        vals = [beliefs[i] for i in idxs]
        if vals:
            block_means.append(float(np.mean(vals)))
        start = end

    if len(block_means) < 2:
        return float("nan")
    return float(max(block_means) - min(block_means))


class SocialBeliefModel(Model):
    def __init__(
        self,
        N: int = NUM_AGENTS,
        graph: str = GRAPH_TYPE,
        avg_degree: int = AVG_DEGREE,
        steps: int = STEPS,
        seed: Optional[int] = SEED,
        openness: float = OPENNESS,
        tolerance: float = TOLERANCE,
        stubbornness: float = STUBBORNNESS,
        tolerance_jitter: float = TOLERANCE_JITTER,
        k_exposures: int = K_EXPOSURES,
        beta: float = BETA,
        belief_initialization: int = BELIEF_INITIALIZATION,
    ):
        super().__init__(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.N = N
        self.steps_target = steps
        self.graph_regime = graph
        self.avg_degree = avg_degree
        self.k_exposures = k_exposures
        self.beta = beta
        self.weak_tie_activations = 0
        self.strong_tie_activations = 0

        # Will be set for echo regime
        self.block_bounds: Optional[np.ndarray] = None

        # Construct graph
        self.G = self._make_graph()
        self.grid = NetworkGrid(self.G)

        # Initialize beliefs
        if belief_initialization == 1:
            init_beliefs = mixture_beliefs(N, seed)
        elif belief_initialization == 2:
            init_beliefs = mixture_beliefs_asymmetric_shift(N, seed)
        elif belief_initialization == 3:
            init_beliefs = mixture_beliefs_asymmetric_extremists(N, seed)
        else:
            init_beliefs = skewed_beliefs_positive(N, seed)
        
        tol_vals = np.clip(
            np.random.normal(tolerance, tolerance_jitter, N),
            0.01,
            1.0,
        )

        # Create agents
        for i in range(N):
            a = SocialAgent(
                model=self,
                node_id=i,
                belief=float(init_beliefs[i]),
                tolerance=float(tol_vals[i]),
                openness=float(openness),
                stubbornness=float(stubbornness),
            )
            # Place agent on the graph node with same index
            self.grid.place_agent(a, i)

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.step_count,
                "mean_belief": lambda m: float(np.mean([ag.belief for ag in m.agents])),
                "polarization_var": lambda m: float(np.var([ag.belief for ag in m.agents])),
                "share_extremes": lambda m: float(
                    np.mean([abs(ag.belief) >= 0.9 for ag in m.agents])
                ),
                "assortativity": lambda m: assortativity_by_belief_bins(
                    m.G, {ag.node_id: ag.belief for ag in m.agents}
                ),
                "regime": lambda m: m.graph_regime,
            },
            agent_reporters={"belief": lambda a: a.belief},
        )

        self.step_count = 0

    # Graph builders
    def _make_graph(self) -> nx.Graph:
        if self.graph_regime == "mixed":
            # Small-world - Watts–Strogatz
            k = max(2, self.avg_degree - (self.avg_degree % 2))
            G = nx.watts_strogatz_graph(self.N, k=k, p=0.15)
            return G

        if self.graph_regime == "echo":
            # Stochastic block model with communities
            sizes = [int(0.35 * self.N), int(0.30 * self.N), self.N]
            sizes[2] = self.N - sizes[0] - sizes[1]
            p_in = 0.12
            p_mid = 0.08
            p_out = 0.01
            probs = [
                [p_in,  p_out, p_out],
                [p_out, p_mid, p_out],
                [p_out, p_out, p_in],
            ]
            G = nx.stochastic_block_model(sizes, probs, seed=None)
            mapping = {old: i for i, old in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)

            block_bounds = np.cumsum(sizes)
            self.block_bounds = block_bounds

            return G

        if self.graph_regime == "curated":
            p = min(1.0, self.avg_degree / (self.N - 1))
            G = nx.erdos_renyi_graph(self.N, p)
            return G

        raise ValueError(f"Unknown graph regime: {self.graph_regime}")

    def agent_belief(self, node_id: int) -> float:
        cell_agents = self.grid.get_cell_list_contents([node_id])
        if not cell_agents:
            return 0.0
        return cell_agents[0].belief

    def step(self):
        self.datacollector.collect(self)

        # Reset activation counters for the next step
        self.weak_tie_activations = 0
        self.strong_tie_activations = 0

        self.agents.shuffle_do("step")

        # Keep original counter
        self.step_count += 1

    def run(self, steps: Optional[int] = None, agent_log_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        steps = steps if steps is not None else self.steps_target
        agent_rows: List[Tuple[int, int, float]] = []
        for t in range(steps):
            # Optional per-agent logging
            if agent_log_path is not None:
                for a in self.agents:
                    agent_rows.append((t, a.unique_id, a.belief))
            self.step()
        # Final collection
        self.datacollector.collect(self)
        model_df = self.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        agent_df = None
        if agent_log_path is not None:
            agent_df = pd.DataFrame(agent_rows, columns=["step", "agent_id", "belief"])
            agent_df.to_csv(agent_log_path, index=False)
        return model_df, agent_df
