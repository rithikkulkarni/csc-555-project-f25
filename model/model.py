import random
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from mesa import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector


from .agent import SocialAgent
from .utils import mixture_beliefs, clip_belief, assortativity_by_belief_bins

from configs.credibility_influence_configs import (
    HIGH_CRED_FRACTION,
    HIGH_CREDIBILITY,
    LOW_CREDIBILITY
)

def gini(values):
        arr = np.array(values)
        if np.all(arr == 0):
            return 0.0
        arr = arr.flatten()
        arr = np.sort(arr)
        n = len(arr)
        cumulative = np.cumsum(arr)
        return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

def rank_top_fraction(arr, frac=0.1):
    arr = np.array(arr)
    threshold = np.quantile(arr, 1 - frac)
    return arr >= threshold

class SocialBeliefModel(Model):
    def __init__(
        self,
        N: int = 500,
        graph: str = "mixed", # echo | mixed | curated
        avg_degree: int = 10,
        steps: int = 200,
        seed: Optional[int] = None,
        openness: float = 0.5,
        tolerance: float = 0.3,
        stubbornness: float = 0.1,
        tolerance_jitter: float = 0.05,
        k_exposures: int = 8,
        beta: float = 3.0, # similarity bias for curated feeds
        high_credibility: float = HIGH_CREDIBILITY,
        low_credibility: float = LOW_CREDIBILITY,
        high_cred_fraction: float = HIGH_CRED_FRACTION,

    ):
        # Should create self.random, self._agents, self.schedule and self._next_id if I understand correctly
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
        self.high_credibility = high_credibility
        self.low_credibility = low_credibility
        self.high_cred_fraction = high_cred_fraction


        # Construct graph
        self.G = self._make_graph()
        self.grid = NetworkGrid(self.G)

        # Initialize beliefs and heterogeneous tolerances
        # Initialize beliefs and heterogeneous tolerances
        init_beliefs = mixture_beliefs(N, seed)
        tol_vals = np.clip(np.random.normal(tolerance, tolerance_jitter, N), 0.01, 1.0)

        # Precompute centrality for influence initialization
        centrality = nx.betweenness_centrality(self.G, normalized=True)

        # -----------------------------------------------
        # NEW: Credibility drawn from a normal distribution
        # -----------------------------------------------
        # Example defaults (you can configure these via configs.py):
        CRED_MEAN = 0.5
        CRED_STD = 0.15

        # Draw credibility values
        cred_vals = np.random.normal(CRED_MEAN, CRED_STD, N)

        # Clip to valid range [0, 1]
        cred_vals = np.clip(cred_vals, 0.0, 1.0)
        # -----------------------------------------------

        # ------------------------------------------------------
        # Create agents and assign continuous credibility
        # ------------------------------------------------------
        for i in range(N):
            a = SocialAgent(
                model=self,
                node_id=i,
                belief=float(init_beliefs[i]),
                tolerance=float(tol_vals[i]),
                openness=float(openness),
                stubbornness=float(stubbornness),
            )

            # Assign continuous credibility value
            a.credibility = float(cred_vals[i])

            # Influence initialized from centrality
            a.base_centrality = float(centrality[i])
            a.influence = float(centrality[i])

            # Place agent on graph
            self.grid.place_agent(a, i)


        # ------------------------------------------------------------------------------
        # Data Collection (Updated for Ethan Experiment)
        # ------------------------------------------------------------------------------
        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.step_count,

                # Belief dynamics
                "mean_belief": lambda m: float(np.mean([ag.belief for ag in m.agents])),
                "polarization_var": lambda m: float(np.var([ag.belief for ag in m.agents])),

                # Assortativity (homophily)
                "assortativity": lambda m: assortativity_by_belief_bins(
                    m.G, {ag.node_id: ag.belief for ag in m.agents}
                ),

                # Extremism
                "share_extremes": lambda m: float(np.mean([abs(ag.belief) >= 0.9 for ag in m.agents])),

                # -----------------
                # Influence metrics
                # -----------------
                "mean_influence": lambda m: float(np.mean([ag.influence for ag in m.agents])),
                "var_influence": lambda m: float(np.var([ag.influence for ag in m.agents])),
                "mean_cred_weighted_influence": lambda m: float(
                    np.mean([ag.influence * ag.credibility for ag in m.agents])
                ),

                # Gini coefficient of influence (inequality)
                "influence_gini": lambda m: gini([ag.influence for ag in m.agents]),

                # Fraction of agents in top 10% influence
                "elite_fraction": lambda m: float(
                    np.mean(rank_top_fraction([ag.influence for ag in m.agents], frac=self.high_cred_fraction))
                ),

                "regime": lambda m: m.graph_regime,
            },

            # Agent-level reporters
            agent_reporters={
                "belief": lambda a: a.belief,
                "influence": lambda a: a.influence,
                "credibility": lambda a: a.credibility,
            }
        )
        # ------------------------------------------------------------------------------

        # Maintain original step indexing behavior
        self.step_count = 0

    # ---------- Graph builders ----------
    def _make_graph(self) -> nx.Graph:
        if self.graph_regime == "mixed":
            # Small-world mixed network
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

    def agent_belief(self, node_id: int) -> float:
        # In NetworkGrid, multiple agents can occupy a node, but we place 1:1
        # So, find the agent at node id == node_id
        cell_agents = self.grid.get_cell_list_contents([node_id])
        if not cell_agents:
            return 0.0
        return cell_agents[0].belief

    def step(self):
        # Collect BEFORE updates
        self.datacollector.collect(self)

        self.agents.shuffle_do("step")

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
        # Final collect
        self.datacollector.collect(self)
        model_df = self.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        agent_df = None
        if agent_log_path is not None:
            agent_df = pd.DataFrame(agent_rows, columns=["step", "agent_id", "belief"])
            agent_df.to_csv(agent_log_path, index=False)
        return model_df, agent_df
