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
    ):
        # Mesa 3.x requires explicit super init; seed handled here
        super().__init__(seed=seed)

            # Reproducibility for numpy and stdlib random if user passes seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.N = N
        self.steps_target = steps
        self.graph_regime = graph
        self.avg_degree = avg_degree
        self.k_exposures = k_exposures
        self.beta = beta

        # Construct graph
        self.G = self._make_graph()
        self.grid = NetworkGrid(self.G)

        # Initialize beliefs and heterogeneous tolerances
        init_beliefs = mixture_beliefs(N, seed)
        tol_vals = np.clip(np.random.normal(tolerance, tolerance_jitter, N), 0.01, 1.0)

        # Create agents and place them (agents are auto-registered with the model)
        for i in range(N):
            a = SocialAgent(
                model=self,
                node_id=i,
                belief=float(init_beliefs[i]),
                tolerance=float(tol_vals[i]),
                openness=float(openness),
                stubbornness=float(stubbornness),
            )
            # Place on the graph node with same index
            self.grid.place_agent(a, i)

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.step_count,
                "mean_belief": lambda m: float(np.mean([ag.belief for ag in m.agents])),
                "polarization_var": lambda m: float(np.var([ag.belief for ag in m.agents])),
                "share_extremes": lambda m: float(np.mean([abs(ag.belief) >= 0.9 for ag in m.agents])),
                "assortativity": lambda m: assortativity_by_belief_bins(
                    m.G, {ag.node_id: ag.belief for ag in m.agents}
                ),
                "regime": lambda m: m.graph_regime,
            },
            agent_reporters={"belief": lambda a: a.belief},
        )

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

    # ---------- Utilities ----------
    def agent_belief(self, node_id: int) -> float:
        # In NetworkGrid, multiple agents can occupy a node, but we place 1:1
        # So, find the agent at node id == node_id
        cell_agents = self.grid.get_cell_list_contents([node_id])
        if not cell_agents:
            return 0.0
        return cell_agents[0].belief

    # ---------- Simulation loop ----------
    def step(self):
        # Collect BEFORE updates (keeps original CSV semantics)
        self.datacollector.collect(self)

        # Mesa 3.x: replace scheduler with AgentSet activation
        # RandomActivation → agents.shuffle_do("step")
        self.agents.shuffle_do("step")

        # Maintain original counter
        self.step_count += 1

    def run(self, steps: Optional[int] = None, agent_log_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        steps = steps if steps is not None else self.steps_target
        agent_rows: List[Tuple[int, int, float]] = []
        for t in range(steps):
            # Optional per-agent logging (before update to log current state)
            if agent_log_path is not None:
                for a in self.agents:
                    agent_rows.append((t, a.unique_id, a.belief))
            self.step()
        # Final collect (post-final state)
        self.datacollector.collect(self)
        model_df = self.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        agent_df = None
        if agent_log_path is not None:
            agent_df = pd.DataFrame(agent_rows, columns=["step", "agent_id", "belief"])
            agent_df.to_csv(agent_log_path, index=False)
        return model_df, agent_df