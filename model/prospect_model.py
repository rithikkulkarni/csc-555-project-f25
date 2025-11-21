import random
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from mesa import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

from .prospect_agent import ProspectTheoryAgent
from .utils import mixture_beliefs, clip_belief, assortativity_by_belief_bins, mixture_beliefs_asymmetric_shift, mixture_beliefs_asymmetric_extremists, skewed_beliefs_positive 
from configs.prospect_configs import BELIEF_DISTRIBUTION

class ProspectTheoryModel(Model):
    """
    Opinion dynamics model using Prospect Theory for agent decision-making.
    Tests loss aversion, reference dependence, and probability weighting in social networks.
    """
    
    def __init__(
        self,
        N: int,
        graph: str,
        avg_degree: int,
        steps: int,
        seed: Optional[int],
        openness: float,
        tolerance: float,
        stubbornness: float,
        tolerance_jitter: float,
        k_exposures: int,
        beta: float,
        loss_aversion: float,
        risk_aversion: float,
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
        self.loss_aversion = loss_aversion
        self.risk_aversion = risk_aversion

        # Construct graph
        self.G = self._make_graph()
        self.grid = NetworkGrid(self.G)

        # Initialize beliefs and heterogeneous tolerances - Uncomment the version to be used in an experiment
        # init_beliefs = mixture_beliefs(N, seed) # Type 1
        init_beliefs = mixture_beliefs_asymmetric_shift(N, seed) # Type 2
        # init_beliefs = mixture_beliefs_asymmetric_extremists(N, seed) # Type 3
        # init_beliefs = skewed_beliefs_positive(N, seed) # Type 4
        if BELIEF_DISTRIBUTION == "trimodal":
            init_beliefs = mixture_beliefs(N, seed)
        elif BELIEF_DISTRIBUTION == "asymmetric_shift":
            init_beliefs = mixture_beliefs_asymmetric_shift(N, seed)
        elif BELIEF_DISTRIBUTION == "asymmetric_extremists":
            init_beliefs = mixture_beliefs_asymmetric_extremists(N, seed)
        elif BELIEF_DISTRIBUTION == "skewed_positive":
            init_beliefs = skewed_beliefs_positive(N, seed)
        else:
            raise ValueError(f"Unknown belief distribution: {BELIEF_DISTRIBUTION}")

        tol_vals = np.clip(np.random.normal(tolerance, tolerance_jitter, N), 0.01, 1.0)

        # Create Prospect Theory agents
        for i in range(N):
            a = ProspectTheoryAgent(
                model=self,
                node_id=i,
                belief=float(init_beliefs[i]),
                tolerance=float(tol_vals[i]),
                openness=float(openness),
                stubbornness=float(stubbornness),
                loss_aversion=loss_aversion,
                risk_aversion=risk_aversion,
            )
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
                "loss_aversion": lambda m: m.loss_aversion,
                "belief_drift": lambda m: float(np.mean([abs(ag.belief - ag.initial_belief) for ag in m.agents])),
            },
            agent_reporters={
                "belief": lambda a: a.belief,
                "initial_belief": lambda a: a.initial_belief,
            },
        )

        self.step_count = 0

    def _make_graph(self) -> nx.Graph:
        """Create network based on regime type"""
        if self.graph_regime == "mixed":
            # Small-world network (recommended for Prospect Theory)
            k = max(2, self.avg_degree - (self.avg_degree % 2))
            return nx.watts_strogatz_graph(self.N, k=k, p=0.15)

        if self.graph_regime == "echo":
            # Stochastic block model with clustered communities
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
            return G

        if self.graph_regime == "curated":
            # Random graph for algorithmic feed
            p = min(1.0, self.avg_degree / (self.N - 1))
            return nx.erdos_renyi_graph(self.N, p)

        raise ValueError(f"Unknown graph regime: {self.graph_regime}")

    def agent_belief(self, node_id: int) -> float:
        """Get belief of agent at node_id"""
        cell_agents = self.grid.get_cell_list_contents([node_id])
        if not cell_agents:
            return 0.0
        return cell_agents[0].belief

    def step(self):
        """Advance simulation by one step"""
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")
        self.step_count += 1

    def run(
        self, 
        steps: Optional[int] = None, 
        agent_log_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Run simulation for specified steps"""
        steps = steps if steps is not None else self.steps_target
        agent_rows: List[Tuple[int, int, float, float]] = []
        
        for t in range(steps):
            if agent_log_path is not None:
                for a in self.agents:
                    agent_rows.append((t, a.unique_id, a.belief, a.initial_belief))
            self.step()
        
        # Final collect
        self.datacollector.collect(self)
        model_df = self.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        
        agent_df = None
        if agent_log_path is not None:
            agent_df = pd.DataFrame(
                agent_rows, 
                columns=["step", "agent_id", "belief", "initial_belief"]
            )
            agent_df.to_csv(agent_log_path, index=False)
        
        return model_df, agent_df
