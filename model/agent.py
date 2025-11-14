import random
from typing import List
import numpy as np
from mesa import Agent
from .utils import clip_belief


class SocialAgent(Agent):
    def __init__(self, model, node_id, belief, tolerance, openness, stubbornness):
        super().__init__(model)
        self.node_id = int(node_id) # graph node this agent occupies
        self.belief = float(belief)
        self.tolerance = float(tolerance) # max distance they'll consider
        self.openness = float(openness) # weight put on peers vs self
        self.stubbornness = float(stubbornness) # slows movement toward target
        self.credibility: float     # fixed over time (e.g. 0.2 vs 0.9)
        self.influence: float       # dynamic, updated each round
        self.base_centrality: float # from betweenness
    def sample_exposures(self) -> List[int]:
        G = self.model.G
        k = self.model.k_exposures
        regime = self.model.graph_regime

        if regime in ("echo", "mixed"):
            # Local neighborhood exposure (who your edges connect you to)
            nbrs = list(G.neighbors(self.node_id))
            if not nbrs:
                return []
            return random.sample(nbrs, k=min(k, len(nbrs)))

        if regime == "curated":
            # Global exposure with similarity-biased sampling ("feed")
            # p(i sees j) ∝ exp(beta * similarity), similarity = 1 - |bi - bj|
            beta = self.model.beta
            all_nodes = list(G.nodes)
            # don't show self as content source
            candidates = [nid for nid in all_nodes if nid != self.node_id]
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

        # Convert to agent objects
        peer_agents = [
            self.model.grid.get_cell_list_contents([pid])[0]
            for pid in peers
        ]

        # Bounded confidence
        close_peer_agents = [
            ag for ag in peer_agents
            if abs(ag.belief - self.belief) <= self.tolerance
        ]
        if not close_peer_agents:
            return

        # Credibility × Influence weighting
        weights = np.array([ag.influence * ag.credibility for ag in close_peer_agents])
        weights = weights / weights.sum()

        # Weighted peer belief
        mean_peer = float(np.sum([ag.belief * w for ag, w in zip(close_peer_agents, weights)]))

        # DeGroot update with stubbornness
        target = (1.0 - self.openness) * self.belief + self.openness * mean_peer
        new_belief = self.belief + (1.0 - self.stubbornness) * (target - self.belief)
        self.belief = clip_belief(new_belief)

        # ------------------------------------------------------------------
        # Dynamic Influence Update
        # ------------------------------------------------------------------
        movement = 0.0
        for ag in close_peer_agents:
            # high score if neighbor is close to this agent's belief
            movement += max(0.0, 1 - abs(ag.belief - self.belief))

        if close_peer_agents:
            movement /= len(close_peer_agents)

        alpha = 0.7
        self.influence = alpha * self.influence + (1 - alpha) * movement
        # ------------------------------------------------------------------