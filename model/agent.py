import random
from typing import List
import numpy as np
from mesa import Agent
from .utils import clip_belief


class SocialAgent(Agent):
    def __init__(self, model, node_id, belief, tolerance, openness, stubbornness):
        super().__init__(model)
        self.node_id = int(node_id) # graph node this agent occupies
        self.belief = float(belief) # current opinion in [-1, 1]
        self.tolerance = float(tolerance) # max distance they'll consider
        self.openness = float(openness) # weight put on peers vs self
        self.stubbornness = float(stubbornness) # slows movement toward target
        
        # These will be assigned by the model
        self.credibility: float
        self.influence: float
        self.base_centrality: float 
        
    def sample_exposures(self) -> List[int]:
        G = self.model.G
        k = self.model.k_exposures
        regime = self.model.graph_regime

        # Local neighborhood exposure (who their edges connect you to)
        if regime in ("echo", "mixed"):
            nbrs = list(G.neighbors(self.node_id))
            if not nbrs:
                return []
            return random.sample(nbrs, k=min(k, len(nbrs)))

        # Global exposure with similarity-biased sampling ("feed")
        # p(i sees j) ∝ exp(beta * similarity), similarity = 1 - |bi - bj|
        if regime == "curated":
            beta = self.model.beta
            all_nodes = list(G.nodes)
            candidates = [nid for nid in all_nodes if nid != self.node_id]
            if not candidates:
                return []
            myb = self.belief

             # similarity = 1 - distance in belief
            sims = np.array([1.0 - abs(myb - self.model.agent_belief(nid)) for nid in candidates])
            weights = np.exp(beta * sims)
            # # softmax-like weighting
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

        # Retrieve agent objects for each peer id
        peer_agents = [
            self.model.grid.get_cell_list_contents([pid])[0]
            for pid in peers
        ]

        # Bounded confidence
        # Only accept influence from peers within tolerance range
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
        # target belief = blend of own belief and peer mean
        target = (1.0 - self.openness) * self.belief + self.openness * mean_peer
       
        # Move toward target slowed by stubbornness
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

        # Exponential moving average: 70% old influence, 30% new signal
        alpha = 0.7
        self.influence = alpha * self.influence + (1 - alpha) * movement
        # ------------------------------------------------------------------