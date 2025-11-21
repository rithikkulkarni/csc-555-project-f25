import random
from typing import List
import numpy as np
from mesa import Agent
from .utils import clip_belief


class SocialAgent(Agent):
    def __init__(self, model, node_id, belief, tolerance, openness, stubbornness):
        super().__init__(model)
        self.node_id = int(node_id) # graph node that this agent occupies
        self.belief = float(belief)
        self.tolerance = float(tolerance) # max distance they'll consider
        self.openness = float(openness) # weight put on peers vs self
        self.stubbornness = float(stubbornness) # slows agent's belief movement

    # Exposure policies
    def sample_exposures(self) -> List[int]:
        G = self.model.G
        k = self.model.k_exposures
        regime = self.model.graph_regime

        if regime in ("echo", "mixed"):
            # Local exposure
            nbrs = list(G.neighbors(self.node_id))
            if not nbrs:
                return []
            return random.sample(nbrs, k=min(k, len(nbrs)))

        if regime == "curated":
            # Global exposure with similarity-biased sampling
            beta = self.model.beta
            all_nodes = list(G.nodes)

            candidates = [nid for nid in all_nodes if nid != self.node_id]
            if not candidates:
                return []
            myb = self.belief
            sims = np.array([1.0 - abs(myb - self.model.agent_belief(nid)) for nid in candidates])
            weights = np.exp(beta * sims)
            # Smoothing
            weights = weights + 1e-9
            probs = weights / weights.sum()
            k_eff = min(k, len(candidates))
            return list(np.random.choice(candidates, size=k_eff, replace=False, p=probs))

        # Fallback
        return []

    def step(self):
        peers = self.sample_exposures()
        if not peers:
            return

        G = self.model.G

        # Collect belief & tie_strength for peers
        close_beliefs = []
        weights = []

        for pid in peers:
            peer_belief = self.model.agent_belief(pid)
            # Confidence filter
            if abs(peer_belief - self.belief) <= self.tolerance:
                # Curated regime might have no edge:
                edge_data = G.get_edge_data(self.node_id, pid, default=None)
                if edge_data is not None:
                    tie_w = float(edge_data.get("tie_strength", 1.0))

                    # Strong vs. Weak tie activation
                    if np.isclose(tie_w, self.model.weak_tie_weight):
                        self.model.weak_tie_activations += 1
                    else:
                        self.model.strong_tie_activations += 1
                else:
                    # If no edge exists (curated), just use a baseline weight
                    tie_w = 1.0

                close_beliefs.append(peer_belief)
                weights.append(tie_w)

        if not close_beliefs:
            return

        weights = np.array(weights, dtype=float)
        # avoid division by zero
        if weights.sum() <= 0:
            mean_peer = float(np.mean(close_beliefs))
        else:
            probs = weights / weights.sum()
            mean_peer = float(np.dot(probs, np.array(close_beliefs, dtype=float)))

        # DeGroot
        target = (1.0 - self.openness) * self.belief + self.openness * mean_peer

        # Stubbornness
        new_belief = self.belief + (1.0 - self.stubbornness) * (target - self.belief)
        self.belief = clip_belief(new_belief)
