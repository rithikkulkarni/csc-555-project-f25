import random
from typing import List
import numpy as np
from mesa import Agent
from .utils import clip_belief


class SocialAgent(Agent):
    def __init__(self, model, node_id, belief, tolerance, openness, stubbornness):
        # Mesa 3.x: unique_id is auto-assigned; call super with just model
        super().__init__(model)
        self.node_id = int(node_id) # graph node this agent occupies
        self.belief = float(belief)
        self.tolerance = float(tolerance) # max distance they'll consider
        self.openness = float(openness) # weight put on peers vs self
        self.stubbornness = float(stubbornness) # slows movement toward target

    # --- Exposure policies ---
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

        # --- Update rule per step (CONTROL: no tie-strength logic) ---
    def step(self):
        # Sample exposure set according to the regime
        peers = self.sample_exposures()
        if not peers:
            return

        # Collect beliefs of peers within tolerance
        close_beliefs = []
        for pid in peers:
            peer_belief = self.model.agent_belief(pid)
            # Bounded confidence filter
            if abs(peer_belief - self.belief) <= self.tolerance:
                close_beliefs.append(peer_belief)

        # If no peers pass the tolerance filter, do nothing
        if not close_beliefs:
            return

        # In the control model, all tolerated peers are weighted equally
        close_beliefs_arr = np.array(close_beliefs, dtype=float)
        mean_peer = float(np.mean(close_beliefs_arr))

        # DeGroot-style target: blend self and peer mean
        target = (1.0 - self.openness) * self.belief + self.openness * mean_peer

        # Stubbornness damps motion toward target
        new_belief = self.belief + (1.0 - self.stubbornness) * (target - self.belief)
        self.belief = clip_belief(new_belief)

