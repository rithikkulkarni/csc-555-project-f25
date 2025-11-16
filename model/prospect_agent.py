import random
from typing import List
import numpy as np
from mesa import Agent

try:
    from .utils import clip_belief
except ImportError:
    from utils import clip_belief


class ProspectTheoryAgent(Agent):
    """
    Agent that uses Prospect Theory principles for belief updating.
    Implements loss aversion, reference-dependent evaluation, and probability weighting.
    """
    
    def __init__(self, model, node_id, belief, tolerance, openness, stubbornness, 
                 loss_aversion=2.0, risk_aversion=0.88):
        super().__init__(model)
        self.node_id = int(node_id)
        self.belief = float(belief)
        self.initial_belief = float(belief)  # Reference point
        self.tolerance = float(tolerance)
        self.openness = float(openness)
        self.stubbornness = float(stubbornness)
        
        # Prospect Theory parameters with individual variation
        self.loss_aversion = loss_aversion
        self.risk_aversion = random.uniform(risk_aversion - 0.1, risk_aversion + 0.1)
        self.probability_gamma = random.uniform(0.55, 0.67)
        
    def value_function(self, x):
        """
        Prospect theory value function relative to reference point (initial belief).
        - Concave for gains (risk averse in gains domain)
        - Convex for losses (risk seeking in losses domain)
        - Steeper for losses (loss aversion)
        """
        if x >= 0:
            return x ** self.risk_aversion
        else:
            return -self.loss_aversion * ((-x) ** self.risk_aversion)
    
    def probability_weighting(self, p):
        """
        Probability weighting function - overweights small probabilities,
        underweights large probabilities.
        """
        return (p ** self.probability_gamma) / (
            (p ** self.probability_gamma + (1 - p) ** self.probability_gamma) ** (1/self.probability_gamma)
        )
    
    def evaluate_belief_change(self, target_belief, certainty=1.0):
        """
        Evaluate a potential belief change using prospect theory.
        
        Args:
            target_belief: The belief being considered
            certainty: Probability of adopting this belief (based on number of peers, credibility, etc.)
        
        Returns:
            Weighted value of the change
        """
        # Calculate change relative to reference point (initial belief)
        change = target_belief - self.initial_belief
        
        # Apply value function to the change
        value = self.value_function(change)
        
        # Weight by probability
        weighted_value = self.probability_weighting(certainty) * value
        
        return weighted_value
    
    def sample_exposures(self) -> List[int]:
        """Same exposure logic as SocialAgent"""
        G = self.model.G
        k = self.model.k_exposures
        regime = self.model.graph_regime

        if regime in ("echo", "mixed"):
            nbrs = list(G.neighbors(self.node_id))
            if not nbrs:
                return []
            return random.sample(nbrs, k=min(k, len(nbrs)))

        if regime == "curated":
            beta = self.model.beta
            all_nodes = list(G.nodes)
            candidates = [nid for nid in all_nodes if nid != self.node_id]
            if not candidates:
                return []
            myb = self.belief
            sims = np.array([1.0 - abs(myb - self.model.agent_belief(nid)) for nid in candidates])
            weights = np.exp(beta * sims)
            weights = weights + 1e-9
            probs = weights / weights.sum()
            k_eff = min(k, len(candidates))
            return list(np.random.choice(candidates, size=k_eff, replace=False, p=probs))

        return []
    
    def step(self):
        """
        Update beliefs using prospect theory evaluation.
        """
        peers = self.sample_exposures()
        if not peers:
            return
        
        peer_beliefs = [self.model.agent_belief(pid) for pid in peers]
        
        # Bounded confidence filter
        close_peers = [b for b in peer_beliefs if abs(b - self.belief) <= self.tolerance]
        if not close_peers:
            return
        
        # Calculate target belief (weighted average of acceptable peers)
        mean_peer = float(np.mean(close_peers))
        target = (1.0 - self.openness) * self.belief + self.openness * mean_peer
        
        # Calculate the proposed change
        proposed_change = target - self.belief
        
        # If change is negligible, skip
        if abs(proposed_change) < 0.001:
            return
        
        # Apply stubbornness as base resistance
        movement = (1.0 - self.stubbornness) * proposed_change
        
        # PROSPECT THEORY: Apply loss aversion based on reference point
        current_distance = self.belief - self.initial_belief
        new_distance = target - self.initial_belief
        
        # Check if we're crossing the reference point (changing sign)
        crossing_reference = (current_distance * new_distance < 0) and (current_distance != 0)
        
        # Check if we're moving away from reference point
        moving_away = abs(new_distance) > abs(current_distance)
        
        if crossing_reference:
            # STRONG resistance: crossing reference point feels like a big loss
            # Apply loss aversion squared
            movement /= (self.loss_aversion ** 2)
        elif moving_away:
            # MODERATE resistance: moving away from reference point
            # Apply loss aversion once
            movement /= self.loss_aversion
        # else: moving toward reference point - no additional resistance
        
        # Apply the movement
        new_belief = self.belief + movement
        self.belief = clip_belief(new_belief)