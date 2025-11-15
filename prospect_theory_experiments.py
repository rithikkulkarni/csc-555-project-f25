"""
Improved Prospect Theory Experiments with Balanced Scenarios
"""

import pandas as pd
import random
import numpy as np
from datetime import datetime

class ProspectTheoryAgent:
    """An agent that makes decisions based on prospect theory with individual variation"""
    
    def __init__(self, agent_id, loss_aversion=2.0, risk_aversion=0.88, reference_point=100):
        self.agent_id = agent_id
        self.wealth = reference_point
        self.reference_point = reference_point
        self.loss_aversion = loss_aversion
        
        # Add individual variation to risk aversion
        self.risk_aversion = random.uniform(risk_aversion - 0.1, risk_aversion + 0.1)
        
        # Individual variation in probability weighting
        self.probability_gamma = random.uniform(0.55, 0.67)  # Typical range from literature
        
        self.choices_made = []
        self.random = random.Random()
        
    def value_function(self, x):
        """Prospect theory value function with individual variation"""
        if x >= 0:
            return x ** self.risk_aversion  # Concave for gains
        else:
            return -self.loss_aversion * ((-x) ** self.risk_aversion)  # Convex for losses
    
    def probability_weighting(self, p):
        """Probability weighting function with individual variation"""
        return (p ** self.probability_gamma) / ((p ** self.probability_gamma + (1 - p) ** self.probability_gamma) ** (1/self.probability_gamma))
    
    def make_decision(self, choice_A, choice_B, choice_description=""):
        """
        Make decision between two prospects
        """
        # Calculate prospect values
        value_A = self.probability_weighting(choice_A[1]) * self.value_function(choice_A[0])
        value_B = self.probability_weighting(choice_B[1]) * self.value_function(choice_B[0])
        
        # Add proportional random noise to simulate decision uncertainty
        value_range = max(abs(value_A), abs(value_B))
        noise_scale = value_range * 0.05  # 5% of value range
        noise = random.gauss(0, noise_scale)
        value_A += noise
        value_B -= noise
        
        # Choose the higher value prospect
        if value_A > value_B:
            choice = "A"
            outcome = self.simulate_outcome(choice_A)
        else:
            choice = "B"
            outcome = self.simulate_outcome(choice_B)
            
        # Update wealth and record choice
        self.wealth += outcome
        self.choices_made.append({
            "choice": choice,
            "outcome": outcome,
            "wealth_after": self.wealth,
            "relative_wealth": self.wealth - self.reference_point,
            "description": choice_description
        })
        
        return choice, outcome
    
    def simulate_outcome(self, choice):
        """Simulate the outcome of a choice based on probability"""
        outcome, probability = choice
        if self.random.random() < probability:
            return outcome
        else:
            return 0

class ProspectTheoryModel:
    def __init__(self, N=100, loss_aversion=2.0, risk_aversion=0.88, reference_point=100):
        self.num_agents = N
        self.agents = []
        
        # Create agents with individual variation
        for i in range(self.num_agents):
            agent = ProspectTheoryAgent(i, loss_aversion, risk_aversion, reference_point)
            self.agents.append(agent)

def run_loss_aversion_experiment():
    """Experiment 1: Test different loss aversion parameters"""
    print("=== Experiment 1: Loss Aversion ===")
    
    results = []
    loss_aversion_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    for lambda_val in loss_aversion_values:
        print(f"Testing loss aversion λ = {lambda_val}")
        
        model = ProspectTheoryModel(N=100, loss_aversion=lambda_val)
        risky_choices = 0
        
        # More balanced scenario: sure $45 vs 50% chance of $100
        for agent in model.agents:
            sure_thing = (45, 1.0)      # Sure $45
            risky_gain = (100, 0.5)     # 50% chance of $100
            
            choice, _ = agent.make_decision(sure_thing, risky_gain)
            
            if choice == "B":
                risky_choices += 1
        
        results.append({
            'loss_aversion': lambda_val,
            'risky_choices_percent': (risky_choices / len(model.agents)) * 100,
            'average_wealth': sum(agent.wealth for agent in model.agents) / len(model.agents)
        })
    
    df = pd.DataFrame(results)
    print("\nResults:")
    print(df[['loss_aversion', 'risky_choices_percent']])
    return df

def run_framing_experiment():
    """Experiment 2: Framing effects - this works perfectly!"""
    print("\n=== Experiment 2: Framing Effects ===")
    
    results = []
    
    # Test 1: Gain Frame
    model_gain = ProspectTheoryModel(N=100)
    gain_frame_risky = 0
    
    for agent in model_gain.agents:
        sure_thing = (200, 1.0)    # Positive frame
        risky_option = (400, 0.5)  
        choice, _ = agent.make_decision(sure_thing, risky_option, "Gain frame")
        if choice == "B":
            gain_frame_risky += 1
    
    # Test 2: Loss Frame  
    model_loss = ProspectTheoryModel(N=100)
    loss_frame_risky = 0
    
    for agent in model_loss.agents:
        sure_thing = (-400, 1.0)   # Negative frame  
        risky_option = (-200, 0.5) 
        choice, _ = agent.make_decision(sure_thing, risky_option, "Loss frame")
        if choice == "B":
            loss_frame_risky += 1
    
    results.append({
        'frame': 'Gain',
        'risky_choices_percent': (gain_frame_risky / len(model_gain.agents)) * 100,
        'description': 'Save 200 vs 50% save 400'
    })
    
    results.append({
        'frame': 'Loss', 
        'risky_choices_percent': (loss_frame_risky / len(model_loss.agents)) * 100,
        'description': 'Lose 400 vs 50% lose 200'
    })
    
    df = pd.DataFrame(results)
    print("Results:")
    print(df[['frame', 'risky_choices_percent', 'description']])
    return df

def run_reference_point_experiment():
    """Experiment 3: Reference points with more balanced scenario"""
    print("\n=== Experiment 3: Reference Points ===")
    
    results = []
    reference_points = [80, 100, 120]  # More meaningful reference points
    
    for ref_point in reference_points:
        model = ProspectTheoryModel(N=100, reference_point=ref_point)
        risky_choices = 0
        
        for agent in model.agents:
            agent.wealth = 100  # Same actual wealth, different reference points
            
            # More balanced scenario
            safe_choice = (25, 1.0)     # Sure $25
            risky_choice = (60, 0.5)    # 50% chance of $60
            
            choice, _ = agent.make_decision(safe_choice, risky_choice)
            
            if choice == "B":
                risky_choices += 1
        
        above_ref = sum(1 for agent in model.agents if agent.wealth > agent.reference_point)
        position = 'Above' if 100 > ref_point else 'Below'
        
        results.append({
            'reference_point': ref_point,
            'risky_choices_percent': (risky_choices / len(model.agents)) * 100,
            'agents_above_reference': above_ref,
            'position': position
        })
    
    df = pd.DataFrame(results)
    print("Results:")
    print(df[['reference_point', 'risky_choices_percent', 'position']])
    return df

def run_certainty_effect_experiment():
    """Experiment 4: Certainty Effect with better balanced values"""
    print("\n=== Experiment 4: Certainty Effect ===")
    
    results = []
    
    # Scenario A: Classic certainty effect - make the risky option more attractive
    model_a = ProspectTheoryModel(N=100)
    scenario_a_risky = 0
    for agent in model_a.agents:
        certain = (3000, 1.0)      # $3000 for sure
        probable = (4000, 0.85)    # 85% chance of $4000 (increased from 80%)
        choice, _ = agent.make_decision(certain, probable, "Certain $3000 vs 85% $4000")
        if choice == "B":
            scenario_a_risky += 1
    
    # Scenario B: When no certainty, use values where risky is actually better
    model_b = ProspectTheoryModel(N=100)
    scenario_b_risky = 0
    for agent in model_b.agents:
        probable = (4000, 0.2)     # 20% chance of $4000
        less_probable = (3000, 0.25) # 25% chance of $3000
        choice, _ = agent.make_decision(probable, less_probable, "20% $4000 vs 25% $3000")
        if choice == "B":
            scenario_b_risky += 1
    
    results.append({
        'scenario': 'Certainty Effect',
        'certain_choice_percent': 100 - (scenario_a_risky / len(model_a.agents)) * 100,
        'risky_choice_percent': (scenario_a_risky / len(model_a.agents)) * 100,
        'description': 'Prefer certainty even with lower EV'
    })
    
    results.append({
        'scenario': 'No Certainty',
        'certain_choice_percent': 100 - (scenario_b_risky / len(model_b.agents)) * 100,
        'risky_choice_percent': (scenario_b_risky / len(model_b.agents)) * 100,
        'description': 'Follow expected value more'
    })
    
    df = pd.DataFrame(results)
    print("Results:")
    print(df[['scenario', 'certain_choice_percent', 'risky_choice_percent']])
    return df

def run_probability_weighting_experiment():
    """Experiment 5: Test probability weighting with small vs large probabilities"""
    print("\n=== Experiment 5: Probability Weighting ===")
    
    results = []
    
    # Test small probabilities (usually overweighted)
    model_small = ProspectTheoryModel(N=100)
    small_prob_risky = 0
    for agent in model_small.agents:
        sure_thing = (5, 1.0)       # Sure $5
        small_chance = (100, 0.05)  # 5% chance of $100
        choice, _ = agent.make_decision(sure_thing, small_chance, "5% chance of $100")
        if choice == "B":
            small_prob_risky += 1
    
    # Test medium probabilities
    model_medium = ProspectTheoryModel(N=100)
    medium_prob_risky = 0
    for agent in model_medium.agents:
        sure_thing = (50, 1.0)      # Sure $50
        medium_chance = (100, 0.5)  # 50% chance of $100
        choice, _ = agent.make_decision(sure_thing, medium_chance, "50% chance of $100")
        if choice == "B":
            medium_prob_risky += 1
    
    # Test large probabilities (usually underweighted)
    model_large = ProspectTheoryModel(N=100)
    large_prob_risky = 0
    for agent in model_large.agents:
        sure_thing = (95, 1.0)      # Sure $95
        large_chance = (100, 0.95)  # 95% chance of $100
        choice, _ = agent.make_decision(sure_thing, large_chance, "95% chance of $100")
        if choice == "B":
            large_prob_risky += 1
    
    results.append({
        'probability_type': 'Small (5%)',
        'risky_choice_percent': (small_prob_risky / len(model_small.agents)) * 100,
        'description': 'Usually overweight small probabilities'
    })
    
    results.append({
        'probability_type': 'Medium (50%)',
        'risky_choice_percent': (medium_prob_risky / len(model_medium.agents)) * 100,
        'description': 'Fair probability weighting'
    })
    
    results.append({
        'probability_type': 'Large (95%)',
        'risky_choice_percent': (large_prob_risky / len(model_large.agents)) * 100,
        'description': 'Usually underweight large probabilities'
    })
    
    df = pd.DataFrame(results)
    print("Results:")
    print(df[['probability_type', 'risky_choice_percent', 'description']])
    return df

def main():
    """Run all experiments"""
    print("Running Final Improved Prospect Theory Experiments")
    print("=" * 60)
    
    # Run experiments
    exp1_results = run_loss_aversion_experiment()
    exp2_results = run_framing_experiment()
    exp3_results = run_reference_point_experiment()
    exp4_results = run_certainty_effect_experiment()
    exp5_results = run_probability_weighting_experiment()
    
    print(f"\nAll experiments completed!")

if __name__ == "__main__":
    main()