"""
Prospect Theory Experiments Runner
Place this file in the root directory (same level as run_simulation.py)
"""

import os
from pathlib import Path
import pandas as pd
from model.prospect_model import ProspectTheoryModel
from configs.prospect_configs import (
    NUM_AGENTS, STEPS, K_EXPOSURES, AVG_DEGREE, BETA,
    OPENNESS, TOLERANCE, TOLERANCE_JITTER, STUBBORNNESS,
    LOSS_AVERSION, RISK_AVERSION, BELIEF_DISTRIBUTION, SEED as CONFIG_SEED
)

# Detect if running from batch script and determine which seed to use
if 'SEED' in os.environ:
    # Running from batch script - use environment variable
    CURRENT_SEED = int(os.environ['SEED'])
else:
    # Running individually - use config file seed
    CURRENT_SEED = CONFIG_SEED

# Configuration - create seed-specific directory
#RESULTS_DIR = Path("results") / "prospect_theory_experiment" / f"prospect_seed{SEED}"
#RESULTS_DIR = Path("results") / "prospect_theory_experiment" / f"seed{SEED}_{BELIEF_DISTRIBUTION}"
RESULTS_DIR = Path("results") / "prospect_theory_experiment" / f"seed{CURRENT_SEED}_{BELIEF_DISTRIBUTION}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def makeModel(
    regime: str, 
    loss_aversion: float = LOSS_AVERSION,
    stubbornness: float = STUBBORNNESS
) -> ProspectTheoryModel:
    """Helper function to create ProspectTheoryModel with consistent parameters"""
    model = ProspectTheoryModel(
        N=NUM_AGENTS,
        graph=regime,
        avg_degree=AVG_DEGREE,
        steps=STEPS,
        #seed=SEED,
        seed=CURRENT_SEED,
        openness=OPENNESS,
        tolerance=TOLERANCE,
        stubbornness=stubbornness,
        tolerance_jitter=TOLERANCE_JITTER,
        k_exposures=K_EXPOSURES,
        beta=BETA,
        loss_aversion=loss_aversion,
        risk_aversion=RISK_AVERSION
    )
    # print(f"Creating model: regime={regime}, loss_aversion={loss_aversion}, stubbornness={stubbornness}")
    return model

def run_loss_aversion_experiment():
    """
    Experiment 1: Test different loss aversion parameters across network regimes.
    
    Hypothesis: Higher loss aversion leads to stronger polarization and resistance
    to belief change, especially in echo chamber networks where agents reinforce
    each other's loss-averse tendencies.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Loss Aversion Effects")
    print("=" * 70)
    
    results = []
    loss_aversion_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    regimes = ["mixed", "echo", "curated"]
    
    for regime in regimes:
        print(f"\nTesting {regime} regime...")
        for lambda_val in loss_aversion_values:
            print(f"  Loss aversion lamba = {lambda_val}")
            model = makeModel(regime=regime, loss_aversion=lambda_val)

            model_df, _ = model.run()
            
            # Get final metrics
            final = model_df.iloc[-1]
            
            results.append({
                'regime': regime,
                'loss_aversion': lambda_val,
                'final_polarization': final['polarization_var'],
                'final_extremes': final['share_extremes'],
                'final_mean_belief': final['mean_belief'],
                'belief_drift': final['belief_drift'],
                'final_assortativity': final['assortativity'],
            })

    df = pd.DataFrame(results)
    output_path = RESULTS_DIR / f"seed{CURRENT_SEED}_{BELIEF_DISTRIBUTION}_prospect_exp1.csv"
    df.to_csv(output_path, index=False)
    
    print("\nSummary by regime:")
    print(df.groupby('regime')[['loss_aversion', 'final_polarization', 'final_extremes']].describe())
    
    return df


def run_reference_point_experiment():
    """
    Experiment 2: Test how belief distribution affects outcomes under loss aversion.
    
    Hypothesis: Asymmetric initial distributions will lead to different polarization
    patterns because agents evaluate changes relative to their initial position.
    Loss-averse agents resist crossing their reference point.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Reference Point Effects")
    print("=" * 70)
    
    results = []
    regimes = ["mixed", "echo", "curated"]
    
    for regime in regimes:
        print(f"\nTesting {regime} regime with lambda =2.0...")
        
        model = makeModel(regime)
        model_df, _ = model.run()
        
        # Calculate how many agents crossed their reference point (initial belief)
        final = model_df.iloc[-1]
        
        # Since we're not saving agent logs, we can't calculate crossings
        crossing_rate = 0.0
        
        results.append({
            'regime': regime,
            'final_polarization': final['polarization_var'],
            'final_extremes': final['share_extremes'],
            'belief_drift': final['belief_drift'],
            'reference_crossing_rate': crossing_rate,
        })
    
    df = pd.DataFrame(results)
    output_path = RESULTS_DIR / f"seed{CURRENT_SEED}_{BELIEF_DISTRIBUTION}_prospect_exp2.csv"
    df.to_csv(output_path, index=False)
    
    print("\nReference Point Crossings:")
    print(df[['regime', 'reference_crossing_rate', 'belief_drift']])
    
    return df


def run_network_regime_comparison():
    """
    Experiment 3: Compare how prospect theory behaves across network structures.
    
    Hypothesis: 
    - Mixed (small-world): Moderate polarization, some bridge agents overcome loss aversion
    - Echo: High polarization, loss aversion reinforced within clusters
    - Curated: Algorithmic exposure can either amplify or reduce loss aversion effects
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Network Regime Comparison")
    print("=" * 70)
    
    results = []
    regimes = ["mixed", "echo", "curated"]
    
    for regime in regimes:
        print(f"\nTesting {regime} regime...")
        
        model = makeModel(regime)
        model_df, _ = model.run()
        
        # Track evolution over time
        for step_idx in range(300):
            row = model_df.iloc[step_idx]
            results.append({
                'regime': regime,
                'step': row['step'],
                'mean_belief': row['mean_belief'],
                'polarization_var': row['polarization_var'],
                'share_extremes': row['share_extremes'],
                'assortativity': row['assortativity'],
                'belief_drift': row['belief_drift'],
            })
    
    df = pd.DataFrame(results)
    output_path = RESULTS_DIR / f"seed{CURRENT_SEED}_{BELIEF_DISTRIBUTION}_prospect_exp3.csv"
    df.to_csv(output_path, index=False)
    
    print("\nFinal state by regime:")
    final_df = df[df['step'] == 299]
    print(final_df[['regime', 'polarization_var', 'share_extremes', 'belief_drift']])
    
    return df


def run_loss_aversion_vs_stubbornness():
    """
    Experiment 4: Interaction between loss aversion and stubbornness.
    
    Hypothesis: Loss aversion and stubbornness have compounding effects.
    High loss aversion + high stubbornness = extreme resistance to change.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Loss Aversion × Stubbornness Interaction")
    print("=" * 70)
    
    results = []
    loss_aversion_values = [1.0, 2.0, 3.0]
    stubbornness_values = [0.05, 0.15, 0.30]
    
    for lambda_val in loss_aversion_values:
        for stub in stubbornness_values:
            print(f"Testing lambda={lambda_val}, stubbornness={stub}")
            # Pass both custom parameters
            model = makeModel(regime="mixed", loss_aversion=lambda_val, stubbornness=stub)
            model_df, _ = model.run()
            final = model_df.iloc[-1]
            
            results.append({
                'loss_aversion': lambda_val,
                'stubbornness': stub,
                'final_polarization': final['polarization_var'],
                'final_extremes': final['share_extremes'],
                'belief_drift': final['belief_drift'],
            })
    
    df = pd.DataFrame(results)
    output_path = RESULTS_DIR / f"seed{CURRENT_SEED}_{BELIEF_DISTRIBUTION}_prospect_exp4.csv"
    df.to_csv(output_path, index=False)
    
    print("\nInteraction effects:")
    print(df.pivot_table(
        values='final_polarization',
        index='stubbornness',
        columns='loss_aversion'
    ))
    
    return df


def main():
    """Run all prospect theory experiments"""
    print("\n" + "=" * 70)
    print("PROSPECT THEORY EXPERIMENTS - MESA OPINION DYNAMICS")
    print("=" * 70)
    print("\nThese experiments test how Kahneman & Tversky's Prospect Theory")
    print("principles affect opinion dynamics in social networks:\n")
    print("  1. Loss Aversion: Changes away from reference point hurt more")
    print("  2. Reference Dependence: Evaluated relative to initial belief")
    print("  3. Probability Weighting: Over/under-weight social influence")
    print("  4. Diminishing Sensitivity: Small changes matter more\n")
    
    # Run all experiments
    exp1_results = run_loss_aversion_experiment()
    exp2_results = run_reference_point_experiment()
    exp3_results = run_network_regime_comparison()
    exp4_results = run_loss_aversion_vs_stubbornness()
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR.absolute()}")
    print(f"\nFiles created (seed={CURRENT_SEED}):")
    print(f"  - seed{CURRENT_SEED}_{BELIEF_DISTRIBUTION}_prospect_exp1.csv")
    print(f"  - seed{CURRENT_SEED}_{BELIEF_DISTRIBUTION}_prospect_exp2.csv")
    print(f"  - seed{CURRENT_SEED}_{BELIEF_DISTRIBUTION}_prospect_exp3.csv")
    print(f"  - seed{CURRENT_SEED}_{BELIEF_DISTRIBUTION}_prospect_exp4.csv")

if __name__ == "__main__":
    main()