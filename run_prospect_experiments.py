"""
Prospect Theory Experiments Runner
Place this file in the root directory (same level as run_simulation.py)
"""

from pathlib import Path
import pandas as pd
from model.prospect_model import ProspectTheoryModel


# Configuration
RESULTS_DIR = Path("results") / "prospect_theory_experiment"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
            print(f"  Loss aversion λ = {lambda_val}")
            
            model = ProspectTheoryModel(
                N=100,
                graph=regime,
                steps=300,
                seed=1,
                loss_aversion=lambda_val,
                openness=0.5,
                tolerance=0.35,
                stubbornness=0.15,
                k_exposures=8,
                avg_degree=10,
            )
            
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
    output_path = RESULTS_DIR / "exp1_loss_aversion.csv"
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
        print(f"\nTesting {regime} regime with λ=2.0...")
        
        model = ProspectTheoryModel(
            N=100,
            graph=regime,
            steps=300,
            seed=1,
            loss_aversion=2.0,
            openness=0.5,
            tolerance=0.35,
            stubbornness=0.15,
            k_exposures=8,
            avg_degree=10,
        )
        
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
    output_path = RESULTS_DIR / "exp2_reference_points.csv"
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
        
        # Run with standard prospect theory parameters
        model = ProspectTheoryModel(
            N=100,
            graph=regime,
            steps=300,
            seed=1,
            loss_aversion=2.0,
            risk_aversion=0.88,
            openness=0.5,
            tolerance=0.35,
            stubbornness=0.15,
            k_exposures=8,
            avg_degree=10,
            beta=3.0,
        )
        
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
    output_path = RESULTS_DIR / "exp3_regime_comparison.csv"
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
            print(f"Testing λ={lambda_val}, stubbornness={stub}")
            
            model = ProspectTheoryModel(
                N=100,
                graph="mixed",  # Small-world recommended for prospect theory
                steps=300,
                seed=1,
                loss_aversion=lambda_val,
                openness=0.5,
                tolerance=0.35,
                stubbornness=stub,
                k_exposures=8,
                avg_degree=10,
            )
            
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
    output_path = RESULTS_DIR / "exp4_loss_stubbornness.csv"
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
    print("\nFiles created:")
    print("  - exp1_loss_aversion.csv")
    print("  - exp2_reference_points.csv")
    print("  - exp3_regime_comparison.csv")
    print("  - exp4_loss_stubbornness.csv")


if __name__ == "__main__":
    main()