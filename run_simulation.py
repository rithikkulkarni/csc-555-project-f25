import argparse
from configs.credibility_influence_configs import (
    METRICS_FILE_PATH,
    AGENT_FILE_PATH,
    NUM_AGENTS,
    GRAPH_TYPE,
    STEPS,
    SEED,
    K_EXPOSURES,
    AVG_DEGREE,
    OPENNESS,
    TOLERANCE,
    TOLERANCE_JITTER,
    STUBBORNNESS,
    BETA,
    HIGH_CREDIBILITY,
    LOW_CREDIBILITY,
    HIGH_CRED_FRACTION
)

def main():
    print("starting main")
    from model.model import SocialBeliefModel
    print("mesa imported")
     # Create model instance with parameters supplied from config file.
    # These include behavioral prefs, credibility settings, graph type, etc.
    model = SocialBeliefModel(
        N=NUM_AGENTS,
        graph=GRAPH_TYPE,
        avg_degree=AVG_DEGREE,
        steps=STEPS,
        seed=SEED,
        openness=OPENNESS,
        tolerance=TOLERANCE,
        tolerance_jitter=TOLERANCE_JITTER,
        stubbornness=STUBBORNNESS,
        k_exposures=K_EXPOSURES,
        beta=BETA,
        high_credibility=HIGH_CREDIBILITY,
        low_credibility=LOW_CREDIBILITY,
        high_cred_fraction=HIGH_CRED_FRACTION,
    )

    # Run the simulation for STEPS ticks and optionally record agent-level logs
    model_df, agent_df = model.run(
        steps=STEPS,
        agent_log_path=None,
    )

    # Save step-by-step model metrics to a CSV
    model_df.to_csv(METRICS_FILE_PATH, index=False)

    print("\nFinished! Saved:")
    print(f"  - {METRICS_FILE_PATH} (per-step metrics)")
    if AGENT_FILE_PATH:
        print(f"  - {METRICS_FILE_PATH} (per-agent trajectories)")
    print("\nColumns in per-step metrics:\n", list(model_df.columns))

if __name__ == "__main__":
    main()