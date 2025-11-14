import argparse
from configs.granovetter_configs import (
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
    STRONG_TIE_WEIGHT,
    WEAK_TIE_WEIGHT,
)

from model.model import SocialBeliefModel

def main():
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
        strong_tie_weight=STRONG_TIE_WEIGHT,
        weak_tie_weight=WEAK_TIE_WEIGHT,
    )


    model_df, agent_df = model.run(
        steps=STEPS,
        agent_log_path=str(AGENT_FILE_PATH),
    )

    model_df.to_csv(METRICS_FILE_PATH, index=False)

    print("\nFinished! Saved:")
    print(f"  - {METRICS_FILE_PATH} (per-step metrics)")
    if AGENT_FILE_PATH:
        print(f"  - {METRICS_FILE_PATH} (per-agent trajectories)")
    print("\nColumns in per-step metrics:\n", list(model_df.columns))

if __name__ == "__main__":
    main()