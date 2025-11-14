import argparse
from experiment_config import METRICS_FILE_PATH, AGENT_FILE_PATH
from model.model import SocialBeliefModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=500, help="Number of agents")
    ap.add_argument("--graph", type=str, default="mixed", choices=["mixed", "echo", "curated"], help="Network/feed regime")
    ap.add_argument("--avg-degree", type=int, default=10, help="Approx. average degree for base graph")
    ap.add_argument("--steps", type=int, default=200, help="Steps to simulate")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--openness", type=float, default=0.5, help="Weight on peers vs self (0..1)")
    ap.add_argument("--tolerance", type=float, default=0.3, help="Max distance accepted in bounded confidence (0..1)")
    ap.add_argument("--tolerance-jitter", type=float, default=0.05, help="Stddev for individual tolerance heterogeneity")
    ap.add_argument("--stubbornness", type=float, default=0.1, help="Damping on movement toward target (0..1)")
    ap.add_argument("--k-exposures", type=int, default=8, help="Number of posts/peers exposed to per step")
    ap.add_argument("--beta", type=float, default=3.0, help="Similarity bias for curated feeds (higher = stronger)")

    args = ap.parse_args()

    model = SocialBeliefModel(
        N=args.N,
        graph=args.graph,
        avg_degree=args.avg_degree,
        steps=args.steps,
        seed=args.seed,
        openness=args.openness,
        tolerance=args.tolerance,
        tolerance_jitter=args.tolerance_jitter,
        stubbornness=args.stubbornness,
        k_exposures=args.k_exposures,
        beta=args.beta,
    )

    model_df, agent_df = model.run(
        steps=args.steps,
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