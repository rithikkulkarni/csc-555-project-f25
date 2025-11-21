import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Where the CSV metrics files live
RESULTS_DIR = "results"

# Where to save the PNG figures
OUTPUT_ROOT = "plotted_project_results"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Seed folders
seed_names = [
    "first_seed",
    "second_seed",
    "third_seed",
    "fourth_seed",
    "fifth_seed",
]

# Credibility modes
credibility_modes = [
    "discrete_credibility",
    "normally_distributed_credibility",
]

# Belief types
belief_types = {
    "type1": "Trimodal",
    "type2": "Asymmetric Shift Right",
    "type3": "Asymmetric Extremist",
    "type4": "Right Skewed",
}

# Regimes
regimes = ["mixed", "echo", "curated"]

# Metrics from the experiment
METRIC_LABELS = {
    "mean_belief": "Mean Belief",
    "polarization_var": "Belief Variance (Polarization)",
    "share_extremes": "Share of Extremists",
    "assortativity": "Assortativity (Homophily)",
    "mean_influence": "Mean Influence",
    "var_influence": "Influence Variance",
    "mean_cred_weighted_influence": "Credibility-Weighted Influence",
    "influence_gini": "Influence Gini (Inequality)",
    "elite_fraction": "Top-10% Elite Fraction",
}


def plot_for_belief_and_cred(btype_key: str, cred_mode: str):
    """
    Aggregate across seeds for a given belief type and credibility mode,
    and plot all metrics over time with one line per regime.
    """
    # Make output subfolder for this credibility mode
    out_dir = os.path.join(OUTPUT_ROOT, cred_mode)
    os.makedirs(out_dir, exist_ok=True)

    # 3x3 grid for the 9 metrics
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    belief_num = btype_key[-1]

    for i, (metric, label) in enumerate(METRIC_LABELS.items()):
        ax = axes[i]

        for regime in regimes:
            dfs = []

            # Collect data across seeds
            for seed in seed_names:
                filename = f"{btype_key}_{regime}_metrics.csv"
                file_path = os.path.join(RESULTS_DIR, seed, cred_mode, filename)

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)

                    # Keep only numeric columns so aggregation works cleanly
                    num_df = df.select_dtypes(include=[np.number])
                    dfs.append(num_df)

            if not dfs:
                continue

            combined = pd.concat(dfs, ignore_index=True)
            grouped = combined.groupby("step").agg(["mean", "std"])
            grouped.columns = ["_".join(col) for col in grouped.columns]

            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col in grouped.columns and std_col in grouped.columns:
                # Plot mean across seeds
                ax.plot(
                    grouped.index,
                    grouped[mean_col],
                    label=regime.capitalize(),
                    linewidth=2,
                )
                
                ax.fill_between(
                    grouped.index,
                    grouped[mean_col] - grouped[std_col],
                    grouped[mean_col] + grouped[std_col],
                    alpha=0.15,
                )

        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"{belief_types[btype_key]} — {cred_mode.replace('_', ' ').title()}",
        fontsize=18,
        y=1.02,
    )
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{btype_key}_{cred_mode}.png")
    fig.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# Main loop: all credibility modes × all belief types
for cred_mode in credibility_modes:
    for btype in belief_types.keys():
        plot_for_belief_and_cred(btype, cred_mode)
