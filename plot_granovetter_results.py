import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

# Explicit seed folders
seed_names = [
    "first_seed",
    "second_seed",
    "third_seed",
    "fourth_seed",
    "fifth_seed",
]

# Weak-tie fraction experiment folders
weak_tie_modes = {
    "granovetter_5_percent_runs":  "5% Weak Ties",
    "granovetter_10_percent_runs": "10% Weak Ties",
    "granovetter_20_percent_runs": "20% Weak Ties",
}

# Belief types
belief_types = {
    "type1": "Trimodal",
    "type2": "Asymmetric Shift Right",
    "type3": "Asymmetric Extremist",
    "type4": "Right Skewed",
}

# Regimes used in the experiment
regimes = ["mixed", "echo", "curated"]

# Metrics to visualize (all numeric except regime + weak_tie_fraction)
METRIC_LABELS = {
    "mean_belief": "Mean Belief",
    "polarization_var": "Belief Variance",
    "share_extremes": "Share of Extremists",
    "assortativity": "Assortativity (Homophily)",
    "weak_edge_disagreement": "Weak-Edge Disagreement",
    "strong_edge_disagreement": "Strong-Edge Disagreement",
    "inter_cluster_gap": "Inter-Cluster Gap",
    "weak_tie_activations": "Weak Tie Activations",
    "strong_tie_activations": "Strong Tie Activations",
    "weak_activation_share": "Weak Activation Share",
    "adoption_share": "Adoption Share",
}

OUTPUT_ROOT = "plotted_project_results"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def plot_for_belief_and_weak_ties(btype_key: str, weak_mode_dir: str):

    out_dir = os.path.join(OUTPUT_ROOT, weak_mode_dir)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.flatten()

    belief_num = btype_key[-1]   # "type1" -> "1"

    for i, (metric, label) in enumerate(METRIC_LABELS.items()):
        ax = axes[i]

        for regime in regimes:
            dfs = []

            # ✓ Explicitly iterate over the 5 seed folders
            for seed in seed_names:

                filename = f"type_{belief_num}_{regime}_experiment_metrics.csv"
                file_path = os.path.join(RESULTS_DIR, seed, weak_mode_dir, filename)

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    dfs.append(df.select_dtypes(include=[np.number]))

            if not dfs:
                continue

            combined = pd.concat(dfs, ignore_index=True)
            grouped = combined.groupby("step").agg(["mean", "std"])
            grouped.columns = ["_".join(col) for col in grouped.columns]

            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col in grouped and std_col in grouped:

                # plot mean line
                ax.plot(
                    grouped.index, grouped[mean_col],
                    label=regime.capitalize(), linewidth=2
                )

                # plot ± std shaded region
                ax.fill_between(
                    grouped.index,
                    grouped[mean_col] - grouped[std_col],
                    grouped[mean_col] + grouped[std_col],
                    alpha=0.15
                )

        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()

    for j in range(len(METRIC_LABELS), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{belief_types[btype_key]} — {weak_tie_modes[weak_mode_dir]}",
        fontsize=18, y=1.02
    )
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{btype_key}_{weak_mode_dir}.png")
    fig.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


# Generate all plots
for weak_mode_dir in weak_tie_modes:
    for btype in belief_types:
        plot_for_belief_and_weak_ties(btype, weak_mode_dir)
