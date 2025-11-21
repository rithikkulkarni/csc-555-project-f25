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

# Belief types
belief_types = {
    "type1": "Trimodal",
    "type2": "Asymmetric Shift Right",
    "type3": "Asymmetric Extremist",
    "type4": "Right Skewed",
}

# Regimes used in the experiment
regimes = ["mixed", "echo", "curated"]

# Metrics to visualize for the CONTROL experiment
METRIC_LABELS = {
    "mean_belief": "Mean Belief",
    "polarization_var": "Belief Variance (Polarization)",
    "share_extremes": "Share of Extremists (|belief| ≥ 0.9)",
    "assortativity": "Assortativity (Homophily)",
}

OUTPUT_ROOT = "plotted_project_results"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def plot_for_belief_control(btype_key: str):
    """
    Aggregate across seeds for a given belief type in the CONTROL experiment
    (no weak-tie-specific metrics), and plot all metrics over time with
    one line per regime.
    """
    # All control plots go into a single folder
    out_dir = os.path.join(OUTPUT_ROOT, "control_experiment")
    os.makedirs(out_dir, exist_ok=True)

    # 2x2 grid for 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    belief_num = btype_key[-1]  # "type1" -> "1"

    for i, (metric, label) in enumerate(METRIC_LABELS.items()):
        ax = axes[i]

        for regime in regimes:
            dfs = []

            # Collect data across seeds
            for seed in seed_names:
                filename = f"type_{belief_num}_{regime}_control_metrics.csv"
                file_path = os.path.join(
                    RESULTS_DIR, seed, "control_experiment", filename
                )

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

                # Shaded ±1 std band
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

    # Turn off any unused axes (if any, though we used all 4)
    for j in range(len(METRIC_LABELS), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{belief_types[btype_key]} — Control (No Weak-Tie Weighting)",
        fontsize=18,
        y=1.02,
    )
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{btype_key}_control_experiment.png")
    fig.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


# Generate all control plots (one per belief type)
for btype in belief_types.keys():
    plot_for_belief_control(btype)
