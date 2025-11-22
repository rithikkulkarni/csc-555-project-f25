import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

# Seed folders
seed_names = [
    "first_seed",
    "second_seed",
    "third_seed",
    "fourth_seed",
    "fifth_seed",
]

WEAK_TIE_DIR = "granovetter_10_percent_runs"

# Belief types
belief_types = {
    "type1": "Trimodal",
    "type2": "Asymmetric Shift Right",
    "type3": "Asymmetric Extremist",
    "type4": "Right Skewed",
}

REGIME = "mixed"

# Metrics to visualize
METRICS_TO_PLOT = {
    "mean_belief": "Mean Belief",
    "polarization_var": "Belief Variance",
    "share_extremes": "Share of Extremists",
    "assortativity": "Assortativity (Homophily)",
}

OUTPUT_ROOT = "plotted_project_results"
OUT_DIR = os.path.join(OUTPUT_ROOT, "granovetter_10_percent_runs_mixed_type_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)


def aggregate_for_belief_type(btype_key: str):
    """
    For a given belief initialization type (type1..type4),
    load all seeds for the mixed regime at 10% weak-tie fraction,
    and return a grouped DataFrame with mean/std over seeds.
    """
    belief_num = btype_key[-1]
    dfs = []

    for seed in seed_names:
        filename = f"type_{belief_num}_{REGIME}_experiment_metrics.csv"
        file_path = os.path.join(RESULTS_DIR, seed, WEAK_TIE_DIR, filename)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Keep only the numeric columns for aggregation
            num_df = df.select_dtypes(include=[np.number])
            dfs.append(num_df)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    grouped = combined.groupby("step").agg(["mean", "std"])
    grouped.columns = ["_".join(col) for col in grouped.columns]
    return grouped


def plot_metric_comparison(metric: str, label: str):
    """
    Create one plot comparing type1..type4 for the given metric
    on mixed regime at 10% weak-tie fraction, aggregated over seeds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for btype_key, btype_label in belief_types.items():
        grouped = aggregate_for_belief_type(btype_key)
        if grouped is None:
            continue

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in grouped.columns or std_col not in grouped.columns:
            continue

        steps = grouped.index
        mean_vals = grouped[mean_col]
        std_vals = grouped[std_col]

        # Plot mean line for this belief type
        ax.plot(
            steps,
            mean_vals,
            label=btype_label,
            linewidth=2,
        )

        # Plot shaded region for std
        ax.fill_between(
            steps,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.15,
        )

    ax.set_title(f"{label} — Mixed Regime, 10% Weak Ties")
    ax.set_xlabel("Step")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"mixed_10pct_{metric}_type_comparison.png")
    fig.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for metric, label in METRICS_TO_PLOT.items():
        plot_metric_comparison(metric, label)
