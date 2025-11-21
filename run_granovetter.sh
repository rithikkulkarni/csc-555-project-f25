#!/usr/bin/env bash
set -euo pipefail

# Root directory for all results
RESULTS_ROOT="results"

# Names for seed folders in results/
declare -A SEED_LABEL
SEED_LABEL[1]="first_seed"
SEED_LABEL[2]="second_seed"
SEED_LABEL[3]="third_seed"
SEED_LABEL[4]="fourth_seed"
SEED_LABEL[5]="fifth_seed"

# Weak tie fractions
WEAK_FRACS=("0.05" "0.10" "0.20")

weak_frac_label() {
  case "$1" in
    0.05) echo "5" ;;
    0.10) echo "10" ;;
    0.20) echo "20" ;;
    *) echo "unknown" ;;
  esac
}

# Loops:
#  5 seeds
#  3 weak tie fractions
#  4 belief initializations
#  3 graph regimes
for SEED in 1 2 3 4 5; do
  SEED_DIRNAME="${SEED_LABEL[$SEED]}"
  SEED_DIR="${RESULTS_ROOT}/${SEED_DIRNAME}"
  mkdir -p "${SEED_DIR}"

  for WEAK_FRAC in "${WEAK_FRACS[@]}"; do
    PCT_LABEL="$(weak_frac_label "${WEAK_FRAC}")"
    EXP_DIR="${SEED_DIR}/granovetter_${PCT_LABEL}_percent_runs"
    mkdir -p "${EXP_DIR}"

    for BELIEF_INIT in 1 2 3 4; do
      for REGIME in mixed echo curated; do
        METRICS_FILE="${EXP_DIR}/type_${BELIEF_INIT}_${REGIME}_experiment_metrics.csv"

        echo "Running seed=${SEED}, weak_tie_fraction=${WEAK_FRAC}, belief_init=${BELIEF_INIT}, regime=${REGIME}"
        echo "  -> ${METRICS_FILE}"

        SEED_ENV="${SEED}" \
        GRAPH_ENV="${REGIME}" \
        BELIEF_INIT_ENV="${BELIEF_INIT}" \
        WEAK_FRAC_ENV="${WEAK_FRAC}" \
        METRICS_PATH_ENV="${METRICS_FILE}" \
        python - << 'PY'
import os
import os.path

from model.model import SocialBeliefModel
from configs.granovetter_configs import (
    NUM_AGENTS,
    STEPS,
    AVG_DEGREE,
    K_EXPOSURES,
    BETA,
    OPENNESS,
    TOLERANCE,
    TOLERANCE_JITTER,
    STUBBORNNESS,
    STRONG_TIE_WEIGHT,
    WEAK_TIE_WEIGHT,
)

# Read parameters from environment
seed = int(os.environ["SEED_ENV"])
graph = os.environ["GRAPH_ENV"]
belief_init = int(os.environ["BELIEF_INIT_ENV"])
weak_frac = float(os.environ["WEAK_FRAC_ENV"])
metrics_path = os.environ["METRICS_PATH_ENV"]

# Ensure directory exists
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

# Instantiate model with explicit parameters;
# note: we do NOT pass agent_log_path, so no agent CSV is written.
m = SocialBeliefModel(
    N=NUM_AGENTS,
    graph=graph,
    avg_degree=AVG_DEGREE,
    steps=STEPS,
    seed=seed,
    openness=OPENNESS,
    tolerance=TOLERANCE,
    stubbornness=STUBBORNNESS,
    tolerance_jitter=TOLERANCE_JITTER,
    k_exposures=K_EXPOSURES,
    beta=BETA,
    belief_initialization=belief_init,
    strong_tie_weight=STRONG_TIE_WEIGHT,
    weak_tie_weight=WEAK_TIE_WEIGHT,
    weak_tie_fraction=weak_frac,
)

model_df, _ = m.run()  # agent_df is ignored; not logged
model_df.to_csv(metrics_path, index=False)
PY

      done
    done
  done
done

echo "All runs complete."
