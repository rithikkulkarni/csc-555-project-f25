#!/usr/bin/env bash
set -euo pipefail

# Root directory for all results
RESULTS_ROOT="results"

# Map seed number -> folder name
declare -A SEED_LABEL
SEED_LABEL[1]="first_seed"
SEED_LABEL[2]="second_seed"
SEED_LABEL[3]="third_seed"
SEED_LABEL[4]="fourth_seed"
SEED_LABEL[5]="fifth_seed"

# Main loops:
#  5 seeds
#  4 belief initializations
#  3 graph regimes
for SEED in 1 2 3 4 5; do
  SEED_DIRNAME="${SEED_LABEL[$SEED]}"
  SEED_DIR="${RESULTS_ROOT}/${SEED_DIRNAME}"
  mkdir -p "${SEED_DIR}"

  # Folder for this control experiment under each seed
  EXP_DIR="${SEED_DIR}/control_experiment"
  mkdir -p "${EXP_DIR}"

  for BELIEF_INIT in 1 2 3 4; do
    for REGIME in mixed echo curated; do
      METRICS_FILE="${EXP_DIR}/type_${BELIEF_INIT}_${REGIME}_control_metrics.csv"

      echo "Running CONTROL: seed=${SEED}, belief_init=${BELIEF_INIT}, regime=${REGIME}"
      echo "  -> ${METRICS_FILE}"

      # Pass parameters through environment variables into Python
      SEED_ENV="${SEED}" \
      GRAPH_ENV="${REGIME}" \
      BELIEF_INIT_ENV="${BELIEF_INIT}" \
      METRICS_PATH_ENV="${METRICS_FILE}" \
      python - << 'PY'
import os
import os.path

from model.model import SocialBeliefModel
from configs.control_configs import (
    NUM_AGENTS,
    STEPS,
    AVG_DEGREE,
    K_EXPOSURES,
    BETA,
    OPENNESS,
    TOLERANCE,
    TOLERANCE_JITTER,
    STUBBORNNESS,
)

# Read parameters from environment
seed = int(os.environ["SEED_ENV"])
graph = os.environ["GRAPH_ENV"]
belief_init = int(os.environ["BELIEF_INIT_ENV"])
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
)

model_df, _ = m.run()  # agent_df is ignored; not logged
model_df.to_csv(metrics_path, index=False)
PY

    done
  done
done

echo "All CONTROL runs complete."
