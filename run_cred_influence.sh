#!/usr/bin/env bash
set -euo pipefail

# Root directory for all results (keep matching your config)
RESULTS_ROOT="results"

# Map seed number -> folder name
declare -A SEED_LABEL=(
  [1]="first_seed"
  [2]="second_seed"
  [3]="third_seed"
  [4]="fourth_seed"
  [5]="fifth_seed"
)

# Credibility distribution modes
# cred_distribution = 1 → continuous credibility (normal distribution)
# cred_distribution = 2 → discrete high/low credibility
declare -A CRED_MODE=(
  [1]="normally_distributed_credibility"
  [2]="discrete_credibility"
)

# ---- Experiment Sweep ----
# 5 seeds × 2 credibility distributions × 4 belief inits × 3 regimes
for SEED in 1 2 3 4 5; do
  SEED_DIRNAME="${SEED_LABEL[$SEED]}"
  SEED_DIR="${RESULTS_ROOT}/${SEED_DIRNAME}"
  mkdir -p "${SEED_DIR}"

  # Two folders per seed:
  # - discrete_credibility/
  # - normally_distributed_credibility/
  for CRED_DIST in 1 2; do
    EXP_NAME="${CRED_MODE[$CRED_DIST]}"
    EXP_DIR="${SEED_DIR}/${EXP_NAME}"
    mkdir -p "${EXP_DIR}"

    for BELIEF_INIT in 1 2 3 4; do
      BTYPE="type${BELIEF_INIT}"

      for REGIME in mixed echo curated; do
        METRICS_FILE="${EXP_DIR}/${BTYPE}_${REGIME}_metrics.csv"

        echo "Running: seed=${SEED}, cred_dist=${CRED_DIST} (${EXP_NAME}), belief_init=${BELIEF_INIT}, regime=${REGIME}"
        echo "  → Writing to ${METRICS_FILE}"

        SEED_ENV="${SEED}" \
        GRAPH_ENV="${REGIME}" \
        BELIEF_INIT_ENV="${BELIEF_INIT}" \
        CRED_DIST_ENV="${CRED_DIST}" \
        METRICS_PATH_ENV="${METRICS_FILE}" \
        python - << 'PY'
import os
from model.model import SocialBeliefModel
from configs.credibility_influence_configs import (
    NUM_AGENTS, STEPS, AVG_DEGREE, K_EXPOSURES, BETA,
    OPENNESS, TOLERANCE, TOLERANCE_JITTER,
    STUBBORNNESS, HIGH_CRED_FRACTION,
    HIGH_CREDIBILITY, LOW_CREDIBILITY,
)

seed = int(os.environ["SEED_ENV"])
graph = os.environ["GRAPH_ENV"]
belief_init = int(os.environ["BELIEF_INIT_ENV"])
cred_dist = int(os.environ["CRED_DIST_ENV"])
metrics_path = os.environ["METRICS_PATH_ENV"]

# Ensure directory exists
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

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
    high_cred_fraction=HIGH_CRED_FRACTION,
    high_credibility=HIGH_CREDIBILITY,
    low_credibility=LOW_CREDIBILITY,
    cred_distribution=cred_dist,
)

df, _ = m.run()
df.to_csv(metrics_path, index=False)
PY

      done
    done
  done
done

echo "All credibility/influence simulations complete."
