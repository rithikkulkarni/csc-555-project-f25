from pathlib import Path


RESULTS_ROOT = "results" # DO NOT CHANGE THIS
EXPERIMENT_NAME = "granovetter_5_percent_runs" # Change when running entirely new experiment

METRICS_FILENAME = "type_4_curated_experiment_metrics.csv" # Change each time you run the simulation
AGENT_FILENAME = "type_4_curated_experiment_agent_log.csv" # Change each time you run the simulation

EXP_DIR = Path(RESULTS_ROOT) / EXPERIMENT_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE_PATH = EXP_DIR / METRICS_FILENAME
AGENT_FILE_PATH = EXP_DIR / AGENT_FILENAME

# Model Size (number of agents)
NUM_AGENTS = 100

# Network and Simulation Settings
GRAPH_TYPE = "curated"
STEPS = 300
SEED=1
K_EXPOSURES = 8
AVG_DEGREE = 10
BETA = 3.0
BELIEF_INITIALIZATION = 1

# Behavioral Parameters
OPENNESS = 0.50
TOLERANCE = 0.35
TOLERANCE_JITTER = 0.05
STUBBORNNESS = 0.15

# Granovetter parameters
STRONG_TIE_WEIGHT = 1.0
WEAK_TIE_WEIGHT = 0.2
WEAK_TIE_FRACTION = 0.05