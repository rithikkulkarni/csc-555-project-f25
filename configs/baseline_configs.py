from pathlib import Path


RESULTS_ROOT = "results" # DO NOT CHANGE THIS
EXPERIMENT_NAME = "<experiment_name>" # Change when running entirely new experiment

METRICS_FILENAME = "<metrics_filename>.csv" # Change each time you run the simulation
AGENT_FILENAME = "<agent_log_filename>.csv" # Change each time you run the simulation

EXP_DIR = Path(RESULTS_ROOT) / EXPERIMENT_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE_PATH = EXP_DIR / METRICS_FILENAME
AGENT_FILE_PATH = EXP_DIR / AGENT_FILENAME

# Model Size (number of agents)
NUM_AGENTS = 100

# Network and Simulation Settings
GRAPH_TYPE = "mixed"
STEPS = 300
SEED=1
K_EXPOSURES = 8
AVG_DEGREE = 10
BETA = 3.0

# Behavioral Parameters
OPENNESS = 0.50
TOLERANCE = 0.35
TOLERANCE_JITTER = 0.05
STUBBORNNESS = 0.15