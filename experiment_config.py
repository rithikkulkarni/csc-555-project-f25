from pathlib import Path


RESULTS_ROOT = "results" # DO NOT CHANGE THIS
EXPERIMENT_NAME = "test_4" # Change when running entirely new experiment

METRICS_FILENAME = "control_mixed.csv" # Change each time you run the simulation
AGENT_FILENAME = "test4.csv" # Change each time you run the simulation

EXP_DIR = Path(RESULTS_ROOT) / EXPERIMENT_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE_PATH = EXP_DIR / METRICS_FILENAME
AGENT_FILE_PATH = EXP_DIR / AGENT_FILENAME