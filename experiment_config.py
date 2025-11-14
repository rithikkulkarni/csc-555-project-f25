from pathlib import Path


RESULTS_ROOT = "results" # DO NOT CHANGE THIS
EXPERIMENT_NAME = "example_experiment_2" # Change when running entirely new experiment

METRICS_FILENAME = "control_mixed.csv" # Change each time you run the simulation
AGENT_FILENAME = "example_test_for_experiment_2.csv" # Change each time you run the simulation

EXP_DIR = Path(RESULTS_ROOT) / EXPERIMENT_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE_PATH = EXP_DIR / METRICS_FILENAME
AGENT_FILE_PATH = EXP_DIR / AGENT_FILENAME