from pathlib import Path


RESULTS_ROOT = "results" # DO NOT CHANGE THIS

EXPERIMENT_NAME = "prospect_theory_experiment" 
METRICS_FILENAME = "prospect_metrics.csv"
AGENT_FILENAME = "prospect_agents.csv"

EXP_DIR = Path(RESULTS_ROOT) / EXPERIMENT_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE_PATH = EXP_DIR / METRICS_FILENAME
AGENT_FILE_PATH = EXP_DIR / AGENT_FILENAME

# Model Size (number of agents)
NUM_AGENTS = 100

# Network and Simulation Settings
GRAPH_TYPE = "mixed"
STEPS = 300
SEED=67
K_EXPOSURES = 8
AVG_DEGREE = 10
BETA = 3.0

# Behavioral Parameters
OPENNESS = 0.50
TOLERANCE = 0.35
TOLERANCE_JITTER = 0.05
STUBBORNNESS = 0.15

# Prospect Theory Specific Parameters
LOSS_AVERSION = 2.0  # λ coefficient (1.0 = no loss aversion, 2.0 = standard)
RISK_AVERSION = 0.88  # α coefficient for value function curvature
# Belief distribution configuration - Change this too adjust agnts for each type
BELIEF_DISTRIBUTION = "trimodal"  # Options: trimodal, asymmetric_shift, asymmetric_extremists, skewed_positive