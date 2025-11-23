from model.model import SocialBeliefModel
from configs.control_configs import (
    GRAPH_TYPE,
    BELIEF_INITIALIZATION,
    NUM_AGENTS,
    STEPS,
    AVG_DEGREE,
    K_EXPOSURES,
    BETA,
    OPENNESS,
    TOLERANCE,
    TOLERANCE_JITTER,
    STUBBORNNESS
)

m = SocialBeliefModel(
    N=NUM_AGENTS,
    graph=GRAPH_TYPE,
    avg_degree=AVG_DEGREE,
    steps=STEPS,
    seed=1,
    openness=OPENNESS,
    tolerance=TOLERANCE,
    stubbornness=STUBBORNNESS,
    tolerance_jitter=TOLERANCE_JITTER,
    k_exposures=K_EXPOSURES,
    beta=BETA,
    belief_initialization=BELIEF_INITIALIZATION
)

df, _ = m.run()
print(df.head(STEPS + 1))