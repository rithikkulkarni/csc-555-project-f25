from model.model import SocialBeliefModel
from configs.granovetter_configs import (
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
    STUBBORNNESS,
    STRONG_TIE_WEIGHT,
    WEAK_TIE_WEIGHT,
    WEAK_TIE_FRACTION,
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
    belief_initialization=BELIEF_INITIALIZATION,
    strong_tie_weight=STRONG_TIE_WEIGHT,
    weak_tie_weight=WEAK_TIE_WEIGHT,
    weak_tie_fraction=WEAK_TIE_FRACTION,
)

df, _ = m.run()
print(df.head())