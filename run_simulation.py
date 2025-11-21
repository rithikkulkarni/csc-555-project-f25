from model.model import SocialBeliefModel
from configs.credibility_influence_configs import (
    GRAPH_TYPE, BELIEF_INITIALIZATION, CRED_DISTRIBUTION,
    NUM_AGENTS, STEPS, AVG_DEGREE, K_EXPOSURES,
    BETA, OPENNESS, TOLERANCE, TOLERANCE_JITTER,
    STUBBORNNESS, HIGH_CRED_FRACTION, HIGH_CREDIBILITY, LOW_CREDIBILITY
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
    high_cred_fraction=HIGH_CRED_FRACTION,
    high_credibility=HIGH_CREDIBILITY,
    low_credibility=LOW_CREDIBILITY,
    cred_distribution=CRED_DISTRIBUTION,
)

df, _ = m.run()
print(df)
