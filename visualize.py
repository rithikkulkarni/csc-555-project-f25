from __future__ import annotations
import numpy as np
from mesa.visualization import SolaraViz, make_space_component, make_mpl_plot_component, Slider, Choice
from mesa.visualization.space import SpaceRenderer
from sim import SocialBeliefModel

# Map belief [-1,1] → RGB
def belief_color(b: float) -> tuple[int, int, int]:
    r = (b + 1) / 2.0          # -1 -> 0, +1 -> 1
    g = 1.0 - abs(b)           # 0 strongest green
    bl = (1 - b) / 2.0
    return int(r*255), int(g*255), int(bl*255)

# Portray each agent (x,y are auto for NetworkGrid; we only set color/size)
def agent_portrayal(agent):
    r, g, bl = belief_color(agent.belief)
    return dict(color=(r, g, bl), size=30)

# Tell the renderer we want a NETWORK space (it will read model.grid)
renderer = SpaceRenderer(model=None, backend="matplotlib")        # model gets injected by SolaraViz
space_component = make_space_component(renderer, space_type="network", agent_portrayal=agent_portrayal)

# Matplotlib plot of model measures collected by your DataCollector
plot_component, _ = make_mpl_plot_component(["mean_belief", "polarization_var", "share_extremes"])

# Exposed parameters (sliders / choices show up in the UI)
model_params = dict(
    N=Slider("Number of agents", value=100, min=50, max=1000, step=50),
    graph=Choice("Graph Regime", value="mixed", choices=["mixed", "echo", "curated"]),
    avg_degree=Slider("Avg Degree", value=10, min=2, max=30, step=1),
    steps=Slider("Steps", value=300, min=50, max=1000, step=10),
    openness=Slider("Openness", value=0.5, min=0.0, max=1.0, step=0.05),
    tolerance=Slider("Tolerance", value=0.3, min=0.0, max=1.0, step=0.05),
    tolerance_jitter=Slider("Tol Jitter", value=0.05, min=0.0, max=0.5, step=0.01),
    stubbornness=Slider("Stubbornness", value=0.1, min=0.0, max=1.0, step=0.05),
    k_exposures=Slider("k exposures", value=8, min=1, max=30, step=1),
    beta=Slider("Curation β", value=3.0, min=0.0, max=6.0, step=0.5),
)

viz = SolaraViz(
    model_cls=SocialBeliefModel,
    model_params=model_params,
    measures=["mean_belief", "polarization_var", "share_extremes", "assortativity"],
    components=[space_component, plot_component],
)

if __name__ == "__main__":
    # Launches a small app at http://localhost:8765
    viz.launch()