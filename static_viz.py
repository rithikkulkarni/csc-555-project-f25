# static_viz.py
import matplotlib.pyplot as plt
import pandas as pd
from model.model import SocialBeliefModel

def create_visualization():
    # Run a simulation
    print("Running simulation for visualization...")
    model = SocialBeliefModel(N=100, steps=200, graph='mixed')
    df, _ = model.run()
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    metrics = [
        ('mean_belief', 'Mean Belief', 'blue'),
        ('polarization_var', 'Polarization', 'red'), 
        ('share_extremes', 'Share of Extremes', 'green'),
        ('assortativity', 'Assortativity', 'purple')
    ]
    
    for idx, (metric, title, color) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.plot(df['step'], df[metric], color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('belief_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'belief_dynamics.png'")
    print(f"Final stats - Mean: {df['mean_belief'].iloc[-1]:.3f}, "
          f"Polarization: {df['polarization_var'].iloc[-1]:.3f}")

if __name__ == "__main__":
    create_visualization()