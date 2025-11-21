Credibility & Influence Simulation

This readme explains how to run the credibility + influence experiment codebase.
First, download and extract the zip file. Then cd into its root directory.

Create and activate a virtual environment:

    python3 -m venv venv
    source venv/bin/activate      -> macOS/Linux
    venv\Scripts\activate         -> Windows

If you have scripts disabled on your terminal (common for Windows), you may 
have to temporarily bypass it using this command:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then try activating the environment again.

Install required Python dependencies:

    pip install -r requirements.txt

-----------------------------------------------------

The batch script sweeps over:

- 5 random seeds (1–5)
- 2 credibility modes:
    • normally_distributed_credibility (continuous credibility)
    • discrete_credibility (high/low credibility)
- 4 belief initialization modes
- 3 graph regimes: mixed, echo, curated

This produces 120 simulations.

Run the experiment:

    ./run_cred_influence.sh

-----------------------------------------------------

After running the batch script, results will appear under:

results/
  first_seed/
    discrete_credibility/
      type1_mixed_metrics.csv
      type1_echo_metrics.csv
      type1_curated_metrics.csv
      ...
    normally_distributed_credibility/
      type1_mixed_metrics.csv
      ...
  second_seed/
  third_seed/
  fourth_seed/
  fifth_seed/

Each file contains aggregated model-level metrics for a single simulation run.

-----------------------------------------------------

After simulations complete, run:

    python plot_cred_influence.py

This script:

- Aggregates all 5 seeds
- Computes means and standard deviations
- Plots curves for each regime (mixed/echo/curated)
- Saves charts to:

plotted_project_results/
  discrete_credibility/
    type1_discrete_credibility.png
    type2_discrete_credibility.png
    ...
  normally_distributed_credibility/
    type1_normally_distributed_credibility.png
    ...

Each PNG summarizes all metrics for that belief type + credibility mode.

-----------------------------------------------------

You can manually run a single simulation to test your setup with the command:

    python run_simulation.py
    
This script will use the default parameters set in configs/granovetter_configs.py and then print the resulting metrics dataframe to the terminal.