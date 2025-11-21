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
- 3 weak-tie fractions:
      • 0.05
      • 0.10
      • 0.20
- 4 belief initialization modes (type1, type2, type3, type4)
- 3 graph regimes:
      • mixed
      • echo
      • curated

This produces 180 simulations.

Run the experiment:

    ./run_granovetter.sh

-----------------------------------------------------

After running the script, results will appear under:

results/
  first_seed/
    granovetter_5_percent_runs/
      type_1_mixed_experiment_metrics.csv
      type_1_echo_experiment_metrics.csv
      type_1_curated_experiment_metrics.csv
      ...
    granovetter_10_percent_runs/
    granovetter_20_percent_runs/
  second_seed/
  third_seed/
  fourth_seed/
  fifth_seed/

Each file contains aggregated model-level metrics for a single simulation run.

-----------------------------------------------------

After simulations complete, run:

    python plot_granovetter_results.py

This script:

- Aggregates all 5 seeds
- Computes means and standard deviations
- Plots curves for each regime (mixed/echo/curated)
- Saves charts to:

plotted_project_results/
  granovetter_5_percent_runs/
    type1_granovetter_5_percent.png
    type2_granovetter_5_percent.png
    ...
  granovetter_10_percent_runs/
  granovetter_20_percent_runs/

Each PNG summarizes all metrics for that belief type + credibility mode.

-----------------------------------------------------

You can manually run a single simulation to test your setup with the command:

    python run_simulation.py

This script will use the default parameters set in configs/granovetter_configs.py and then print the resulting metrics dataframe to the terminal.