Control Experiment Simulation

This readme explains how to run the control experiment codebase.
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
- 4 belief initialization modes (type1, type2, type3, type4)
- 3 graph regimes:
      • mixed
      • echo
      • curated

This produces 60 simulations.

Run the experiment (Through Git Bash or some Linux Shell):

    ./run_control.sh

-----------------------------------------------------

After running the script, results will appear under:

results/
  first_seed/
    control_experiment/
      type_1_mixed_control_metrics.csv
      type_1_echo_control_metrics.csv
      ...
  second_seed/
  ...

Each file contains aggregated model-level metrics for a single simulation run.

-----------------------------------------------------

After simulations complete, run:

    python plot_results.py

This script:

- Aggregates all 5 seeds
- Computes means and standard deviations
- Plots curves for each regime (mixed/echo/curated)
- Saves charts to:

plotted_project_results/control_experiment/
  type1_control_experiment.png
  type2_control_experiment.png
  ...

Each PNG summarizes all metrics for that belief type + credibility mode.

-----------------------------------------------------

You can manually run a single simulation to test your setup with the command:

    python run_simulation.py

This script will use the default parameters set in configs/control_configs.py and then print the resulting metrics dataframe to the terminal.
