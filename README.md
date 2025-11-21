# csc-555-project-f25
Final Project Repo for CSC 555: Social Computing &amp; Decentralized AI


Note: Before starting, make sure you can run scripts in Powershell! It will not run without the permission.
For me running PowerShell as admin and using the command;

    > Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

    Say yes to the warning
    > y

Step 1: Change to the project's root directory

Step 2: Setup and activate the mesa virtual enviroment

    Install mesa enviroment
    > python -m venv mesa_env

    Activte mesa enviroment
    > mesa_env\Scripts\activate

Step 3: Install necessary dependencies

    Pip installs
    > python -m pip install mesa==3.3.1 pandas numpy networkx matplotlib solara

    Other installations
    > pip install altair solara ipywidgets

Step 4: Run prospect experiments

    Command to run a single experiment using the options set in prospect_configs (Approximately 1-2 minutes)
    > python run_prospect_experiments.py

    Command to run a set of experiments 5 given seeds with with all 4 different types of beliefs distributions (Approximately 20-40 minutes)
    > python run_all_prospect_experiments.py


How to Adjust run_prospect_experiments.py

Adjust the seed value to a positive integer
- SEED=1234

Adjust the belief distribution. Options: trimodal, asymmetric_shift, asymmetric_extremists, skewed_positive
- BELIEF_DISTRIBUTION = "trimodal"


General Information
- Output files will be found in results/prospect_theory_experiment/seed{#}_{belief distribution}
- Graphs are found in prospect_theory_graphs/seed{#}_{belief distribution}
- Prospect_Theory_Notebook is a Jupyteer notbook used to create the graphical visualizations of data. Import the saved files above, then
    set the seed and belief distribution in the 2nd code cell to generate the graphs.