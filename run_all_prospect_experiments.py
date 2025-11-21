"""
Run all Prospect Theory experiments across multiple seeds and belief distributions
"""
import os
import subprocess
import sys
from pathlib import Path
import re
# Configuration
SEEDS = [1, 42, 123, 999, 7777]
BELIEF_DISTRIBUTIONS = ["trimodal", "asymmetric_shift", "asymmetric_extremists", "skewed_positive"]

def update_config_file(belief_distribution: str, seed: int):
    """Update the prospect_configs.py file with new values - automatically detects current format"""
    config_file = Path("configs/prospect_configs.py")
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Use regex to find and replace any BELIEF_DISTRIBUTION value (handles any current value)
    content = re.sub(
        r'BELIEF_DISTRIBUTION\s*=\s*"[^"]*"',
        f'BELIEF_DISTRIBUTION = "{belief_distribution}"',
        content
    )
    
    # Use regex to find and replace any SEED value (handles any current seed number)
    content = re.sub(
        r'SEED\s*=\s*\d+',
        f'SEED={seed}',
        content
    )
    
    with open(config_file, 'w') as f:
        f.write(content)

def run_experiments_silently(belief_distribution: str, seed: int):
    """Run experiments without console output"""
    update_config_file(belief_distribution, seed)
    
    try:
        result = subprocess.run(
            [sys.executable, "run_prospect_experiments.py"],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, 'SEED': str(seed)}
        )
        print(f"Completed: {belief_distribution} beliefs, seed {seed}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed: {belief_distribution} beliefs, seed {seed}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Run all combinations by seed first"""
    print("Running all Prospect Theory experiments...")
    print(f"Seeds: {SEEDS}")
    print(f"Belief distributions: {BELIEF_DISTRIBUTIONS}")
    print("Running all distributions for each seed before moving to next seed")
    print("=" * 50)
    
    successful_runs = 0
    total_runs = len(SEEDS) * len(BELIEF_DISTRIBUTIONS)
    
    for seed in SEEDS:
        print(f"\n--- Processing Seed {seed} ---")
        seed_success = 0
        
        for belief_dist in BELIEF_DISTRIBUTIONS:
            if run_experiments_silently(belief_dist, seed):
                successful_runs += 1
                seed_success += 1
        
        print(f"Seed {seed}: {seed_success}/{len(BELIEF_DISTRIBUTIONS)} distributions completed")
    
    # Restore original config
    update_config_file("trimodal", 1)
    
    print("\n" + "=" * 50)
    print(f"Overall: {successful_runs}/{total_runs} runs completed")

if __name__ == "__main__":
    main()