"""Infrastructure for running episodes and training scheduling algorithms."""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Type
import numpy as np
import pandas as pd
from tqdm import tqdm

from stage2 import event_loader
from stage2.ev_charging_env import EVChargingEnv
from stage2 import baselines

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "stage2" / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_baseline(
    env: EVChargingEnv,
    policy: Any,
    episodes: int = 10,
    days_list: List[Any] = None
) -> List[Dict[str, Any]]:
    """Run a baseline policy on the environment for multiple episodes.

    Args:
        env (EVChargingEnv): The Gymnasium environment.
        policy (Any): An instance of a policy with an `act(env)` method.
        episodes (int): Number of episodes to run (if days_list is not provided).
        days_list (List[datetime]): List of specific days to run sequential episodes on.

    Returns:
        List[Dict[str, Any]]: List of episodic metric summaries.
    """
    results = []
    
    if days_list is not None:
        run_days = days_list
        total_episodes = len(run_days)
    else:
        # Sample random days
        run_days = [None] * episodes
        total_episodes = episodes

    print(f"Running {total_episodes} episodes...")
    for ep in tqdm(range(total_episodes)):
        day = run_days[ep]
        options = {}
        if day is not None:
            # We filter sessions for this specific day to avoid loading a random day
            end_dt = day + pd.Timedelta(days=1)
            sessions = event_loader.load_sessions_from_json(env.site, day, end_dt)
            options = {"start_dt": day, "sessions": sessions}

        obs, info = env.reset(options=options)
        
        # Get start date representation
        ep_date = env.start_dt.strftime("%Y-%m-%d") if env.start_dt else "unknown"
        
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = policy.act(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        # Calculate satisfaction ratio
        req = info.get("total_energy_requested_kwh", 0.0)
        deliv = info.get("total_energy_delivered_kwh", 0.0)
        satisfaction = deliv / req if req > 0 else 1.0

        ep_summary = {
            "episode": ep + 1,
            "date": ep_date,
            "total_reward": total_reward,
            "steps": steps,
            "total_energy_requested_kwh": req,
            "total_energy_delivered_kwh": deliv,
            "energy_satisfaction_ratio": satisfaction,
            "peak_demand_kw": info.get("peak_demand_kw", 0.0),
            "jain_fairness": info.get("jain_fairness", 1.0),
            "num_completed_sessions": info.get("num_completed_sessions", 0)
        }
        results.append(ep_summary)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train/Run baselines for ACN EV charging environment.")
    parser.add_argument("--site", type=str, default="caltech", choices=["caltech", "jpl"], help="ACN-Sim site topology.")
    parser.add_argument("--baseline", type=str, default="uncontrolled", 
                        choices=["uncontrolled", "fcfs", "edf", "round_robin"], 
                        help="Baseline policy to execute.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run (if train set is large).")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"], 
                        help="Which data split to run the policy on.")
    
    args = parser.parse_args()

    # Load days split
    train_days, test_days = event_loader.split_train_test(args.site)
    if args.split == "train":
        days_list = train_days[:args.episodes]
    elif args.split == "test":
        days_list = test_days[:args.episodes]
    else:
        days_list = (train_days + test_days)[:args.episodes]

    print(f"Initializing {args.site.upper()} environment on '{args.split}' split...")
    env = EVChargingEnv(site=args.site)

    # Initialize policy
    if args.baseline == "uncontrolled":
        policy = baselines.UncontrolledPolicy()
    elif args.baseline == "fcfs":
        policy = baselines.FCFSPolicy()
    elif args.baseline == "edf":
        policy = baselines.EDFPolicy()
    elif args.baseline == "round_robin":
        policy = baselines.RoundRobinPolicy()
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    # Run policy
    print(f"Running baseline policy: {args.baseline.upper()}...")
    metrics = run_baseline(env, policy, days_list=days_list)
    
    # Save and display results
    df = pd.DataFrame(metrics)
    output_path = OUTPUT_DIR / f"{args.site}_{args.baseline}_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print("\nSummary Statistics:")
    print(df.mean(numeric_only=True))


if __name__ == "__main__":
    main()
