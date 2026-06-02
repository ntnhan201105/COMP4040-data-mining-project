"""Central orchestrator for Stage 2: RL Environment and Baselines evaluation."""
import argparse
import os
from pathlib import Path
import pandas as pd

from stage2 import event_loader
from stage2 import evaluate
from stage2 import visualize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "stage2" / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_pipeline(site: str, num_eval_days: int):
    """Orchestrates the evaluation and plotting pipeline for the given site.

    Args:
        site (str): 'caltech' or 'jpl'.
        num_eval_days (int): Number of test days to evaluate on.
    """
    print(f"\n==========================================")
    # 1. Split train/test days
    print(f"Loading data split for site {site.upper()}...")
    train_days, test_days = event_loader.split_train_test(site)
    print(f"Total days available: {len(train_days) + len(test_days)}")
    print(f"Train days: {len(train_days)} | Test days: {len(test_days)}")
    
    eval_days = test_days[:num_eval_days]
    print(f"Evaluating policies on the first {len(eval_days)} test days...")

    # 2. Run evaluation
    results_df = evaluate.evaluate_baselines_on_days(
        site=site,
        days=eval_days,
        voltage=208.0,
        max_battery_power=6.6,
        site_capacity_kw=150.0 if site == "caltech" else 150.0  # Custom site limit
    )

    # Save details
    details_path = OUTPUT_DIR / f"{site}_evaluation_details.csv"
    results_df.to_csv(details_path, index=False)

    # 3. Aggregate metrics
    summary = results_df.groupby("policy").agg({
        "total_energy_requested_kwh": "mean",
        "total_energy_delivered_kwh": "mean",
        "satisfaction_ratio": "mean",
        "peak_demand_kw": "mean",
        "jain_fairness": "mean",
        "num_sessions": "mean"
    }).reindex(["Uncontrolled", "FCFS", "EDF", "Round-Robin", "MPC Oracle"])

    summary_path = OUTPUT_DIR / f"{site}_evaluation_summary.csv"
    summary.to_csv(summary_path)

    # 4. Generate plots
    print("\nGenerating evaluation plots...")
    
    # Use the first test day for the power profile plot
    first_test_day = eval_days[0]
    visualize.plot_aggregate_power_profile(
        site=site,
        day=first_test_day,
        voltage=208.0,
        max_battery_power=6.6,
        site_capacity_kw=150.0
    )
    
    # Generate aggregate bar charts
    visualize.plot_summary_charts(site, summary_path, details_path)

    # 5. Print results summary
    print(f"\n==========================================")
    print(f"STAGE 2 PIPELINE COMPLETE FOR SITE: {site.upper()}")
    print(f"==========================================")
    print(summary.to_markdown())
    print(f"\nSaved CSV files and plots in: {OUTPUT_DIR}")
    print(f"==========================================\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 2 EV Scheduling Pipeline.")
    parser.add_argument("--site", type=str, default="caltech", choices=["caltech", "jpl"], help="ACN-Sim site topology.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of test days to evaluate on.")
    args = parser.parse_args()

    run_pipeline(args.site, args.episodes)


if __name__ == "__main__":
    main()
