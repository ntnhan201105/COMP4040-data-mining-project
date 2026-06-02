"""Evaluation script to compare baseline policies and the MPC Oracle side-by-side."""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from stage2 import event_loader
from stage2.ev_charging_env import EVChargingEnv
from stage2 import baselines

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "stage2" / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_baselines_on_days(
    site: str,
    days: List[Any],
    voltage: float = 208.0,
    max_battery_power: float = 6.6,
    site_capacity_kw: float = 150.0
) -> pd.DataFrame:
    """Run all policies on a fixed set of days and collect comparative results.

    Args:
        site (str): 'caltech' or 'jpl'.
        days (List[datetime]): List of days to evaluate.
        voltage (float): Operating voltage.
        max_battery_power (float): Battery power limit.
        site_capacity_kw (float): Grid connection limit.

    Returns:
        pd.DataFrame: Comparative metrics dataframe.
    """
    env = EVChargingEnv(
        site=site,
        voltage=voltage,
        max_battery_power=max_battery_power,
        site_capacity_kw=site_capacity_kw
    )

    policies = {
        "Uncontrolled": baselines.UncontrolledPolicy(),
        "FCFS": baselines.FCFSPolicy(),
        "EDF": baselines.EDFPolicy(),
        "Round-Robin": baselines.RoundRobinPolicy(),
    }

    records = []

    print(f"\nEvaluating baseline policies on {len(days)} days...")
    for day in tqdm(days):
        end_dt = day + pd.Timedelta(days=1)
        sessions = event_loader.load_sessions_from_json(site, day, end_dt)
        if not sessions:
            continue
        
        # 1. Run simulator-based baselines
        for name, policy in policies.items():
            obs, info = env.reset(options={"start_dt": day, "sessions": sessions})
            done = False
            total_reward = 0.0
            
            while not done:
                action = policy.act(env)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            req = info.get("total_energy_requested_kwh", 0.0)
            deliv = info.get("total_energy_delivered_kwh", 0.0)
            satisfaction = deliv / req if req > 0 else 1.0

            records.append({
                "date": day.strftime("%Y-%m-%d"),
                "policy": name,
                "total_energy_requested_kwh": req,
                "total_energy_delivered_kwh": deliv,
                "satisfaction_ratio": satisfaction,
                "peak_demand_kw": info.get("peak_demand_kw", 0.0),
                "jain_fairness": info.get("jain_fairness", 1.0),
                "num_sessions": info.get("num_completed_sessions", 0) + info.get("num_active_evs", 0)
            })

        # 2. Run MPC Oracle (perfect foresight offline LP)
        oracle = baselines.MPCOraclePolicy()
        # The solver does everything internally
        schedule = oracle.solve(
            sessions=sessions,
            site=site,
            start_dt=day,
            voltage=voltage,
            max_battery_power=max_battery_power,
            peak_penalty_weight=0.5
        )

        # Parse sessions to compute MPC metrics
        queue = event_loader.sessions_to_event_queue(
            sessions=sessions,
            start_dt=day,
            period=5,
            voltage=voltage,
            max_battery_power=max_battery_power,
            force_feasible=True
        )
        evs = []
        while not queue.empty():
            evs.append(queue.get_event().ev)

        # Compute total requested
        mpc_req = sum(ev.requested_energy for ev in evs)

        # Reconstruct delivered energy per EV from schedule
        # Decision variables are matching rates
        station_ids = env.station_ids
        T = 288
        period = 5
        
        # Calculate delivered energy for each EV
        mpc_delivered = 0.0
        ratios = []
        
        for ev in evs:
            # sum rates at ev's station during its window
            arr = max(0, min(ev.arrival, T - 1))
            dep = max(0, min(ev.departure, T))
            ev_amps = schedule[ev.station_id][arr:dep].sum()
            ev_kwh = ev_amps * (voltage / 1000.0) * (period / 60.0)
            # Clip due to precision / request limit
            ev_kwh = min(ev_kwh, ev.requested_energy)
            mpc_delivered += ev_kwh
            
            ratio = ev_kwh / max(1.0, ev.requested_energy)
            ratios.append(ratio)

        # Peak demand
        # sum of rates across stations at each timestep
        time_amps = np.zeros(T)
        for sid in station_ids:
            time_amps += schedule[sid]
        time_kw = time_amps * voltage / 1000.0
        mpc_peak = float(time_kw.max()) if len(time_kw) > 0 else 0.0

        # Fairness
        if ratios:
            sum_ratios = sum(ratios)
            sum_sq_ratios = sum(r**2 for r in ratios)
            mpc_fairness = (sum_ratios ** 2) / (len(ratios) * sum_sq_ratios) if sum_sq_ratios > 0 else 1.0
        else:
            mpc_fairness = 1.0

        mpc_satisfaction = mpc_delivered / mpc_req if mpc_req > 0 else 1.0

        records.append({
            "date": day.strftime("%Y-%m-%d"),
            "policy": "MPC Oracle",
            "total_energy_requested_kwh": mpc_req,
            "total_energy_delivered_kwh": mpc_delivered,
            "satisfaction_ratio": mpc_satisfaction,
            "peak_demand_kw": mpc_peak,
            "jain_fairness": mpc_fairness,
            "num_sessions": len(evs)
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Evaluate EV charging schedulers side-by-side.")
    parser.add_argument("--site", type=str, default="caltech", choices=["caltech", "jpl"], help="ACN-Sim site topology.")
    parser.add_argument("--episodes", type=int, default=15, help="Number of test days to evaluate on.")
    args = parser.parse_args()

    # Load test days
    _, test_days = event_loader.split_train_test(args.site)
    eval_days = test_days[:args.episodes]

    results_df = evaluate_baselines_on_days(site=args.site, days=eval_days)

    # Save details
    results_path = OUTPUT_DIR / f"{args.site}_evaluation_details.csv"
    results_df.to_csv(results_path, index=False)

    # Aggregate metrics
    summary = results_df.groupby("policy").agg({
        "total_energy_requested_kwh": "mean",
        "total_energy_delivered_kwh": "mean",
        "satisfaction_ratio": "mean",
        "peak_demand_kw": "mean",
        "jain_fairness": "mean",
        "num_sessions": "mean"
    }).reindex(["Uncontrolled", "FCFS", "EDF", "Round-Robin", "MPC Oracle"])

    summary_path = OUTPUT_DIR / f"{args.site}_evaluation_summary.csv"
    summary.to_csv(summary_path)

    print(f"\n=== COMPARATIVE EVALUATION SUMMARY FOR {args.site.upper()} ===")
    print(summary.to_markdown())
    print(f"\nDetailed logs saved to: {results_path}")
    print(f"Summary metrics saved to: {summary_path}")


if __name__ == "__main__":
    main()
