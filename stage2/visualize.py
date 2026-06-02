"""Visualization module to generate publication-quality plots comparing policies."""
import os
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage2 import event_loader
from stage2.ev_charging_env import EVChargingEnv
from stage2 import baselines

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "stage2" / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Premium colors
COLORS = {
    "Uncontrolled": "#E76F51",  # Warm terracotta/orange
    "FCFS": "#F4A261",          # Sandy orange
    "EDF": "#2A9D8F",           # Teal
    "Round-Robin": "#457B9D",   # Soft blue
    "MPC Oracle": "#1D3557",    # Deep navy blue
    "Capacity": "#E63946"       # Red dash for capacity
}

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'legend.fontsize': 10,
    'figure.dpi': 200
})


def plot_aggregate_power_profile(
    site: str,
    day: Any,
    voltage: float = 208.0,
    max_battery_power: float = 6.6,
    site_capacity_kw: float = 150.0
):
    """Run all policies for a single day and plot their aggregate power profiles over 24 hours."""
    end_dt = day + pd.Timedelta(days=1)
    sessions = event_loader.load_sessions_from_json(site, day, end_dt)
    if not sessions:
        print(f"No sessions to plot for day {day.date()}")
        return

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
        "Round-Robin": baselines.RoundRobinPolicy()
    }

    profiles = {}
    T = 288
    time_hours = np.arange(T) * 5 / 60.0

    # 1. Run baselines
    for name, policy in policies.items():
        env.reset(options={"start_dt": day, "sessions": sessions})
        done = False
        rates_history = []
        
        while not done:
            action = policy.act(env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record current active rates
            curr_iter = max(0, env.simulator.iteration - 1)
            actual_rates = env.simulator.charging_rates[:, curr_iter]
            rates_history.append(np.sum(actual_rates))
            
        # If simulation finished early or step count != 288, pad with zeros
        while len(rates_history) < T:
            rates_history.append(0.0)
            
        profiles[name] = np.array(rates_history) * voltage / 1000.0

    # 2. Run MPC Oracle
    oracle = baselines.MPCOraclePolicy()
    schedule = oracle.solve(
        sessions=sessions,
        site=site,
        start_dt=day,
        voltage=voltage,
        max_battery_power=max_battery_power
    )
    mpc_rates = np.zeros(T)
    for sid in env.station_ids:
        mpc_rates += schedule[sid]
    profiles["MPC Oracle"] = mpc_rates * voltage / 1000.0

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, kw_profile in profiles.items():
        ax.plot(
            time_hours,
            kw_profile,
            label=name,
            color=COLORS[name],
            linewidth=1.8 if name == "MPC Oracle" else 1.5,
            alpha=0.9
        )

    # Capacity line
    ax.axhline(
        y=site_capacity_kw,
        color=COLORS["Capacity"],
        linestyle="--",
        linewidth=1.5,
        label=f"Site Capacity ({site_capacity_kw} kW)"
    )

    ax.set_title(f"Aggregate Charging Power Profile - {site.upper()} ({day.strftime('%Y-%m-%d')})")
    ax.set_xlabel("Time of Day (Hours)")
    ax.set_ylabel("Charging Power (kW)")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{site}_aggregate_power_profiles.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated power profile plot: {output_path}")


def plot_summary_charts(site: str, summary_csv: Path, details_csv: Path):
    """Plot aggregate satisfaction ratios and peak demand comparisons from evaluation CSV files."""
    if not summary_csv.exists() or not details_csv.exists():
        print("CSV files missing. Cannot generate summary charts.")
        return

    # Load data
    summary_df = pd.read_csv(summary_csv)
    # Check if policy is index or column
    if "policy" not in summary_df.columns:
        summary_df = summary_df.rename(columns={summary_df.columns[0]: "policy"})
    
    details_df = pd.read_csv(details_csv)

    # Reorder
    order = ["Uncontrolled", "FCFS", "EDF", "Round-Robin", "MPC Oracle"]
    summary_df = summary_df.set_index("policy").reindex(order).reset_index()

    # 1. Satisfaction Ratio Plot (with error bars showing std dev across days)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Calculate error bars (std dev of satisfaction ratio per policy)
    errors = []
    means = []
    for policy in order:
        policy_data = details_df[details_df["policy"] == policy]["satisfaction_ratio"]
        means.append(policy_data.mean())
        errors.append(policy_data.std() if len(policy_data) > 1 else 0.0)

    bars = ax.bar(
        order,
        means,
        yerr=errors,
        color=[COLORS[p] for p in order],
        edgecolor="none",
        alpha=0.85,
        capsize=5,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#4A4A4A"}
    )
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height*100:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height - 0.08 if height > 0.15 else height + 0.01),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9,
            color="white" if height > 0.15 else "black",
            weight="bold"
        )

    ax.set_title(f"Average Energy Satisfaction Ratio ({site.upper()})")
    ax.set_ylabel("Satisfaction Ratio (Delivered / Requested)")
    ax.set_ylim(0, 1.05)
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    satisfaction_path = OUTPUT_DIR / f"{site}_energy_satisfaction.png"
    plt.savefig(satisfaction_path, dpi=300)
    plt.close()
    print(f"Generated energy satisfaction plot: {satisfaction_path}")

    # 2. Peak Demand Plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    peak_means = []
    peak_errors = []
    for policy in order:
        policy_data = details_df[details_df["policy"] == policy]["peak_demand_kw"]
        peak_means.append(policy_data.mean())
        peak_errors.append(policy_data.std() if len(policy_data) > 1 else 0.0)

    bars = ax.bar(
        order,
        peak_means,
        yerr=peak_errors,
        color=[COLORS[p] for p in order],
        edgecolor="none",
        alpha=0.85,
        capsize=5,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#4A4A4A"}
    )

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f} kW",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9,
            weight="bold"
        )

    ax.set_title(f"Average Peak Power Demand ({site.upper()})")
    ax.set_ylabel("Peak Demand (kW)")
    ax.set_ylim(0, max(peak_means) * 1.2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    peak_path = OUTPUT_DIR / f"{site}_peak_demand_comparison.png"
    plt.savefig(peak_path, dpi=300)
    plt.close()
    print(f"Generated peak demand plot: {peak_path}")


def main():
    # Simple CLI fallback
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, default="caltech")
    args = parser.parse_args()

    # Find the first test day to plot single day profile
    _, test_days = event_loader.split_train_test(args.site)
    day = test_days[0]
    
    # Plot single day profile
    plot_aggregate_power_profile(args.site, day)

    # Check for CSVs to plot summaries
    summary_csv = OUTPUT_DIR / f"{args.site}_evaluation_summary.csv"
    details_csv = OUTPUT_DIR / f"{args.site}_evaluation_details.csv"
    plot_summary_charts(args.site, summary_csv, details_csv)


if __name__ == "__main__":
    main()
