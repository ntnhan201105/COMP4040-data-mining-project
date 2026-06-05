"""Publication-quality visualizations for RL experiments.

Generates seven plot types that tell the Stage 2 RL story:
1. Learning curves            (reward over training)
2. Performance comparison     (all policies bar chart)
3. RL power profiles          (24h power draw)
4. Weather ablation heatmap
5. DQN strategy timeline
6. Radar chart                (multi-metric comparison)
7. Cross-site transfer matrix
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "stage2" / "output"
MODEL_DIR = PROJECT_ROOT / "stage2" / "models"

# ── Consistent style ────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 14,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.titlesize": 16, "legend.fontsize": 9, "figure.dpi": 200,
})

COLORS = {
    # Baselines (same as existing)
    "Uncontrolled": "#E76F51", "FCFS": "#F4A261", "EDF": "#2A9D8F",
    "Round-Robin": "#457B9D", "MPC Oracle": "#1D3557",
    # RL agents
    "DQN": "#9B59B6", "PPO": "#27AE60", "DDPG": "#E67E22", "SAC": "#3498DB",
    # Weather variants
    "PPO +W": "#1ABC9C", "SAC +W": "#2980B9",
    "DDPG +W": "#D35400", "DQN +W": "#8E44AD",
    # Misc
    "Capacity": "#E63946",
}

def _color(name: str) -> str:
    return COLORS.get(name, "#7f8c8d")


# ── 1. Learning Curves ──────────────────────────────────────────────────
def plot_learning_curves(site: str = "caltech"):
    """Plot episode reward over training for all discovered models."""
    if not MODEL_DIR.exists():
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_map: Dict[str, pd.DataFrame] = {}

    for d in sorted(MODEL_DIR.iterdir()):
        csv = d / "training_metrics.csv"
        if csv.exists() and site in d.name:
            df = pd.read_csv(csv)
            if len(df) > 0:
                algo = d.name.split("_")[0]
                metrics_map[d.name] = df

    if not metrics_map:
        plt.close(fig)
        return

    for label, df in metrics_map.items():
        algo = label.split("_")[0]
        style = "--" if "weather" in label else "-"
        tag = algo + (" +W" if "weather" in label else "")
        window = max(1, len(df) // 20)

        # Smoothed reward
        axes[0].plot(df["episode"], df["reward"].rolling(window, min_periods=1).mean(),
                     style, color=_color(tag), label=tag, linewidth=1.5)
        # Satisfaction
        axes[1].plot(df["episode"], df["satisfaction_ratio"].rolling(window, min_periods=1).mean(),
                     style, color=_color(tag), label=tag, linewidth=1.5)
        # Peak demand
        axes[2].plot(df["episode"], df["peak_demand_kw"].rolling(window, min_periods=1).mean(),
                     style, color=_color(tag), label=tag, linewidth=1.5)

    axes[0].set(xlabel="Episode", ylabel="Episode Reward", title="Learning Curve — Reward")
    axes[1].set(xlabel="Episode", ylabel="Satisfaction Ratio", title="Learning Curve — Satisfaction")
    axes[2].set(xlabel="Episode", ylabel="Peak Demand (kW)", title="Learning Curve — Peak Demand")
    for ax in axes:
        ax.legend(frameon=True, facecolor="white")
        ax.grid(True, ls=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{site}_rl_learning_curves.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {site}_rl_learning_curves.png")


# ── 2. Performance Comparison Bar Chart ─────────────────────────────────
def plot_performance_comparison(site: str = "caltech"):
    """Bar chart comparing all policies on satisfaction, peak demand, fairness."""
    csv = OUTPUT_DIR / f"{site}_rl_evaluation_details.csv"
    if not csv.exists():
        return

    df = pd.read_csv(csv)
    policies = df["policy"].unique().tolist()

    # Desired order: baselines first, then RL
    baseline_order = ["Uncontrolled", "FCFS", "EDF", "Round-Robin", "MPC Oracle"]
    rl_order = [p for p in policies if p not in baseline_order]
    order = [p for p in baseline_order if p in policies] + sorted(rl_order)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics = [
        ("satisfaction_ratio", "Satisfaction Ratio", True),
        ("peak_demand_kw", "Peak Demand (kW)", False),
        ("jain_fairness", "Jain Fairness Index", True),
    ]

    for ax, (col, title, higher_better) in zip(axes, metrics):
        means = [df[df["policy"] == p][col].mean() for p in order]
        stds = [df[df["policy"] == p][col].std() for p in order]
        colors = [_color(p) for p in order]

        bars = ax.bar(range(len(order)), means, yerr=stds, color=colors,
                      edgecolor="none", alpha=0.85, capsize=4,
                      error_kw={"elinewidth": 1, "capthick": 1, "ecolor": "#4A4A4A"})

        # Value labels
        for i, bar in enumerate(bars):
            h = bar.get_height()
            fmt = f"{h:.1%}" if "ratio" in col or "fairness" in col else f"{h:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, h + stds[i] + 0.01,
                    fmt, ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", ls=":", alpha=0.6)

    plt.suptitle(f"Policy Comparison — {site.upper()}", fontsize=16, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{site}_rl_performance_comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {site}_rl_performance_comparison.png")


# ── 3. RL Power Profiles ────────────────────────────────────────────────
def plot_rl_power_profiles(site: str = "caltech"):
    """Plot 24h aggregate power for RL agents vs MPC on a sample test day."""
    from stage2 import event_loader, baselines
    from stage2.ev_charging_env import EVChargingEnv
    from stage2.discrete_env import DiscreteSchedulingEnv
    from stable_baselines3 import PPO, SAC, DDPG, DQN

    algo_cls = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG, "DQN": DQN}

    _, test_days = event_loader.split_train_test(site)
    day = test_days[0]
    sessions = event_loader.load_sessions_from_json(
        site, day, day + pd.Timedelta(days=1)
    )
    if not sessions:
        return

    env = EVChargingEnv(site=site)
    T = 288
    voltage = env.voltage
    time_hours = np.arange(T) * 5 / 60

    profiles: Dict[str, np.ndarray] = {}

    # MPC Oracle
    oracle = baselines.MPCOraclePolicy()
    schedule = oracle.solve(sessions=sessions, site=site, start_dt=day, voltage=voltage,
                            max_battery_power=env.max_battery_power)
    mpc_kw = np.zeros(T)
    for sid in env.station_ids:
        mpc_kw += schedule[sid]
    mpc_kw *= voltage / 1000
    profiles["MPC Oracle"] = mpc_kw

    # RL agents
    if MODEL_DIR.exists():
        for d in sorted(MODEL_DIR.iterdir()):
            model_file = d / "model.zip"
            config_file = d / "config.json"
            if not (model_file.exists() and config_file.exists()):
                continue
            import json
            with open(config_file) as f:
                cfg = json.load(f)
            if cfg.get("site") != site:
                continue

            algo = cfg["algo"]
            is_dqn = algo == "DQN"
            use_weather = cfg.get("use_weather", False)
            model = algo_cls[algo].load(str(d / "model"))
            tag = algo + (" +W" if use_weather else "")

            if is_dqn:
                wrapped = DiscreteSchedulingEnv(env)
                obs, _ = wrapped.reset(options={"start_dt": day, "sessions": sessions})
            elif use_weather:
                from stage2.weather_env import WeatherEVChargingEnv as _WEnv
                w_env = _WEnv(site=site, use_weather=True)
                obs, _ = w_env.reset(options={"start_dt": day, "sessions": sessions})
            else:
                obs, _ = env.reset(options={"start_dt": day, "sessions": sessions})

            power_kw = []
            done = False
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                if is_dqn:
                    obs, _, term, trunc, _ = wrapped.step(int(act))
                elif use_weather:
                    obs, _, term, trunc, _ = w_env.step(act)
                else:
                    obs, _, term, trunc, _ = env.step(act)
                done = term or trunc
                cur_env = w_env if use_weather else env
                it = max(0, cur_env.simulator.iteration - 1)
                rates = cur_env.simulator.charging_rates[:, it]
                power_kw.append(float(np.sum(rates) * voltage / 1000))

            while len(power_kw) < T:
                power_kw.append(0.0)
            profiles[tag] = np.array(power_kw[:T])

    if len(profiles) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, kw in profiles.items():
        lw = 2.0 if name == "MPC Oracle" else 1.5
        ax.plot(time_hours, kw, label=name, color=_color(name), linewidth=lw,
                alpha=0.9)

    ax.axhline(y=150, color=COLORS["Capacity"], ls="--", lw=1.5, label="Site Capacity")
    ax.set(xlabel="Time of Day (h)", ylabel="Power (kW)",
           title=f"RL Power Profiles — {site.upper()} ({day.strftime('%Y-%m-%d')})")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, ls=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{site}_rl_power_profiles.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {site}_rl_power_profiles.png")


# ── 4. Weather Ablation Heatmap ─────────────────────────────────────────
def plot_weather_ablation(ablation_df: pd.DataFrame, site: str = "caltech"):
    """Heatmap: performance delta (weather-aware − weather-blind) by condition."""
    if ablation_df.empty:
        return

    pivot = ablation_df.groupby(["algo", "weather_aware", "condition"])[
        "satisfaction_ratio"
    ].mean().unstack("condition")

    algos = pivot.index.get_level_values("algo").unique()
    conditions = [c for c in ["Cold", "Mild", "Hot", "Windy"] if c in pivot.columns]

    if not conditions:
        return

    # Compute delta: weather_aware=True minus weather_aware=False
    delta_rows = []
    for algo in algos:
        try:
            aware = pivot.loc[(algo, True)]
            blind = pivot.loc[(algo, False)]
            delta = (aware - blind).reindex(conditions).fillna(0)
            delta_rows.append(delta)
        except KeyError:
            continue

    if not delta_rows:
        return

    delta_df = pd.DataFrame(delta_rows, index=algos)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(delta_df.values, cmap="RdYlGn", aspect="auto",
                   vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos)

    for i in range(len(algos)):
        for j in range(len(conditions)):
            v = delta_df.values[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if abs(v) > 0.05 else "black")

    plt.colorbar(im, ax=ax, label="Δ Satisfaction (Weather − No Weather)")
    ax.set_title(f"Weather Ablation — {site.upper()}")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{site}_rl_weather_ablation.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {site}_rl_weather_ablation.png")


# ── 5. DQN Strategy Timeline ────────────────────────────────────────────
def plot_dqn_timeline(timeline_df: pd.DataFrame, site: str = "caltech"):
    """Stacked area chart showing which strategy DQN selects over a day."""
    if timeline_df.empty:
        return

    strategy_colors = {
        "Uncontrolled": COLORS["Uncontrolled"], "FCFS": COLORS["FCFS"],
        "EDF": COLORS["EDF"], "Round-Robin": COLORS["Round-Robin"],
        "Conservative": "#95A5A6",
    }

    days = timeline_df["date"].unique()[:3]  # Plot up to 3 days
    fig, axes = plt.subplots(len(days), 1, figsize=(14, 3.5 * len(days)), sharex=True)
    if len(days) == 1:
        axes = [axes]

    for ax, day in zip(axes, days):
        day_df = timeline_df[timeline_df["date"] == day]
        for strat, color in strategy_colors.items():
            mask = day_df["strategy"] == strat
            ax.scatter(day_df.loc[mask, "hour"], [strat] * mask.sum(),
                       c=color, s=12, alpha=0.8, label=strat)
        ax.set_ylabel(day, fontsize=9)
        ax.set_yticks(list(strategy_colors.keys()))
        ax.grid(True, axis="x", ls=":", alpha=0.5)

    axes[-1].set_xlabel("Time of Day (h)")
    axes[0].set_title(f"DQN Strategy Selection Over Time — {site.upper()}")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=8, label=s) for s, c in strategy_colors.items()]
    axes[0].legend(handles=handles, loc="upper right", ncol=5, fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{site}_rl_dqn_timeline.png", dpi=300)
    plt.close(fig)
    print(f"  Saved: {site}_rl_dqn_timeline.png")


# ── 6. Radar Chart ──────────────────────────────────────────────────────
def plot_radar(site: str = "caltech"):
    """Multi-metric radar comparing policies."""
    csv = OUTPUT_DIR / f"{site}_rl_evaluation_details.csv"
    if not csv.exists():
        return

    df = pd.read_csv(csv)
    policies = df["policy"].unique()

    # Pick a subset for readability (max 7)
    priority = ["MPC Oracle", "PPO", "SAC", "DDPG", "DQN", "Uncontrolled", "Round-Robin"]
    selected = [p for p in priority if p in policies][:7]

    categories = ["Satisfaction", "1/Peak Demand", "Fairness"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    max_peak = df["peak_demand_kw"].max()

    for pol in selected:
        sub = df[df["policy"] == pol]
        vals = [
            sub["satisfaction_ratio"].mean(),
            1.0 - sub["peak_demand_kw"].mean() / max_peak,  # Invert so lower peak = better
            sub["jain_fairness"].mean(),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", label=pol, color=_color(pol), linewidth=1.5,
                markersize=5, alpha=0.85)
        ax.fill(angles, vals, alpha=0.08, color=_color(pol))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Multi-Metric Radar — {site.upper()}", y=1.08, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{site}_rl_radar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {site}_rl_radar.png")


# ── 7. Cross-site Transfer Matrix ───────────────────────────────────────
def plot_transfer_matrix(transfer_df: pd.DataFrame):
    """2×2 heatmap showing performance when train_site ≠ eval_site."""
    if transfer_df.empty:
        return

    for algo in transfer_df["algo"].unique():
        sub = transfer_df[transfer_df["algo"] == algo]
        pivot = sub.groupby(["train_site", "eval_site"])["satisfaction"].mean().unstack()
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1)
        sites = pivot.columns.tolist()
        ax.set_xticks(range(len(sites)))
        ax.set_xticklabels([s.upper() for s in sites])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([s.upper() for s in pivot.index])
        ax.set_xlabel("Eval Site")
        ax.set_ylabel("Train Site")

        for i in range(len(pivot.index)):
            for j in range(len(sites)):
                ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center",
                        fontsize=14, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Mean Satisfaction Ratio")
        ax.set_title(f"Cross-Site Transfer — {algo}")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"rl_transfer_{algo.lower()}.png", dpi=300)
        plt.close(fig)
        print(f"  Saved: rl_transfer_{algo.lower()}.png")
