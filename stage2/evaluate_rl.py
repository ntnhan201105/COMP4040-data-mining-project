"""Unified evaluation: RL agents + baselines + MPC Oracle on the same test days.

Usage::

    python -m stage2.evaluate_rl --site caltech --episodes 20
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import PPO, SAC, DDPG, DQN

from stage2 import event_loader, baselines
from stage2.ev_charging_env import EVChargingEnv
from stage2.weather_env import WeatherEVChargingEnv
from stage2.discrete_env import DiscreteSchedulingEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "stage2" / "output"
MODEL_DIR = PROJECT_ROOT / "stage2" / "models"

ALGO_CLASSES = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG, "DQN": DQN}


# ── Helpers ──────────────────────────────────────────────────────────────
def _metrics_from_info(info: dict, reward: float = 0.0) -> dict:
    req = info.get("total_energy_requested_kwh", 0.0)
    deliv = info.get("total_energy_delivered_kwh", 0.0)
    return {
        "total_energy_requested_kwh": req,
        "total_energy_delivered_kwh": deliv,
        "satisfaction_ratio": deliv / req if req > 0 else 1.0,
        "peak_demand_kw": info.get("peak_demand_kw", 0.0),
        "jain_fairness": info.get("jain_fairness", 1.0),
        "num_sessions": info.get("num_completed_sessions", 0)
                        + info.get("num_active_evs", 0),
        "total_reward": reward,
    }


def run_baseline_episode(env, policy, day, sessions):
    """Run one episode with a heuristic baseline."""
    env.reset(options={"start_dt": day, "sessions": sessions})
    done, reward = False, 0.0
    while not done:
        action = policy.act(env)
        _, r, terminated, truncated, info = env.step(action)
        reward += r
        done = terminated or truncated
    return _metrics_from_info(info, reward)


def run_mpc_episode(env, day, sessions, voltage, max_battery_power):
    """Run the MPC Oracle offline LP for one day."""
    oracle = baselines.MPCOraclePolicy()
    schedule = oracle.solve(
        sessions=sessions, site=env.site, start_dt=day,
        voltage=voltage, max_battery_power=max_battery_power,
    )
    queue = event_loader.sessions_to_event_queue(
        sessions, day, period=5, voltage=voltage,
        max_battery_power=max_battery_power, force_feasible=True,
    )
    evs = []
    while not queue.empty():
        evs.append(queue.get_event().ev)

    T, period = 288, 5
    mpc_deliv, ratios = 0.0, []
    for ev in evs:
        arr = max(0, min(ev.arrival, T - 1))
        dep = max(0, min(ev.departure, T))
        kwh = min(
            schedule[ev.station_id][arr:dep].sum() * (voltage / 1000) * (period / 60),
            ev.requested_energy,
        )
        mpc_deliv += kwh
        ratios.append(kwh / max(1.0, ev.requested_energy))

    time_kw = np.zeros(T)
    for sid in env.station_ids:
        time_kw += schedule[sid]
    time_kw *= voltage / 1000

    mpc_req = sum(ev.requested_energy for ev in evs)
    if ratios:
        s, sq = sum(ratios), sum(r ** 2 for r in ratios)
        fairness = (s ** 2) / (len(ratios) * sq) if sq > 0 else 1.0
    else:
        fairness = 1.0

    return {
        "total_energy_requested_kwh": mpc_req,
        "total_energy_delivered_kwh": mpc_deliv,
        "satisfaction_ratio": mpc_deliv / mpc_req if mpc_req > 0 else 1.0,
        "peak_demand_kw": float(time_kw.max()),
        "jain_fairness": fairness,
        "num_sessions": len(evs),
        "total_reward": 0.0,
    }


def run_rl_episode(model, env, day, sessions, is_dqn=False, use_weather=False):
    """Run one episode with a trained RL agent."""
    if is_dqn:
        wrapped = DiscreteSchedulingEnv(env)
        obs, info = wrapped.reset(options={"start_dt": day, "sessions": sessions})
    else:
        obs, info = env.reset(options={"start_dt": day, "sessions": sessions})

    done, reward = False, 0.0
    strategies = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if is_dqn:
            strategies.append(int(action))
            obs, r, terminated, truncated, info = wrapped.step(int(action))
        else:
            obs, r, terminated, truncated, info = env.step(action)
        reward += r
        done = terminated or truncated

    result = _metrics_from_info(info, reward)
    if strategies:
        result["dqn_strategies"] = strategies
    return result


# ── Main evaluation ──────────────────────────────────────────────────────
def evaluate_all(
    site: str = "caltech",
    num_days: int = 20,
    voltage: float = 208.0,
    max_battery_power: float = 6.6,
    site_capacity_kw: float = 150.0,
) -> pd.DataFrame:
    """Run every available policy on a shared set of test days.

    Returns:
        DataFrame with columns [date, policy, satisfaction_ratio, peak_demand_kw,
        jain_fairness, …].
    """
    _, test_days = event_loader.split_train_test(site)
    eval_days = test_days[:num_days]
    print(f"\nEvaluating on {len(eval_days)} test days for {site.upper()}…")

    # Base environment (for baselines + continuous RL agents)
    env_base = EVChargingEnv(
        site=site, voltage=voltage, max_battery_power=max_battery_power,
        site_capacity_kw=site_capacity_kw,
    )
    # Weather environment (for weather-aware RL agents)
    env_weather = WeatherEVChargingEnv(
        site=site, use_weather=True, voltage=voltage,
        max_battery_power=max_battery_power, site_capacity_kw=site_capacity_kw,
    )

    # Discover trained RL models
    rl_models = {}
    if MODEL_DIR.exists():
        for d in sorted(MODEL_DIR.iterdir()):
            model_file = d / "model.zip"
            config_file = d / "config.json"
            if model_file.exists() and config_file.exists():
                with open(config_file) as f:
                    cfg = json.load(f)
                if cfg.get("site") == site:
                    rl_models[d.name] = cfg

    baseline_policies = {
        "Uncontrolled": baselines.UncontrolledPolicy(),
        "FCFS": baselines.FCFSPolicy(),
        "EDF": baselines.EDFPolicy(),
        "Round-Robin": baselines.RoundRobinPolicy(),
    }

    records: List[Dict[str, Any]] = []

    for day in tqdm(eval_days, desc="Test days"):
        end_dt = day + pd.Timedelta(days=1)
        sessions = event_loader.load_sessions_from_json(site, day, end_dt)
        if not sessions:
            continue
        date_str = day.strftime("%Y-%m-%d")

        # 1. Heuristic baselines
        for name, pol in baseline_policies.items():
            m = run_baseline_episode(env_base, pol, day, sessions)
            records.append({"date": date_str, "policy": name, **m})

        # 2. MPC Oracle
        m = run_mpc_episode(env_base, day, sessions, voltage, max_battery_power)
        records.append({"date": date_str, "policy": "MPC Oracle", **m})

        # 3. RL agents
        for run_name, cfg in rl_models.items():
            algo = cfg["algo"]
            use_weather = cfg.get("use_weather", False)
            is_dqn = algo == "DQN"
            cls = ALGO_CLASSES[algo]
            model = cls.load(str(MODEL_DIR / run_name / "model"))

            label = f"{algo}" + (" +W" if use_weather else "")
            chosen_env = env_weather if use_weather else env_base
            m = run_rl_episode(model, chosen_env, day, sessions, is_dqn=is_dqn, use_weather=use_weather)
            records.append({"date": date_str, "policy": label, **m})

    return pd.DataFrame(records)


def save_evaluation(df: pd.DataFrame, site: str):
    """Persist detail + summary CSVs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    details_path = OUTPUT_DIR / f"{site}_rl_evaluation_details.csv"
    df.to_csv(details_path, index=False)

    summary = (
        df.groupby("policy")
        .agg({
            "satisfaction_ratio": ["mean", "std"],
            "peak_demand_kw": ["mean", "std"],
            "jain_fairness": ["mean", "std"],
            "total_energy_delivered_kwh": "mean",
            "total_energy_requested_kwh": "mean",
            "num_sessions": "mean",
        })
    )
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary_path = OUTPUT_DIR / f"{site}_rl_evaluation_summary.csv"
    summary.to_csv(summary_path)

    print(f"\nSaved: {details_path}")
    print(f"Saved: {summary_path}")
    print("\n" + summary.to_markdown())
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="caltech", choices=["caltech", "jpl"])
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    df = evaluate_all(site=args.site, num_days=args.episodes)
    save_evaluation(df, args.site)


if __name__ == "__main__":
    main()
