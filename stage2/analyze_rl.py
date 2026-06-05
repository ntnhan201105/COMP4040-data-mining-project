"""Policy analysis connecting RL results back to Stage 1 findings.

Produces:
- Action distribution by hour-of-day (do RL agents match Stage 1 temporal patterns?)
- Weather ablation (hot vs cold vs windy days)
- DQN strategy selection timeline
- Cross-site generalization matrix
"""
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import PPO, SAC, DDPG, DQN

from stage2 import event_loader, baselines
from stage2.ev_charging_env import EVChargingEnv
from stage2.weather_env import WeatherEVChargingEnv, WEATHER_FEATURES
from stage2.discrete_env import DiscreteSchedulingEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "stage2" / "output"
MODEL_DIR = PROJECT_ROOT / "stage2" / "models"

ALGO_CLASSES = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG, "DQN": DQN}


# ── 1. Action Distribution by Hour ──────────────────────────────────────
def action_distribution_by_hour(
    site: str = "caltech",
    algo: str = "PPO",
    use_weather: bool = False,
    num_days: int = 10,
) -> pd.DataFrame:
    """Record mean charging intensity by hour-of-day for an RL agent.

    Returns a DataFrame with columns [hour, mean_action, std_action,
    num_occupied_stations].
    """
    weather_tag = "weather" if use_weather else "base"
    run_name = f"{algo.upper()}_{site}_{weather_tag}"
    model_path = MODEL_DIR / run_name / "model.zip"
    if not model_path.exists():
        print(f"  [skip] Model not found: {model_path}")
        return pd.DataFrame()

    model = ALGO_CLASSES[algo.upper()].load(str(MODEL_DIR / run_name / "model"))
    is_dqn = algo.upper() == "DQN"

    if use_weather:
        env = WeatherEVChargingEnv(site=site, use_weather=True)
    else:
        env = EVChargingEnv(site=site)

    _, test_days = event_loader.split_train_test(site)
    eval_days = test_days[:num_days]
    records = []

    for day in eval_days:
        sessions = event_loader.load_sessions_from_json(
            site, day, day + pd.Timedelta(days=1)
        )
        if not sessions:
            continue

        if is_dqn:
            wrapped = DiscreteSchedulingEnv(env)
            obs, _ = wrapped.reset(options={"start_dt": day, "sessions": sessions})
        else:
            obs, _ = env.reset(options={"start_dt": day, "sessions": sessions})

        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            hour = (day + timedelta(minutes=step * 5)).hour + (step * 5 % 60) / 60

            if is_dqn:
                records.append({"hour": int(hour), "strategy": int(action)})
                obs, _, terminated, truncated, _ = wrapped.step(int(action))
            else:
                mean_act = float(np.mean(action[action > 0.01])) if (action > 0.01).any() else 0.0
                occupied = int((action > 0.01).sum())
                records.append({
                    "hour": int(hour), "mean_action": mean_act,
                    "num_active": occupied,
                })
                obs, _, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            step += 1

    return pd.DataFrame(records)


# ── 2. Weather Ablation ─────────────────────────────────────────────────
def weather_ablation(
    site: str = "caltech",
    algos: Optional[List[str]] = None,
    num_days: int = 20,
) -> pd.DataFrame:
    """Compare weather-aware vs weather-blind agents across weather conditions.

    Classifies test days as HOT / COLD / WINDY based on daily avg temperature
    and wind speed, then compares performance.
    """
    if algos is None:
        algos = ["PPO", "SAC"]

    _, test_days = event_loader.split_train_test(site)
    eval_days = test_days[:num_days]

    # Load climate data for day classification
    climate = pd.read_csv(
        PROJECT_ROOT / "dataset" / "climate" / "average_all.psv", sep="|"
    )
    for c in ["Year", "Month", "Day"]:
        climate[c] = pd.to_numeric(climate[c], errors="coerce")

    records = []

    for algo in algos:
        for use_weather in [False, True]:
            weather_tag = "weather" if use_weather else "base"
            run_name = f"{algo.upper()}_{site}_{weather_tag}"
            model_file = MODEL_DIR / run_name / "model.zip"
            if not model_file.exists():
                continue

            model = ALGO_CLASSES[algo.upper()].load(str(MODEL_DIR / run_name / "model"))
            is_dqn = algo.upper() == "DQN"

            env = (
                WeatherEVChargingEnv(site=site, use_weather=use_weather)
                if use_weather
                else EVChargingEnv(site=site)
            )

            for day in eval_days:
                sessions = event_loader.load_sessions_from_json(
                    site, day, day + pd.Timedelta(days=1)
                )
                if not sessions:
                    continue

                # Classify day weather
                mask = (
                    (climate["Year"] == day.year)
                    & (climate["Month"] == day.month)
                    & (climate["Day"] == day.day)
                )
                day_weather = climate[mask]
                avg_temp = day_weather["temperature_mean"].mean() if len(day_weather) > 0 else 20.0
                avg_wind = day_weather["wind_speed_mean"].mean() if len(day_weather) > 0 else 3.0

                if avg_temp > 28:
                    condition = "Hot"
                elif avg_temp < 15:
                    condition = "Cold"
                elif avg_wind > 5:
                    condition = "Windy"
                else:
                    condition = "Mild"

                # Run episode
                if is_dqn:
                    wrapped = DiscreteSchedulingEnv(env)
                    obs, _ = wrapped.reset(options={"start_dt": day, "sessions": sessions})
                else:
                    obs, _ = env.reset(options={"start_dt": day, "sessions": sessions})

                done, total_r = False, 0.0
                while not done:
                    act, _ = model.predict(obs, deterministic=True)
                    if is_dqn:
                        obs, r, term, trunc, info = wrapped.step(int(act))
                    else:
                        obs, r, term, trunc, info = env.step(act)
                    total_r += r
                    done = term or trunc

                req = info.get("total_energy_requested_kwh", 0)
                deliv = info.get("total_energy_delivered_kwh", 0)

                records.append({
                    "algo": algo, "weather_aware": use_weather,
                    "condition": condition,
                    "date": day.strftime("%Y-%m-%d"),
                    "satisfaction_ratio": deliv / req if req > 0 else 1.0,
                    "peak_demand_kw": info.get("peak_demand_kw", 0),
                    "jain_fairness": info.get("jain_fairness", 1.0),
                    "avg_temp": avg_temp, "avg_wind": avg_wind,
                })

    return pd.DataFrame(records)


# ── 3. DQN Strategy Timeline ────────────────────────────────────────────
def dqn_strategy_timeline(
    site: str = "caltech",
    use_weather: bool = False,
    num_days: int = 5,
) -> pd.DataFrame:
    """Record which meta-strategy DQN selects at each 5-min step."""
    weather_tag = "weather" if use_weather else "base"
    run_name = f"DQN_{site}_{weather_tag}"
    model_file = MODEL_DIR / run_name / "model.zip"
    if not model_file.exists():
        return pd.DataFrame()

    model = DQN.load(str(MODEL_DIR / run_name / "model"))
    env = (
        WeatherEVChargingEnv(site=site, use_weather=use_weather)
        if use_weather
        else EVChargingEnv(site=site)
    )
    wrapped = DiscreteSchedulingEnv(env)

    _, test_days = event_loader.split_train_test(site)
    records = []

    for day in test_days[:num_days]:
        sessions = event_loader.load_sessions_from_json(
            site, day, day + pd.Timedelta(days=1)
        )
        if not sessions:
            continue
        obs, _ = wrapped.reset(options={"start_dt": day, "sessions": sessions})
        done, step = False, 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = wrapped.step(int(act))
            records.append({
                "date": day.strftime("%Y-%m-%d"),
                "step": step,
                "hour": step * 5 / 60,
                "strategy": DiscreteSchedulingEnv.STRATEGY_NAMES[int(act)],
                "num_active": info.get("num_active_evs", 0),
            })
            done = term or trunc
            step += 1

    return pd.DataFrame(records)


# ── 4. Cross-site Transfer ──────────────────────────────────────────────
def cross_site_transfer(
    algos: Optional[List[str]] = None,
    num_days: int = 10,
) -> pd.DataFrame:
    """Evaluate models trained on one site against the other site."""
    if algos is None:
        algos = ["PPO", "SAC"]

    records = []
    for algo in algos:
        for train_site in ["caltech", "jpl"]:
            run_name = f"{algo.upper()}_{train_site}_base"
            model_file = MODEL_DIR / run_name / "model.zip"
            if not model_file.exists():
                continue
            model = ALGO_CLASSES[algo.upper()].load(str(MODEL_DIR / run_name / "model"))

            for eval_site in ["caltech", "jpl"]:
                env = EVChargingEnv(site=eval_site)
                _, test_days = event_loader.split_train_test(eval_site)

                for day in test_days[:num_days]:
                    sessions = event_loader.load_sessions_from_json(
                        eval_site, day, day + pd.Timedelta(days=1)
                    )
                    if not sessions:
                        continue
                    obs, _ = env.reset(options={"start_dt": day, "sessions": sessions})
                    done, total_r = False, 0.0
                    while not done:
                        act, _ = model.predict(obs, deterministic=True)
                        obs, r, term, trunc, info = env.step(act)
                        total_r += r
                        done = term or trunc

                    req = info.get("total_energy_requested_kwh", 0)
                    deliv = info.get("total_energy_delivered_kwh", 0)
                    records.append({
                        "algo": algo, "train_site": train_site,
                        "eval_site": eval_site,
                        "satisfaction": deliv / req if req > 0 else 1.0,
                        "peak_demand_kw": info.get("peak_demand_kw", 0),
                        "jain_fairness": info.get("jain_fairness", 1.0),
                    })

    return pd.DataFrame(records)
