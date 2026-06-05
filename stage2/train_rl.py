"""Training infrastructure for RL agents on EV charging scheduling.

Usage (standalone)::

    python -m stage2.train_rl --algo PPO --site caltech --timesteps 60000
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from stage2 import event_loader
from stage2.ev_charging_env import EVChargingEnv
from stage2.weather_env import WeatherEVChargingEnv
from stage2.discrete_env import DiscreteSchedulingEnv
from stage2.rl_agents import create_agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "stage2" / "output"
MODEL_DIR = PROJECT_ROOT / "stage2" / "models"


# ── Callback ─────────────────────────────────────────────────────────────
class EpisodeMetricsCallback(BaseCallback):
    """Records per-episode satisfaction, peak demand, and fairness."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_metrics: List[Dict[str, Any]] = []
        self._ep_reward = 0.0
        self._ep_steps = 0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals.get("rewards", [0.0])[0]
        self._ep_steps += 1

        dones = self.locals.get("dones", [False])
        if dones[0]:
            info = (self.locals.get("infos") or [{}])[0]
            req = info.get("total_energy_requested_kwh", 0.0)
            deliv = info.get("total_energy_delivered_kwh", 0.0)

            self.episode_metrics.append(
                {
                    "episode": len(self.episode_metrics) + 1,
                    "timestep": self.num_timesteps,
                    "reward": self._ep_reward,
                    "steps": self._ep_steps,
                    "satisfaction_ratio": deliv / req if req > 0 else 1.0,
                    "peak_demand_kw": info.get("peak_demand_kw", 0.0),
                    "jain_fairness": info.get("jain_fairness", 1.0),
                    "energy_delivered": deliv,
                    "energy_requested": req,
                }
            )
            self._ep_reward = 0.0
            self._ep_steps = 0

        return True


# ── Environment factory ──────────────────────────────────────────────────
def make_env(
    site: str,
    use_weather: bool,
    train_days: List,
    is_dqn: bool = False,
):
    """Return a callable that creates a configured environment."""

    def _init():
        if use_weather:
            env = WeatherEVChargingEnv(
                site=site, use_weather=True, train_days=train_days
            )
        else:
            env = EVChargingEnv(site=site, train_days=train_days)

        if is_dqn:
            env = DiscreteSchedulingEnv(env)

        return Monitor(env)

    return _init


# ── Training entry point ─────────────────────────────────────────────────
def train_agent(
    algo: str,
    site: str = "caltech",
    use_weather: bool = False,
    total_timesteps: int = 60_000,
    seed: int = 42,
    verbose: int = 1,
) -> Dict[str, Any]:
    """Train an RL agent and persist model + metrics.

    Returns:
        Dict with ``model_path``, ``metrics_path``, ``config``,
        and ``training_metrics`` (DataFrame).
    """
    algo_upper = algo.upper()
    weather_tag = "weather" if use_weather else "base"
    run_name = f"{algo_upper}_{site}_{weather_tag}"

    model_dir = MODEL_DIR / run_name
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {run_name}")
    print(f"  Algorithm : {algo_upper}")
    print(f"  Site      : {site}")
    print(f"  Weather   : {use_weather}")
    print(f"  Timesteps : {total_timesteps:,}")
    print(f"{'=' * 60}")

    # Data split
    train_days, _ = event_loader.split_train_test(site)
    is_dqn = algo_upper == "DQN"

    vec_env = DummyVecEnv([make_env(site, use_weather, train_days, is_dqn)])

    # Build agent
    tb_log = str(OUTPUT_DIR / "tb_logs")
    model = create_agent(
        algo_upper, vec_env, seed=seed, tensorboard_log=tb_log, verbose=verbose
    )

    # Train
    cb = EpisodeMetricsCallback()
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=True)
    elapsed = time.time() - t0

    # Persist
    model_path = model_dir / "model"
    model.save(str(model_path))

    metrics_df = pd.DataFrame(cb.episode_metrics)
    metrics_path = model_dir / "training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    config = {
        "algo": algo_upper,
        "site": site,
        "use_weather": use_weather,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "train_time_seconds": round(elapsed, 1),
        "num_episodes": len(cb.episode_metrics),
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Done in {elapsed:.1f}s  ({len(cb.episode_metrics)} episodes)")
    print(f"  Model → {model_path}")
    if len(metrics_df) > 0:
        last10 = metrics_df["satisfaction_ratio"].tail(10).mean()
        print(f"  Last-10 avg satisfaction: {last10:.3f}")

    vec_env.close()
    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "config": config,
        "training_metrics": metrics_df,
    }


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train an RL agent for EV charging.")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC", "DDPG", "DQN"])
    parser.add_argument("--site", type=str, default="caltech", choices=["caltech", "jpl"])
    parser.add_argument("--weather", action="store_true", help="Include weather in observations")
    parser.add_argument("--timesteps", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_agent(
        algo=args.algo,
        site=args.site,
        use_weather=args.weather,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
