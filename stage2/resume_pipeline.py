"""Resume the Stage 2 pipeline from where it was interrupted.

Only trains SAC_caltech_weather (the missing model), then runs
evaluate → analyze → visualize → report.
"""
import os
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "stage2" / "output"
MODEL_DIR = PROJECT_ROOT / "stage2" / "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

t_start = time.time()
site = "caltech"
total_timesteps = 60_000
eval_days_count = 20
seed = 42

# ── Step 1: Train only the missing SAC_caltech_weather ──────────────────
sac_weather_config = MODEL_DIR / "SAC_caltech_weather" / "config.json"
training_results = {}

if not sac_weather_config.exists():
    print("\n" + "=" * 70)
    print("  RESUMING: Training SAC_caltech_weather")
    print("=" * 70)
    from stage2.train_rl import train_agent

    result = train_agent(
        algo="SAC", site=site, use_weather=True,
        total_timesteps=total_timesteps, seed=seed,
    )
    training_results["SAC_weather"] = result
else:
    print("  SAC_caltech_weather already trained — skipping.")

# Collect all training results for the report
import json
for d in sorted(MODEL_DIR.iterdir()):
    cfg_path = d / "config.json"
    metrics_path = d / "training_metrics.csv"
    if cfg_path.exists() and metrics_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("site") == site:
            tag = cfg["algo"] + ("_weather" if cfg.get("use_weather") else "_base")
            training_results[tag] = {
                "config": cfg,
                "training_metrics": pd.read_csv(metrics_path),
                "model_path": str(d / "model"),
                "metrics_path": str(metrics_path),
            }

# ── Step 2: Evaluate all on test days ──────────────────────────────────
print("\n" + "=" * 70)
print("  EVALUATION — ALL POLICIES ON TEST DAYS")
print("=" * 70)

from stage2.evaluate_rl import evaluate_all, save_evaluation

eval_df = evaluate_all(site=site, num_days=eval_days_count)
summary = save_evaluation(eval_df, site)

# ── Step 3: Analysis ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ANALYSIS")
print("=" * 70)

from stage2 import analyze_rl

print("\n  Action distributions by hour …")
for algo in ["PPO", "SAC", "DDPG"]:
    hourly = analyze_rl.action_distribution_by_hour(site=site, algo=algo)
    if not hourly.empty:
        hourly.to_csv(OUTPUT_DIR / f"{site}_{algo}_hourly_actions.csv", index=False)

print("  DQN strategy timeline …")
dqn_timeline = analyze_rl.dqn_strategy_timeline(site=site, use_weather=False)
if not dqn_timeline.empty:
    dqn_timeline.to_csv(OUTPUT_DIR / f"{site}_dqn_strategy_timeline.csv", index=False)

print("  Weather ablation …")
ablation = analyze_rl.weather_ablation(site=site, algos=["PPO", "SAC"], num_days=eval_days_count)
if not ablation.empty:
    ablation.to_csv(OUTPUT_DIR / f"{site}_weather_ablation.csv", index=False)

# ── Step 4: Visualization ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  VISUALIZATION")
print("=" * 70)

from stage2 import visualize_rl

print("\n  Learning curves …")
visualize_rl.plot_learning_curves(site)

print("  Performance comparison …")
visualize_rl.plot_performance_comparison(site)

print("  RL power profiles …")
visualize_rl.plot_rl_power_profiles(site)

print("  Weather ablation heatmap …")
visualize_rl.plot_weather_ablation(ablation, site)

print("  DQN strategy timeline …")
visualize_rl.plot_dqn_timeline(dqn_timeline, site)

print("  Radar chart …")
visualize_rl.plot_radar(site)

# ── Step 5: Generate report ───────────────────────────────────────────
print("\n" + "=" * 70)
print("  GENERATING SUMMARY REPORT")
print("=" * 70)

# Import the report generator from run_experiments
from stage2.run_experiments import _generate_report
_generate_report(site, eval_df, training_results, ablation, dqn_timeline)

elapsed = time.time() - t_start
print(f"\n{'=' * 70}")
print(f"  PIPELINE RESUMED & COMPLETE — {elapsed / 60:.1f} minutes")
print(f"  All outputs in: {OUTPUT_DIR}")
print(f"{'=' * 70}\n")
