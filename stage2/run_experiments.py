"""Stage 2 RL Experiment Orchestrator — "From Heuristics to Intelligence"

Runs the full narrative in 4 chapters:
    Chapter 2: Train DQN, PPO, DDPG, SAC (no weather)
    Chapter 3: Train PPO, SAC with weather context
    Evaluate:  Unified comparison on test days
    Analyze:   Weather ablation, DQN strategy, cross-site transfer
    Visualize: Generate all publication plots
    Report:    Auto-generate STAGE2_RL_SUMMARY.md

Usage::

    python -m stage2.run_experiments --site caltech --timesteps 60000
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "stage2" / "output"
MODEL_DIR = PROJECT_ROOT / "stage2" / "models"


def run_full_pipeline(
    site: str = "caltech",
    total_timesteps: int = 60_000,
    eval_days: int = 20,
    seed: int = 42,
):
    """Execute the full experiment pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    t_start = time.time()

    # ── Chapter 2: Train RL agents (no weather) ─────────────────────────
    print("\n" + "=" * 70)
    print("  CHAPTER 2 — LEARNING TO SCHEDULE (base environment)")
    print("=" * 70)

    from stage2.train_rl import train_agent

    base_algos = ["DQN", "PPO", "DDPG", "SAC"]
    training_results = {}

    for algo in base_algos:
        result = train_agent(
            algo=algo, site=site, use_weather=False,
            total_timesteps=total_timesteps, seed=seed,
        )
        training_results[f"{algo}_base"] = result

    # ── Chapter 3: Train with weather ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  CHAPTER 3 — DOES WEATHER HELP?")
    print("=" * 70)

    weather_algos = ["PPO", "SAC"]
    for algo in weather_algos:
        result = train_agent(
            algo=algo, site=site, use_weather=True,
            total_timesteps=total_timesteps, seed=seed,
        )
        training_results[f"{algo}_weather"] = result

    # ── Evaluate all on test days ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION — ALL POLICIES ON TEST DAYS")
    print("=" * 70)

    from stage2.evaluate_rl import evaluate_all, save_evaluation

    eval_df = evaluate_all(site=site, num_days=eval_days)
    summary = save_evaluation(eval_df, site)

    # ── Analyze ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ANALYSIS")
    print("=" * 70)

    from stage2 import analyze_rl

    print("\n  Action distributions by hour …")
    for algo in base_algos:
        if algo == "DQN":
            continue  # DQN uses discrete meta-actions
        hourly = analyze_rl.action_distribution_by_hour(site=site, algo=algo)
        if not hourly.empty:
            hourly.to_csv(OUTPUT_DIR / f"{site}_{algo}_hourly_actions.csv", index=False)

    print("  DQN strategy timeline …")
    dqn_timeline = analyze_rl.dqn_strategy_timeline(site=site, use_weather=False)
    if not dqn_timeline.empty:
        dqn_timeline.to_csv(OUTPUT_DIR / f"{site}_dqn_strategy_timeline.csv", index=False)

    print("  Weather ablation …")
    ablation = analyze_rl.weather_ablation(site=site, algos=weather_algos, num_days=eval_days)
    if not ablation.empty:
        ablation.to_csv(OUTPUT_DIR / f"{site}_weather_ablation.csv", index=False)

    # ── Visualize ───────────────────────────────────────────────────────
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

    # ── Generate report ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GENERATING SUMMARY REPORT")
    print("=" * 70)
    _generate_report(site, eval_df, training_results, ablation, dqn_timeline)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE — {elapsed / 60:.1f} minutes")
    print(f"  All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


# ── Report generator ─────────────────────────────────────────────────────
def _generate_report(
    site, eval_df, training_results, ablation_df, dqn_timeline_df,
):
    """Auto-generate STAGE2_RL_SUMMARY.md."""
    lines = [
        "# Stage 2: RL Experiments — Summary Report\n",
        "## 1. Introduction\n",
        "This report presents the results of **Stage 2: Reinforcement Learning Experiments**",
        "for EV charging scheduling. We train four RL algorithms (DQN, PPO, DDPG, SAC)",
        "and compare them against heuristic baselines and the MPC Oracle upper bound.\n",
        "**Narrative: \"From Heuristics to Intelligence\"**\n",
        "---\n",
        "## 2. Learning Curves\n",
        f"![Learning Curves](stage2/output/{site}_rl_learning_curves.png)\n",
    ]

    # Training summary table
    lines.append("### Training Summary\n")
    lines.append("| Agent | Episodes | Time (s) | Final Satisfaction |\n")
    lines.append("|-------|----------|----------|-------------------|\n")
    for name, res in training_results.items():
        cfg = res["config"]
        df = res["training_metrics"]
        last = df["satisfaction_ratio"].tail(10).mean() if len(df) > 0 else 0
        lines.append(
            f"| {name} | {cfg['num_episodes']} | {cfg['train_time_seconds']} | {last:.3f} |\n"
        )

    lines += [
        "\n---\n",
        "## 3. Performance Comparison\n",
        f"![Performance](stage2/output/{site}_rl_performance_comparison.png)\n",
    ]

    # Evaluation summary table
    if not eval_df.empty:
        summary = eval_df.groupby("policy").agg({
            "satisfaction_ratio": "mean",
            "peak_demand_kw": "mean",
            "jain_fairness": "mean",
        }).round(3)
        lines.append("\n### Detailed Metrics\n")
        lines.append(summary.to_markdown() + "\n")

    lines += [
        "\n---\n",
        "## 4. RL Power Profiles\n",
        f"![Power Profiles](stage2/output/{site}_rl_power_profiles.png)\n",
        "\n---\n",
        "## 5. Radar Chart\n",
        f"![Radar](stage2/output/{site}_rl_radar.png)\n",
        "\n---\n",
        "## 6. Weather Ablation\n",
    ]

    if not ablation_df.empty:
        lines.append(f"![Weather Ablation](stage2/output/{site}_rl_weather_ablation.png)\n")
        abl_summary = ablation_df.groupby(["algo", "weather_aware"])["satisfaction_ratio"].mean()
        lines.append("\n" + abl_summary.to_markdown() + "\n")
        lines.append("\n**Interpretation:** Comparing weather-aware vs weather-blind agents ")
        lines.append("across different weather conditions reveals whether weather context ")
        lines.append("helps RL agents make better scheduling decisions.\n")
    else:
        lines.append("*Weather ablation was not run.*\n")

    lines += [
        "\n---\n",
        "## 7. DQN Strategy Selection\n",
    ]

    if not dqn_timeline_df.empty:
        lines.append(f"![DQN Timeline](stage2/output/{site}_rl_dqn_timeline.png)\n")
        strat_counts = dqn_timeline_df["strategy"].value_counts()
        lines.append("\n### Strategy Usage Frequency\n")
        lines.append(strat_counts.to_markdown() + "\n")
        lines.append("\n**Interpretation:** The DQN meta-learner reveals which heuristic ")
        lines.append("is most appropriate at different times of day and demand levels.\n")
    else:
        lines.append("*DQN timeline not available.*\n")

    lines += [
        "\n---\n",
        "## 8. Key Findings\n",
        "1. **RL vs Heuristics**: …\n",
        "2. **RL vs MPC Oracle**: …\n",
        "3. **Weather Impact**: …\n",
        "4. **Algorithm Comparison**: …\n",
        "\n*(Fill in findings after reviewing the results above.)*\n",
    ]

    report_path = PROJECT_ROOT / "STAGE2_RL_SUMMARY.md"
    with open(report_path, "w") as f:
        f.writelines(lines)
    print(f"  Report saved → {report_path}")


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 2 RL Experiment Pipeline")
    parser.add_argument("--site", default="caltech", choices=["caltech", "jpl"])
    parser.add_argument("--timesteps", type=int, default=60_000,
                        help="Training timesteps per agent")
    parser.add_argument("--eval-days", type=int, default=20,
                        help="Number of test days to evaluate on")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_full_pipeline(
        site=args.site,
        total_timesteps=args.timesteps,
        eval_days=args.eval_days,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
