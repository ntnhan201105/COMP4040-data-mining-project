"""Stage 1 — EDA Visualizations."""
from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def run_eda(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("PART 2 — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. Distribution of key features
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cols = ["kWhDelivered", "session_duration_min", "charging_rate_kW",
            "connection_hour", "utilization_ratio", "temperature_mean"]
    for ax, col in zip(axes.flat, cols):
        data = df[col].dropna()
        if col in ("kWhDelivered", "session_duration_min", "charging_rate_kW"):
            data = data[data < data.quantile(0.99)]  # trim outliers for viz
        ax.hist(data, bins=50, edgecolor="white", alpha=0.8)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Count")
    plt.suptitle("Feature Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_distributions.png", dpi=150)
    plt.close(fig)

    # 2. Hourly pattern by site
    fig, ax = plt.subplots(figsize=(10, 5))
    for site in df["site"].unique():
        hourly = df[df["site"] == site].groupby("connection_hour").size()
        ax.plot(hourly.index, hourly.values, "o-", label=site.title())
    ax.set(xlabel="Hour of Day", ylabel="Number of Sessions", title="Hourly Connection Pattern by Site")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_hourly_pattern.png", dpi=150)
    plt.close(fig)

    # 3. Day-of-week pattern
    fig, ax = plt.subplots(figsize=(8, 4))
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for site in df["site"].unique():
        dow = df[df["site"] == site].groupby("day_of_week").size()
        ax.bar(dow.index + (0.35 if site == "jpl" else 0), dow.values, width=0.35, label=site.title())
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_labels)
    ax.set(xlabel="Day of Week", ylabel="Sessions", title="Sessions by Day of Week")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_day_of_week.png", dpi=150)
    plt.close(fig)

    # 4. Monthly trend
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    monthly = df.groupby(["month", "site"]).agg(
        count=("kWhDelivered", "size"), avg_kwh=("kWhDelivered", "mean")
    ).reset_index()
    for site in df["site"].unique():
        sub = monthly[monthly["site"] == site]
        axes[0].plot(sub["month"], sub["count"], "o-", label=site.title())
        axes[1].plot(sub["month"], sub["avg_kwh"], "s-", label=site.title())
    axes[0].set(xlabel="Month", ylabel="Sessions", title="Monthly Session Count")
    axes[1].set(xlabel="Month", ylabel="Avg kWh", title="Monthly Avg Energy Delivered")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_monthly_trend.png", dpi=150)
    plt.close(fig)

    # 5. Weather correlation heatmap
    weather_cols = ["kWhDelivered", "session_duration_min", "charging_rate_kW",
                    "temperature_mean", "dew_point_temperature_mean", "wind_speed_mean",
                    "relative_humidity_mean", "visibility_mean", "precipitation_mean"]
    available = [c for c in weather_cols if c in df.columns]
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Correlation: Charging Behavior vs Weather")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_weather_correlation.png", dpi=150)
    plt.close(fig)

    # 6. Energy vs temperature scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df[["temperature_mean", "kWhDelivered"]].dropna()
    ax.scatter(sub["temperature_mean"], sub["kWhDelivered"], s=3, alpha=0.2)
    ax.set(xlabel="Temperature (°C)", ylabel="kWh Delivered", title="Energy Delivered vs Temperature")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_energy_vs_temp.png", dpi=150)
    plt.close(fig)

    # 7. Weekend vs Weekday comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes, ["kWhDelivered", "session_duration_min"],
                               ["Energy (kWh)", "Duration (min)"]):
        sub = df[[col, "is_weekend"]].dropna()
        sub = sub[sub[col] < sub[col].quantile(0.99)]
        sub["label"] = sub["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        sub.boxplot(column=col, by="label", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        plt.suptitle("")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_weekend_vs_weekday.png", dpi=150)
    plt.close(fig)

    print("[eda] All EDA plots saved to output/")
