"""Stage 1 — Data Loading, Merging & Feature Engineering."""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIMATE_PATH = PROJECT_ROOT / "dataset" / "climate" / "average_all.psv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def _load_charging(name: str) -> pd.DataFrame:
    if name == "caltech":
        path = PROJECT_ROOT / "dataset" / "charging" / f"{name}_sessions_full.json"
    elif name == "jpl":
        path = PROJECT_ROOT / "dataset" / "charging" / f"{name}_sessions.json"
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    df = pd.DataFrame(raw["_items"])
    df["site"] = name
    return df


def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    fmt = "%a, %d %b %Y %H:%M:%S GMT"
    df["connection_dt"] = pd.to_datetime(df["connectionTime"], format=fmt, utc=True)
    df["disconnect_dt"] = pd.to_datetime(df["disconnectTime"], format=fmt, utc=True)
    df["done_charging_dt"] = pd.to_datetime(
        df["doneChargingTime"], format=fmt, utc=True, errors="coerce"
    )
    return df


def _extract_user_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the last userInputs entry into columns."""
    records = []
    for _, row in df.iterrows():
        ui = row.get("userInputs")
        if isinstance(ui, list) and len(ui) > 0:
            last = ui[-1]
            records.append(
                {
                    "idx": row.name,
                    "kWhRequested": last.get("kWhRequested"),
                    "milesRequested": last.get("milesRequested"),
                    "minutesAvailable": last.get("minutesAvailable"),
                    "WhPerMile": last.get("WhPerMile"),
                }
            )
        else:
            records.append({"idx": row.name})
    ui_df = pd.DataFrame(records).set_index("idx")
    return df.join(ui_df)


# ── feature engineering ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Duration features (minutes)
    df["session_duration_min"] = (
        (df["disconnect_dt"] - df["connection_dt"]).dt.total_seconds() / 60
    )
    df["charging_duration_min"] = np.where(
        df["done_charging_dt"].notna(),
        (df["done_charging_dt"] - df["connection_dt"]).dt.total_seconds() / 60,
        np.nan,
    )
    df["idle_duration_min"] = np.where(
        df["done_charging_dt"].notna(),
        (df["disconnect_dt"] - df["done_charging_dt"]).dt.total_seconds() / 60,
        np.nan,
    )

    # Charging rate (kW)
    charging_hrs = df["charging_duration_min"] / 60
    df["charging_rate_kW"] = np.where(
        charging_hrs > 0, df["kWhDelivered"] / charging_hrs, np.nan
    )

    # Temporal features
    df["connection_hour"] = df["connection_dt"].dt.hour
    df["day_of_week"] = df["connection_dt"].dt.dayofweek  # 0=Mon
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["connection_dt"].dt.month
    df["week_of_year"] = df["connection_dt"].dt.isocalendar().week.astype(int)
    df["date"] = df["connection_dt"].dt.date

    # Time-of-day bins
    bins = [0, 6, 10, 14, 18, 22, 24]
    labels = ["night", "morning", "midday", "afternoon", "evening", "late_night"]
    df["time_period"] = pd.cut(
        df["connection_hour"], bins=bins, labels=labels, right=False, ordered=False
    )

    # Energy bins
    df["energy_bin"] = pd.cut(
        df["kWhDelivered"],
        bins=[0, 5, 10, 20, 70],
        labels=["low", "medium", "high", "very_high"],
    )

    # Duration bin
    df["duration_bin"] = pd.cut(
        df["session_duration_min"],
        bins=[0, 60, 180, 480, 2000],
        labels=["short", "medium", "long", "very_long"],
    )

    # Utilization ratio (charging vs total session)
    df["utilization_ratio"] = np.where(
        df["session_duration_min"] > 0,
        df["charging_duration_min"] / df["session_duration_min"],
        np.nan,
    )

    return df


# ── merge with climate ──────────────────────────────────────────────────────
def merge_climate(df: pd.DataFrame) -> pd.DataFrame:
    climate = pd.read_csv(CLIMATE_PATH, sep="|")
    for col in ["Year", "Month", "Day", "Hour"]:
        climate[col] = pd.to_numeric(climate[col], errors="coerce").astype("Int64")

    df["Year"] = df["connection_dt"].dt.year
    df["Day"] = df["connection_dt"].dt.day
    df["Hour"] = df["connection_dt"].dt.hour
    # rename month to avoid clash
    df["Month_merge"] = df["connection_dt"].dt.month

    merged = df.merge(
        climate,
        left_on=["Year", "Month_merge", "Day", "Hour"],
        right_on=["Year", "Month", "Day", "Hour"],
        how="left",
        suffixes=("", "_clim"),
    )
    merged.drop(columns=["Month_clim", "Month_merge"], errors="ignore", inplace=True)
    # Drop std columns to keep feature set clean
    std_cols = [c for c in merged.columns if c.endswith("_std")]
    merged.drop(columns=std_cols, errors="ignore", inplace=True)
    return merged


# ── public API ──────────────────────────────────────────────────────────────
def load_and_prepare() -> pd.DataFrame:
    """Full pipeline: load → parse → engineer → merge climate."""
    caltech = _load_charging("caltech")
    jpl = _load_charging("jpl")
    df = pd.concat([caltech, jpl], ignore_index=True)

    df = _parse_times(df)
    df = _extract_user_inputs(df)
    df = engineer_features(df)
    df = merge_climate(df)

    # Remove extreme outliers
    df = df[df["session_duration_min"] > 0]
    df = df[df["session_duration_min"] < 1440]  # < 24h
    df = df[df["kWhDelivered"] > 0]

    print(f"[data_prep] Final dataset: {len(df)} sessions, {df.shape[1]} features")
    return df


if __name__ == "__main__":
    df = load_and_prepare()
    df.to_csv(OUTPUT_DIR / "prepared_data.csv", index=False)
    print(df.head())
    print(df.describe())
