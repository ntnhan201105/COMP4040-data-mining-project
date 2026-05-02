"""Stage 1 — Association Rule Mining (Apriori)."""
from __future__ import annotations
import warnings
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def discretize_for_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Create a transaction-style boolean DataFrame."""
    txn = pd.DataFrame(index=df.index)

    # Time period
    for val in df["time_period"].dropna().unique():
        txn[f"time={val}"] = (df["time_period"] == val).astype(bool)

    # Weekend vs weekday
    txn["is_weekend=True"] = (df["is_weekend"] == 1).astype(bool)
    txn["is_weekend=False"] = (df["is_weekend"] == 0).astype(bool)

    # Energy bin
    for val in df["energy_bin"].dropna().unique():
        txn[f"energy={val}"] = (df["energy_bin"] == val).astype(bool)

    # Duration bin
    for val in df["duration_bin"].dropna().unique():
        txn[f"duration={val}"] = (df["duration_bin"] == val).astype(bool)

    # Site
    for val in df["site"].unique():
        txn[f"site={val}"] = (df["site"] == val).astype(bool)

    # Temperature bin (if available)
    if "temperature_mean" in df.columns:
        temp = df["temperature_mean"].dropna()
        if len(temp) > 0:
            tq = temp.quantile([0.33, 0.66])
            txn["temp=cold"] = (df["temperature_mean"] <= tq.iloc[0]).astype(bool)
            txn["temp=mild"] = (
                (df["temperature_mean"] > tq.iloc[0])
                & (df["temperature_mean"] <= tq.iloc[1])
            ).astype(bool)
            txn["temp=hot"] = (df["temperature_mean"] > tq.iloc[1]).astype(bool)

    # Wind
    if "wind_speed_mean" in df.columns:
        wmed = df["wind_speed_mean"].median()
        txn["wind=low"] = (df["wind_speed_mean"] <= wmed).astype(bool)
        txn["wind=high"] = (df["wind_speed_mean"] > wmed).astype(bool)

    # Utilization
    if "utilization_ratio" in df.columns:
        txn["util=high"] = (df["utilization_ratio"] >= 0.7).fillna(False).astype(bool)
        txn["util=low"] = (df["utilization_ratio"] < 0.7).fillna(False).astype(bool)

    return txn.fillna(False)


def run_association(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("PART 4 — ASSOCIATION RULE MINING")
    print("=" * 60)

    txn = discretize_for_rules(df)
    print(f"[association] Transaction matrix: {txn.shape}")

    freq = apriori(txn, min_support=0.05, use_colnames=True)
    print(f"[association] Frequent itemsets found: {len(freq)}")

    if len(freq) == 0:
        print("[association] No frequent itemsets found. Try lower min_support.")
        return

    rules = association_rules(freq, metric="lift", min_threshold=1.2)
    rules = rules.sort_values("lift", ascending=False)
    print(f"[association] Rules generated: {len(rules)}")

    # Save full rules
    rules.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)

    # Print top 20
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    top = rules.head(20)[cols].copy()
    top["antecedents"] = top["antecedents"].apply(lambda x: ", ".join(x))
    top["consequents"] = top["consequents"].apply(lambda x: ", ".join(x))
    print("\nTop 20 Association Rules (by lift):")
    print(top.to_string(index=False))

    return rules
