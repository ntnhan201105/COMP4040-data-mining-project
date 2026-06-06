"""
Stage 1 — Mining + Prediction: Main Runner
===========================================
Runs the full pipeline:
  1. Data loading, merging & feature engineering
  2. Exploratory Data Analysis
  3. Clustering (K-Means + DBSCAN)
  4. Association Rule Mining (Apriori)
  5. Prediction (XGBoost + Random Forest)
  6. LSTM Time-Series Forecasting

Usage:
    python stage1_main.py
"""
from __future__ import annotations
import time

from stage1_data_prep import load_and_prepare
from stage1_eda import run_eda
from stage1_clustering import run_clustering
from stage1_association import run_association
from stage1_prediction import run_prediction
from stage1_lstm import run_lstm


def main():
    t0 = time.time()

    print("=" * 60)
    print("STAGE 1 — MINING + PREDICTION PIPELINE")
    print("=" * 60)

    # Part 1: Data
    print("\n" + "=" * 60)
    print("PART 1 — DATA LOADING & FEATURE ENGINEERING")
    print("=" * 60)
    df = load_and_prepare()

    # Part 2: EDA
    run_eda(df)

    # Part 3: Clustering
    run_clustering(df)

    # Part 4: Association Rules
    run_association(df)

    # Part 5: Prediction
    # run_prediction(df)

    # Part 6: LSTM
    # run_lstm(df)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE — {elapsed:.1f}s")
    print("All outputs saved to output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
