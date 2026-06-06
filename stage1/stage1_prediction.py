"""Stage 1 — Prediction: XGBoost + Random Forest (Multi-target)."""
from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ── Task A: Predict kWhDelivered (with duration as feature) ─────────────────
TASK_A_FEATURES = [
    "session_duration_min", "charging_duration_min", "connection_hour",
    "day_of_week", "is_weekend", "month",
    "temperature_mean", "dew_point_temperature_mean",
    "wind_speed_mean", "relative_humidity_mean", "visibility_mean",
]
TASK_A_TARGET = "kWhDelivered"

# ── Task B: Predict session_duration_min (pure temporal + weather) ──────────
TASK_B_FEATURES = [
    "connection_hour", "day_of_week", "is_weekend", "month",
    "temperature_mean", "dew_point_temperature_mean",
    "wind_speed_mean", "relative_humidity_mean", "visibility_mean",
    "precipitation_mean",
]
TASK_B_TARGET = "session_duration_min"


def _prepare(df, features, target):
    sub = df[features + [target]].dropna()
    X, y = sub[features], sub[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _eval(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"    {name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  CV-R²={cv.mean():.3f}±{cv.std():.3f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2,
            "cv_r2_mean": cv.mean(), "model": model, "preds": preds}


def _plot_importance(results, features, fname):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        imp = res["model"].feature_importances_
        idx = np.argsort(imp)
        ax.barh(range(len(idx)), imp[idx])
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([features[i] for i in idx], fontsize=8)
        ax.set_title(f"{res['name']} — Feature Importance")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close(fig)


def _plot_scatter(results, y_test, fname, target_label):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        ax.scatter(y_test, res["preds"], s=5, alpha=0.3)
        lims = [0, min(y_test.max(), np.percentile(y_test, 99))]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set(xlabel=f"Actual {target_label}", ylabel=f"Predicted {target_label}",
               title=f"{res['name']} (R²={res['r2']:.3f})")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close(fig)


def run_prediction(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("PART 5 — PREDICTION (XGBoost + Random Forest)")
    print("=" * 60)

    all_metrics = []

    # ── Task A: kWh prediction ──────────────────────────────────────────
    print("\n  Task A — Predict kWhDelivered")
    X_tr, X_te, y_tr, y_te = _prepare(df, TASK_A_FEATURES, TASK_A_TARGET)
    print(f"    Train={len(X_tr)}, Test={len(X_te)}")
    xgb_a = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
    rf_a = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                  random_state=42, n_jobs=-1)
    res_a = [_eval("XGBoost", xgb_a, X_tr, X_te, y_tr, y_te),
             _eval("RF", rf_a, X_tr, X_te, y_tr, y_te)]
    _plot_importance(res_a, TASK_A_FEATURES, "taskA_feature_importance.png")
    _plot_scatter(res_a, y_te, "taskA_prediction_scatter.png", "kWh")

    for r in res_a:
        all_metrics.append({**{k: v for k, v in r.items() if k not in ("model", "preds")}, "task": "A_kWh"})

    # ── Task B: Duration prediction ─────────────────────────────────────
    print("\n  Task B — Predict session_duration_min")
    X_tr, X_te, y_tr, y_te = _prepare(df, TASK_B_FEATURES, TASK_B_TARGET)
    print(f"    Train={len(X_tr)}, Test={len(X_te)}")
    xgb_b = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
    rf_b = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                  random_state=42, n_jobs=-1)
    res_b = [_eval("XGBoost", xgb_b, X_tr, X_te, y_tr, y_te),
             _eval("RF", rf_b, X_tr, X_te, y_tr, y_te)]
    _plot_importance(res_b, TASK_B_FEATURES, "taskB_feature_importance.png")
    _plot_scatter(res_b, y_te, "taskB_prediction_scatter.png", "Duration (min)")

    for r in res_b:
        all_metrics.append({**{k: v for k, v in r.items() if k not in ("model", "preds")}, "task": "B_duration"})

    pd.DataFrame(all_metrics).to_csv(OUTPUT_DIR / "prediction_metrics.csv", index=False)
    return res_a, res_b
