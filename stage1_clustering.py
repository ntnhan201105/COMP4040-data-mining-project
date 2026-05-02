"""Stage 1 — Clustering Analysis (K-Means + DBSCAN)."""
from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


CLUSTER_FEATURES = [
    "session_duration_min",
    "kWhDelivered",
    "charging_rate_kW",
    "connection_hour",
    "is_weekend",
    "utilization_ratio",
]


def prepare_cluster_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    sub = df[CLUSTER_FEATURES].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(sub)
    return sub, X


# ── K-Means ─────────────────────────────────────────────────────────────────
def kmeans_analysis(X: np.ndarray, k_range: range = range(2, 11)):
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels, sample_size=5000))

    # Plot elbow + silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, "o-")
    ax1.set(xlabel="k", ylabel="Inertia", title="Elbow Method")
    ax2.plot(list(k_range), sil_scores, "s-", color="orange")
    ax2.set(xlabel="k", ylabel="Silhouette Score", title="Silhouette Analysis")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "clustering_elbow_silhouette.png", dpi=150)
    plt.close(fig)

    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"[clustering] Best K by silhouette: {best_k} (score={max(sil_scores):.3f})")
    return best_k


def fit_kmeans(X: np.ndarray, k: int):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return labels, km


# ── DBSCAN ──────────────────────────────────────────────────────────────────
def fit_dbscan(X: np.ndarray, eps: float = 0.8, min_samples: int = 15):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels) - {-1})
    noise = (labels == -1).sum()
    print(f"[clustering] DBSCAN: {n_clusters} clusters, {noise} noise points")
    return labels


# ── Profile & Visualize ─────────────────────────────────────────────────────
def profile_clusters(sub: pd.DataFrame, labels: np.ndarray, method: str):
    sub = sub.copy()
    sub["cluster"] = labels
    profile = sub.groupby("cluster")[CLUSTER_FEATURES].agg(["mean", "median", "count"])
    profile.to_csv(OUTPUT_DIR / f"cluster_profile_{method}.csv")
    print(f"\n[clustering] {method} cluster profiles:")
    summary = sub.groupby("cluster")[CLUSTER_FEATURES].mean().round(2)
    print(summary)
    return summary


def plot_clusters(sub: pd.DataFrame, labels: np.ndarray, method: str):
    sub = sub.copy()
    sub["cluster"] = labels
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for cl in sorted(sub["cluster"].unique()):
        mask = sub["cluster"] == cl
        lbl = f"C{cl}" if cl >= 0 else "Noise"
        axes[0].scatter(
            sub.loc[mask, "session_duration_min"],
            sub.loc[mask, "kWhDelivered"],
            s=5, alpha=0.4, label=lbl,
        )
        axes[1].scatter(
            sub.loc[mask, "connection_hour"],
            sub.loc[mask, "charging_rate_kW"],
            s=5, alpha=0.4, label=lbl,
        )
        axes[2].scatter(
            sub.loc[mask, "utilization_ratio"],
            sub.loc[mask, "kWhDelivered"],
            s=5, alpha=0.4, label=lbl,
        )

    axes[0].set(xlabel="Session Duration (min)", ylabel="kWh Delivered", title=f"{method}: Duration vs Energy")
    axes[1].set(xlabel="Connection Hour", ylabel="Charging Rate (kW)", title=f"{method}: Hour vs Rate")
    axes[2].set(xlabel="Utilization Ratio", ylabel="kWh Delivered", title=f"{method}: Utilization vs Energy")
    for ax in axes:
        ax.legend(fontsize=7, markerscale=3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"clusters_{method}.png", dpi=150)
    plt.close(fig)


def run_clustering(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("PART 3 — CLUSTERING ANALYSIS")
    print("=" * 60)
    sub, X = prepare_cluster_data(df)

    # K-Means
    best_k = kmeans_analysis(X)
    km_labels, km_model = fit_kmeans(X, best_k)
    profile_clusters(sub, km_labels, "kmeans")
    plot_clusters(sub, km_labels, "KMeans")

    # DBSCAN
    db_labels = fit_dbscan(X)
    if len(set(db_labels) - {-1}) >= 2:
        profile_clusters(sub, db_labels, "dbscan")
        plot_clusters(sub, db_labels, "DBSCAN")

    return km_labels, best_k
