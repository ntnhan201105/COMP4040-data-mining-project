"""Stage 1 — LSTM Time-Series Forecasting (PyTorch)."""
from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SEQUENCE_LEN = 14  # 2 weeks lookback


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("date").agg(
        session_count=("kWhDelivered", "size"),
        total_kWh=("kWhDelivered", "sum"),
        avg_kWh=("kWhDelivered", "mean"),
        avg_duration=("session_duration_min", "mean"),
        avg_temp=("temperature_mean", "mean"),
        avg_humidity=("relative_humidity_mean", "mean"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def create_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, 0])  # predict session_count
    return np.array(X), np.array(y)


def run_lstm(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("PART 6 — LSTM TIME-SERIES FORECASTING")
    print("=" * 60)

    daily = aggregate_daily(df)
    print(f"[lstm] Daily aggregation: {len(daily)} days")

    if len(daily) < SEQUENCE_LEN + 20:
        print("[lstm] Not enough data for LSTM. Skipping.")
        return

    features = ["session_count", "total_kWh", "avg_duration", "avg_temp", "avg_humidity"]
    data = daily[features].values.astype(np.float32)

    # Handle NaN
    col_means = np.nanmean(data, axis=0)
    for j in range(data.shape[1]):
        mask = np.isnan(data[:, j])
        data[mask, j] = col_means[j]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = create_sequences(scaled, SEQUENCE_LEN)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    model = LSTMModel(input_size=len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    epochs = 100
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).numpy().flatten()
        test_pred = model(X_test_t).numpy().flatten()

    # Inverse scale for session_count (col 0)
    def inv_scale_col0(vals):
        dummy = np.zeros((len(vals), len(features)))
        dummy[:, 0] = vals
        return scaler.inverse_transform(dummy)[:, 0]

    y_train_inv = inv_scale_col0(y_train)
    y_test_inv = inv_scale_col0(y_test)
    train_pred_inv = inv_scale_col0(train_pred)
    test_pred_inv = inv_scale_col0(test_pred)

    train_mae = np.mean(np.abs(y_train_inv - train_pred_inv))
    test_mae = np.mean(np.abs(y_test_inv - test_pred_inv))
    test_rmse = np.sqrt(np.mean((y_test_inv - test_pred_inv) ** 2))
    print(f"\n  LSTM Results (predicting daily session count):")
    print(f"    Train MAE: {train_mae:.2f}")
    print(f"    Test  MAE: {test_mae:.2f}")
    print(f"    Test RMSE: {test_rmse:.2f}")

    # Plots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Loss curve
    axes[0].plot(losses)
    axes[0].set(xlabel="Epoch", ylabel="MSE Loss", title="LSTM Training Loss")

    # Predictions
    dates = daily["date"].values[SEQUENCE_LEN:]
    train_dates = dates[:split]
    test_dates = dates[split:]
    axes[1].plot(train_dates, y_train_inv, "b-", label="Train Actual", alpha=0.7)
    axes[1].plot(train_dates, train_pred_inv, "c--", label="Train Pred", alpha=0.7)
    axes[1].plot(test_dates, y_test_inv, "r-", label="Test Actual", alpha=0.7)
    axes[1].plot(test_dates, test_pred_inv, "m--", label="Test Pred", alpha=0.7)
    axes[1].axvline(test_dates[0], color="gray", ls=":", label="Train/Test Split")
    axes[1].set(xlabel="Date", ylabel="Daily Session Count", title="LSTM Forecast")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "lstm_forecast.png", dpi=150)
    plt.close(fig)
