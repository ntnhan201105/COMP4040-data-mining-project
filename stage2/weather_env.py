"""Weather-augmented Gymnasium environment for EV charging scheduling.

Extends EVChargingEnv by appending normalized weather features (temperature,
humidity, wind speed, visibility, precipitation) to each station's observation
vector as global context.  This enables RL agents to condition scheduling
decisions on current weather conditions.
"""
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from gymnasium import spaces

from stage2.ev_charging_env import EVChargingEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIMATE_PATH = PROJECT_ROOT / "dataset" / "climate" / "average_all.psv"

# Weather features appended to observations
WEATHER_FEATURES = [
    "temperature_mean",
    "relative_humidity_mean",
    "wind_speed_mean",
    "visibility_mean",
    "precipitation_mean",
]


class WeatherEVChargingEnv(EVChargingEnv):
    """EVChargingEnv with weather context appended to observations.

    When ``use_weather=True``, the observation shape changes from
    ``(num_stations, 8)`` to ``(num_stations, 13)`` — the extra 5 columns
    are z-score-normalized weather features broadcast to every station row.
    """

    def __init__(self, use_weather: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_weather = use_weather
        self._weather_lookup: Dict[tuple, np.ndarray] = {}
        self._weather_stats: Dict[str, tuple] = {}
        self._weather_df: Optional[pd.DataFrame] = None

        if self.use_weather:
            self._load_climate_data()
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_stations, 8 + len(WEATHER_FEATURES)),
                dtype=np.float32,
            )

    # ── Climate data loading ────────────────────────────────────────────
    def _load_climate_data(self):
        """Load climate PSV once and build a fast lookup dict."""
        df = pd.read_csv(CLIMATE_PATH, sep="|")
        for col in ["Year", "Month", "Day", "Hour"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill missing weather columns with 0
        for col in WEATHER_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        # Normalization statistics (z-score)
        self._weather_stats = {
            col: (float(df[col].mean()), float(df[col].std()) + 1e-8)
            for col in WEATHER_FEATURES
        }

        # Build vectorized lookup: (year, month, day, hour) → normalized weather
        keys = list(
            zip(
                df["Year"].astype(int),
                df["Month"].astype(int),
                df["Day"].astype(int),
                df["Hour"].astype(int),
            )
        )
        values = df[WEATHER_FEATURES].values.copy().astype(np.float32)
        for i, col in enumerate(WEATHER_FEATURES):
            mean, std = self._weather_stats[col]
            mask = ~np.isnan(values[:, i])
            values[mask, i] = (values[mask, i] - mean) / std
            values[~mask, i] = 0.0

        self._weather_lookup = {k: values[j] for j, k in enumerate(keys)}
        self._weather_df = df

    # ── Observation augmentation ────────────────────────────────────────
    def _get_weather_vector(self) -> np.ndarray:
        """Return normalized weather vector for the current timestep."""
        if not self.use_weather or self.start_dt is None:
            return np.zeros(len(WEATHER_FEATURES), dtype=np.float32)

        current_dt = self.start_dt + timedelta(
            minutes=self.simulator.iteration * self.period
        )
        key = (current_dt.year, current_dt.month, current_dt.day, current_dt.hour)
        return self._weather_lookup.get(
            key, np.zeros(len(WEATHER_FEATURES), dtype=np.float32)
        )

    def _get_obs(self) -> np.ndarray:
        """Build observation matrix, optionally with weather columns."""
        base_obs = super()._get_obs()  # (num_stations, 8)
        if not self.use_weather:
            return base_obs

        weather = self._get_weather_vector()  # (5,)
        weather_bc = np.tile(weather, (self.num_stations, 1))  # (num_stations, 5)
        return np.concatenate([base_obs, weather_bc], axis=1).astype(np.float32)

    # ── Utility for analysis ────────────────────────────────────────────
    def get_raw_weather(self) -> Dict[str, float]:
        """Return un-normalized weather for the current timestep (for analysis)."""
        if self.start_dt is None or self._weather_df is None:
            return {c: 0.0 for c in WEATHER_FEATURES}
        current_dt = self.start_dt + timedelta(
            minutes=self.simulator.iteration * self.period
        )
        key = (current_dt.year, current_dt.month, current_dt.day, current_dt.hour)
        mask = (
            (self._weather_df["Year"] == key[0])
            & (self._weather_df["Month"] == key[1])
            & (self._weather_df["Day"] == key[2])
            & (self._weather_df["Hour"] == key[3])
        )
        rows = self._weather_df[mask]
        if len(rows) == 0:
            return {c: 0.0 for c in WEATHER_FEATURES}
        row = rows.iloc[0]
        return {
            c: float(row[c]) if not pd.isna(row[c]) else 0.0 for c in WEATHER_FEATURES
        }
