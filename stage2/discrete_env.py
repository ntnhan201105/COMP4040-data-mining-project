"""Discrete action wrapper for DQN compatibility.

Wraps EVChargingEnv (or WeatherEVChargingEnv) with a meta-action space where
the agent selects one of 5 built-in scheduling strategies each timestep.
This creates a "strategy selector" that learns *when* to apply each heuristic.
"""
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stage2.ev_charging_env import EVChargingEnv
from stage2 import baselines


class DiscreteSchedulingEnv(gym.Wrapper):
    """Wraps a continuous-action EVChargingEnv for DQN.

    Meta-action space — ``Discrete(5)``:
        0 = Uncontrolled  (max rate for all occupied stations)
        1 = FCFS           (first-come first-served priority)
        2 = EDF            (earliest-deadline-first priority)
        3 = Round-Robin    (equal capacity sharing)
        4 = Conservative   (50 % max rate for all)
    """

    STRATEGY_NAMES = ["Uncontrolled", "FCFS", "EDF", "Round-Robin", "Conservative"]

    def __init__(self, env: EVChargingEnv):
        super().__init__(env)

        # Override spaces
        self.action_space = spaces.Discrete(5)

        flat_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

        # Pre-built baseline policies (index-aligned with action ints)
        self._policies = [
            baselines.UncontrolledPolicy(),
            baselines.FCFSPolicy(),
            baselines.EDFPolicy(),
            baselines.RoundRobinPolicy(),
            None,  # Conservative — handled inline
        ]
        self._last_strategy: Optional[int] = None

    # ── helpers ──────────────────────────────────────────────────────────
    def _conservative_action(self) -> np.ndarray:
        action = np.zeros(self.env.num_stations, dtype=np.float32)
        for idx, sid in enumerate(self.env.station_ids):
            ev = self.env.simulator.network.get_ev(sid)
            if ev is not None and not ev.fully_charged:
                action[idx] = 0.5
        return action

    # ── Gymnasium API ────────────────────────────────────────────────────
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._last_strategy = None
        return obs.flatten().astype(np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._last_strategy = int(action)

        if action == 4:
            cont_action = self._conservative_action()
        else:
            cont_action = self._policies[action].act(self.env)

        obs, reward, terminated, truncated, info = self.env.step(cont_action)
        info["strategy_selected"] = self.STRATEGY_NAMES[action]
        return obs.flatten().astype(np.float32), reward, terminated, truncated, info

    @property
    def last_strategy(self) -> Optional[int]:
        return self._last_strategy
