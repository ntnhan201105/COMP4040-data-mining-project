"""Gymnasium environment wrapping ACN-Sim for EV charging scheduling."""
from datetime import datetime, timedelta, timezone
import math
from typing import List, Tuple, Dict, Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from acnportal.acnsim import Simulator, sites, EventQueue
from stage2 import event_loader


class EVChargingEnv(gym.Env):
    """Gymnasium environment wrapping ACN-Sim for step-by-step EV scheduling."""
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        site: str = "caltech",
        period: int = 5,
        voltage: float = 208.0,
        max_battery_power: float = 6.6,
        train_days: Optional[List[datetime]] = None,
        reward_alpha: float = 1.0,    # Energy delivery weight
        reward_beta: float = 0.5,     # Peak demand penalty weight
        reward_gamma: float = 0.1,    # Unfairness penalty weight
        site_capacity_kw: float = 150.0
    ):
        super().__init__()
        self.site = site.lower()
        if self.site not in ["caltech", "jpl"]:
            raise ValueError("site must be either 'caltech' or 'jpl'")

        self.period = period
        self.voltage = voltage
        self.max_battery_power = max_battery_power
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_gamma = reward_gamma
        self.site_capacity_kw = site_capacity_kw

        # Load days list
        if train_days is not None:
            self.days = train_days
        else:
            # Fallback to loading all days from JSON
            train_days, _ = event_loader.split_train_test(self.site, train_ratio=1.0)
            self.days = train_days

        self.day_index = 0

        # Initialize network to get station structure
        if self.site == "caltech":
            self.network = sites.caltech_acn(voltage=self.voltage)
        else:
            self.network = sites.jpl_acn(voltage=self.voltage)

        self.station_ids = self.network.station_ids
        self.num_stations = len(self.station_ids)
        self.max_pilot_signals = self.network.max_pilot_signals
        self.allowable_rates = self.network.allowable_rates

        # Action space: normalized charging rate [0, 1] for each station
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_stations,),
            dtype=np.float32
        )

        # Observation space: (num_stations, 8) matrix
        # Features: is_occupied, remaining_demand, time_until_departure, laxity,
        #           prev_charging_rate, energy_delivered_fraction, time_sin, time_cos
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_stations, 8),
            dtype=np.float32
        )

        self.simulator: Optional[Simulator] = None
        self.start_dt: Optional[datetime] = None
        self.prev_rates = np.zeros(self.num_stations, dtype=np.float32)
        self.ev_delivered_history: Dict[str, float] = {}

    def _get_obs(self) -> np.ndarray:
        """Build the observation matrix from the current simulator state."""
        obs = np.zeros((self.num_stations, 8), dtype=np.float32)
        current_iter = self.simulator.iteration
        total_iters = 288  # 24 hours at 5-minute intervals

        # Time-of-day encoding (shared across all stations)
        time_sin = math.sin(2 * math.pi * current_iter / total_iters)
        time_cos = math.cos(2 * math.pi * current_iter / total_iters)

        for idx, station_id in enumerate(self.station_ids):
            ev = self.simulator.network.get_ev(station_id)
            if ev is not None:
                # 1. is_occupied
                obs[idx, 0] = 1.0

                # 2. remaining_demand (normalized)
                obs[idx, 1] = ev.remaining_demand / max(1.0, ev.requested_energy)

                # 3. time_until_departure (normalized)
                remaining_time = max(0, ev.departure - current_iter)
                obs[idx, 2] = remaining_time / total_iters

                # 4. laxity
                # Minimum periods required to charge at maximum charging rate (kW)
                # max_rate in kW = max_pilot_signal * voltage / 1000
                max_ev_power = min(self.max_battery_power, self.max_pilot_signals[idx] * self.voltage / 1000.0)
                min_charge_periods = ev.remaining_demand / max(0.1, max_ev_power * (self.period / 60.0))
                laxity_val = remaining_time - min_charge_periods
                obs[idx, 3] = laxity_val / total_iters

                # 5. prev_charging_rate (normalized)
                obs[idx, 4] = self.prev_rates[idx]

                # 6. energy_delivered_fraction
                obs[idx, 5] = ev.energy_delivered / max(1.0, ev.requested_energy)
            else:
                # Unoccupied stations get 0 for EV-specific features
                obs[idx, 0] = 0.0
                obs[idx, 1] = 0.0
                obs[idx, 2] = 0.0
                obs[idx, 3] = 1.0  # Max laxity when unoccupied
                obs[idx, 4] = 0.0
                obs[idx, 5] = 0.0

            # 7 & 8: Time sin and cos
            obs[idx, 6] = time_sin
            obs[idx, 7] = time_cos

        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new day's simulation."""
        super().reset(seed=seed)

        # 1. Parse reset options or choose day
        day_sessions = None
        if options is not None:
            self.start_dt = options.get("start_dt")
            day_sessions = options.get("sessions")
            site = options.get("site")
            if site:
                self.site = site.lower()

        # If not provided in options, pick the next day in list
        if self.start_dt is None or day_sessions is None:
            if seed is not None:
                np.random.seed(seed)
            # Cycle through or pick random day
            day_idx = np.random.randint(0, len(self.days))
            self.start_dt = self.days[day_idx]
            
            end_dt = self.start_dt + timedelta(days=1)
            day_sessions = event_loader.load_sessions_from_json(self.site, self.start_dt, end_dt)

        # 2. Build event queue
        event_queue = event_loader.sessions_to_event_queue(
            sessions=day_sessions,
            start_dt=self.start_dt,
            period=self.period,
            voltage=self.voltage,
            max_battery_power=self.max_battery_power,
            force_feasible=True
        )

        # 3. Create network and simulator objects
        if self.site == "caltech":
            self.network = sites.caltech_acn(voltage=self.voltage)
        else:
            self.network = sites.jpl_acn(voltage=self.voltage)

        self.station_ids = self.network.station_ids
        self.max_pilot_signals = self.network.max_pilot_signals
        self.allowable_rates = self.network.allowable_rates

        self.simulator = Simulator(
            network=self.network,
            scheduler=None,
            events=event_queue,
            start=self.start_dt,
            period=self.period,
            verbose=False
        )
        self.simulator.max_recompute = 1
        self.simulator._last_schedule_update = 0

        # Reset trackers
        self.prev_rates = np.zeros(self.num_stations, dtype=np.float32)
        self.ev_delivered_history = {}

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply action, advance simulator by 1 period, return step results."""
        if self.simulator is None:
            raise RuntimeError("reset() must be called before step()")

        # Map action [0, 1] to pilot signal in Amperes, projected to nearest allowable rate
        schedule = {}
        for idx, station_id in enumerate(self.station_ids):
            target_rate = action[idx] * self.max_pilot_signals[idx]
            allowable = self.allowable_rates[idx]
            closest_idx = np.argmin(np.abs(allowable - target_rate))
            rate = float(allowable[closest_idx])
            
            schedule[station_id] = [rate]
            self.prev_rates[idx] = rate / self.max_pilot_signals[idx] if self.max_pilot_signals[idx] > 0 else 0.0

        # Force simulator to step exactly 1 period by clearing resolve flags
        self.simulator._resolve = False
        self.simulator._last_schedule_update = self.simulator._iteration

        # Advance simulation by one step
        terminated = self.simulator.step(schedule)

        # Check truncation (hard limit of 24h)
        truncated = self.simulator.iteration >= 288

        # Calculate reward
        reward = self._compute_reward(action)

        # Get observation and info
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute the step reward balancing energy, peak demand, and fairness."""
        # 1. Energy Delivered Reward
        # Track energy delivered to each EV in this specific step
        energy_delivered_step = 0.0
        active_evs = self.simulator.network.active_evs

        for ev in active_evs:
            prev_delivered = self.ev_delivered_history.get(ev.session_id, 0.0)
            delivered_now = ev.energy_delivered - prev_delivered
            energy_delivered_step += max(0.0, delivered_now)
            self.ev_delivered_history[ev.session_id] = ev.energy_delivered

        # 2. Peak Demand Penalty
        # Current rates are actual rates delivered after simulator's infrastructure constraints
        # Get actual charging rate array (in Amperes) from current step
        # iteration - 1 corresponds to the rates that were just applied
        curr_iter = max(0, self.simulator.iteration - 1)
        actual_rates = self.simulator.charging_rates[:, curr_iter]
        
        # Total active power in kW = sum(Amps * Volts / 1000)
        agg_power_kw = np.sum(actual_rates) * self.voltage / 1000.0
        peak_penalty = (agg_power_kw / self.site_capacity_kw) ** 2

        # 3. Unfairness Penalty (std of satisfaction of active EVs)
        satisfactions = []
        for ev in active_evs:
            ratio = ev.energy_delivered / max(1.0, ev.requested_energy)
            satisfactions.append(ratio)
        
        unfairness_penalty = np.std(satisfactions) if len(satisfactions) > 1 else 0.0

        # Combine
        reward = (
            self.reward_alpha * energy_delivered_step
            - self.reward_beta * peak_penalty
            - self.reward_gamma * unfairness_penalty
        )
        return float(reward)

    def _get_info(self) -> Dict[str, Any]:
        """Aggregate metrics into an info dictionary."""
        # Total energy delivered and requested up to now
        total_delivered = sum(ev.energy_delivered for ev in self.simulator.ev_history.values())
        total_requested = sum(ev.requested_energy for ev in self.simulator.ev_history.values())
        
        # Calculate peak demand over the whole episode
        voltage = self.voltage
        # charging_rates shape: (num_stations, num_iterations)
        if self.simulator.iteration > 0:
            actual_rates_history = self.simulator.charging_rates[:, :self.simulator.iteration]
            agg_power_history = np.sum(actual_rates_history, axis=0) * voltage / 1000.0
            peak_demand_kw = float(np.max(agg_power_history))
        else:
            peak_demand_kw = 0.0

        # Jain's fairness index of completed sessions
        completed_evs = [ev for ev in self.simulator.ev_history.values() if ev.departure <= self.simulator.iteration]
        if completed_evs:
            ratios = [ev.energy_delivered / max(1.0, ev.requested_energy) for ev in completed_evs]
            sum_ratios = sum(ratios)
            sum_sq_ratios = sum(r**2 for r in ratios)
            jain_fairness = (sum_ratios ** 2) / (len(ratios) * sum_sq_ratios) if sum_sq_ratios > 0 else 1.0
        else:
            jain_fairness = 1.0

        return {
            "total_energy_delivered_kwh": float(total_delivered),
            "total_energy_requested_kwh": float(total_requested),
            "peak_demand_kw": peak_demand_kw,
            "num_active_evs": len(self.simulator.network.active_evs),
            "num_completed_sessions": len(completed_evs),
            "jain_fairness": float(jain_fairness)
        }

    def render(self):
        """Optional rendering (print status to console)."""
        info = self._get_info()
        print(
            f"Step: {self.simulator.iteration:03d} | "
            f"Active EVs: {info['num_active_evs']} | "
            f"Delivered: {info['total_energy_delivered_kwh']:.2f} kWh | "
            f"Peak: {info['peak_demand_kw']:.2f} kW | "
            f"Fairness: {info['jain_fairness']:.2f}"
        )
