"""Baseline scheduling policies for EV charging."""
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.optimize import linprog

from acnportal.acnsim import sites
from stage2.ev_charging_env import EVChargingEnv
from stage2 import event_loader


class UncontrolledPolicy:
    """Charge every occupied station at its maximum allowable rate."""
    def act(self, env: EVChargingEnv) -> np.ndarray:
        action = np.zeros(env.num_stations, dtype=np.float32)
        for idx, station_id in enumerate(env.station_ids):
            ev = env.simulator.network.get_ev(station_id)
            if ev is not None and not ev.fully_charged:
                action[idx] = 1.0  # Max rate
        return action


class FCFSPolicy:
    """Greedily allocate charging rates in order of EV arrival times."""
    def act(self, env: EVChargingEnv) -> np.ndarray:
        # Get active EVs and sort by arrival time
        active_evs = env.simulator.network.active_evs
        sorted_evs = sorted(active_evs, key=lambda ev: ev.arrival)

        action = np.zeros(env.num_stations, dtype=np.float32)
        temp_schedule = np.zeros((env.num_stations, 1), dtype=np.float32)

        for ev in sorted_evs:
            if ev.fully_charged:
                continue
            idx = env.station_ids.index(ev.station_id)
            allowable = env.allowable_rates[idx]
            
            # Sort allowable rates in descending order
            candidates = sorted(allowable, reverse=True)
            
            # Binary search to find the maximum feasible rate
            best_rate = 0.0
            temp_schedule[idx, 0] = candidates[0]
            if env.simulator.network.is_feasible(temp_schedule, linear=False):
                best_rate = candidates[0]
            else:
                low = 0
                high = len(candidates) - 1
                while low <= high:
                    mid = (low + high) // 2
                    rate = candidates[mid]
                    temp_schedule[idx, 0] = rate
                    if env.simulator.network.is_feasible(temp_schedule, linear=False):
                        best_rate = rate
                        high = mid - 1  # Try higher rates (smaller index)
                    else:
                        low = mid + 1   # Try lower rates (larger index)
            
            temp_schedule[idx, 0] = best_rate
            action[idx] = best_rate / env.max_pilot_signals[idx] if env.max_pilot_signals[idx] > 0 else 0.0
                    
        return action


class EDFPolicy:
    """Greedily allocate charging rates in order of EV departure deadlines."""
    def act(self, env: EVChargingEnv) -> np.ndarray:
        # Get active EVs and sort by estimated departure
        active_evs = env.simulator.network.active_evs
        # Fall back to departure if estimated_departure is None
        sorted_evs = sorted(
            active_evs,
            key=lambda ev: ev.estimated_departure if ev.estimated_departure is not None else ev.departure
        )

        action = np.zeros(env.num_stations, dtype=np.float32)
        temp_schedule = np.zeros((env.num_stations, 1), dtype=np.float32)

        for ev in sorted_evs:
            if ev.fully_charged:
                continue
            idx = env.station_ids.index(ev.station_id)
            allowable = env.allowable_rates[idx]
            
            # Sort allowable rates in descending order
            candidates = sorted(allowable, reverse=True)
            
            # Binary search to find the maximum feasible rate
            best_rate = 0.0
            temp_schedule[idx, 0] = candidates[0]
            if env.simulator.network.is_feasible(temp_schedule, linear=False):
                best_rate = candidates[0]
            else:
                low = 0
                high = len(candidates) - 1
                while low <= high:
                    mid = (low + high) // 2
                    rate = candidates[mid]
                    temp_schedule[idx, 0] = rate
                    if env.simulator.network.is_feasible(temp_schedule, linear=False):
                        best_rate = rate
                        high = mid - 1  # Try higher rates (smaller index)
                    else:
                        low = mid + 1   # Try lower rates (larger index)
            
            temp_schedule[idx, 0] = best_rate
            action[idx] = best_rate / env.max_pilot_signals[idx] if env.max_pilot_signals[idx] > 0 else 0.0
                    
        return action


class RoundRobinPolicy:
    """Share capacity equally among all occupied, non-fully-charged stations."""
    def act(self, env: EVChargingEnv) -> np.ndarray:
        active_evs = env.simulator.network.active_evs
        station_indices = [env.station_ids.index(ev.station_id) for ev in active_evs if not ev.fully_charged]

        action = np.zeros(env.num_stations, dtype=np.float32)
        if not station_indices:
            return action

        temp_schedule = np.zeros((env.num_stations, 1), dtype=np.float32)

        # Binary search for uniform rate cap C
        low = 0.0
        # Find maximum max rate among active stations
        high = float(max(env.max_pilot_signals[idx] for idx in station_indices))
        
        # Binary search over 6 iterations (precision of ~1 Amp for max 32A)
        for _ in range(6):
            mid = (low + high) / 2.0
            for idx in station_indices:
                allowable = env.allowable_rates[idx]
                below = allowable[allowable <= mid]
                temp_schedule[idx, 0] = max(below) if len(below) > 0 else 0.0
                
            if env.simulator.network.is_feasible(temp_schedule, linear=False):
                low = mid
            else:
                high = mid

        # Apply final rates at low
        rates = np.zeros(env.num_stations, dtype=np.float32)
        for idx in station_indices:
            allowable = env.allowable_rates[idx]
            below = allowable[allowable <= low]
            rates[idx] = max(below) if len(below) > 0 else 0.0
            
        # One pass of greedy incremental improvement for leftovers
        temp_schedule[:, 0] = rates
        for idx in station_indices:
            curr_rate = rates[idx]
            allowable = env.allowable_rates[idx]
            higher = allowable[allowable > curr_rate]
            if len(higher) > 0:
                next_rate = min(higher)
                temp_schedule[idx, 0] = next_rate
                if env.simulator.network.is_feasible(temp_schedule, linear=False):
                    rates[idx] = next_rate
                else:
                    temp_schedule[idx, 0] = curr_rate

        # Convert back to normalized actions
        for idx in station_indices:
            action[idx] = rates[idx] / env.max_pilot_signals[idx] if env.max_pilot_signals[idx] > 0 else 0.0
            
        return action


class MPCOraclePolicy:
    """MPC Oracle with perfect foresight of all events in the episode."""
    def solve(
        self,
        sessions: List[Dict[str, Any]],
        site: str,
        start_dt: datetime,
        period: int = 5,
        voltage: float = 208.0,
        max_battery_power: float = 6.6,
        peak_penalty_weight: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """Solve offline LP for optimal charging schedule over the 24h day.

        Returns:
            Dict[str, np.ndarray]: Dict mapping station_id to numpy array of rates (Amps) of length 288.
        """
        # 1. Parse and build list of simulation EVs
        # Re-use event_loader to ensure consistency
        queue = event_loader.sessions_to_event_queue(
            sessions=sessions,
            start_dt=start_dt,
            period=period,
            voltage=voltage,
            max_battery_power=max_battery_power,
            force_feasible=True
        )
        
        # Extract EVs
        evs = []
        while not queue.empty():
            event = queue.get_event()
            evs.append(event.ev)

        # 2. Setup network constraints
        if site.lower() == "caltech":
            network = sites.caltech_acn(voltage=voltage)
        else:
            network = sites.jpl_acn(voltage=voltage)

        station_ids = network.station_ids
        num_stations = len(station_ids)
        max_pilot_signals = network.max_pilot_signals
        
        # Get linearized constraints matrix and magnitudes
        # A_lin shape: (num_constraints, num_stations)
        A_lin = np.abs(network.constraint_matrix)
        b_lim = network.magnitudes
        num_constraints = A_lin.shape[0]

        T = 288  # number of periods in a day

        # Decision variables:
        # For each session i, and each time step t in [ev.arrival, ev.departure):
        # We define a rate variable r_{i, t}.
        # Also, peak power variable P (kW) as the last variable.
        # Let's map (i, t) to decision variable index.
        var_idx = 0
        ev_time_map = {}  # maps (ev_idx, t) -> var_idx
        var_to_ev_time = []  # maps var_idx -> (ev_idx, t)

        for i, ev in enumerate(evs):
            # Clamp arrival and departure to [0, T)
            arr = max(0, min(ev.arrival, T - 1))
            dep = max(0, min(ev.departure, T))
            for t in range(arr, dep):
                ev_time_map[(i, t)] = var_idx
                var_to_ev_time.append((i, t))
                var_idx += 1

        num_rate_vars = var_idx
        num_vars = num_rate_vars + 1  # +1 for Peak P (kW)
        idx_P = num_rate_vars  # Index of peak power variable P

        # Bounds: 0 <= r_{i,t} <= max_pilot_signal for the EV's station
        bounds = []
        for var in range(num_rate_vars):
            ev_idx, t = var_to_ev_time[var]
            ev = evs[ev_idx]
            station_idx = station_ids.index(ev.station_id)
            bounds.append((0.0, float(max_pilot_signals[station_idx])))
        # Peak power P bounds
        bounds.append((0.0, None))

        # Objective: Maximize sum(r_{i,t} * voltage/1000 * period/60) - peak_penalty_weight * P
        # Which is equivalent to minimizing -sum(r_{i,t} * voltage/1000 * period/60) + peak_penalty_weight * P
        c = np.zeros(num_vars)
        for var in range(num_rate_vars):
            c[var] = - (voltage / 1000.0) * (period / 60.0)
        c[idx_P] = peak_penalty_weight

        A_ub = []
        b_ub = []

        # Constraint 1: Energy request cap per EV i
        # sum_{t} r_{i,t} * voltage/1000 * period/60 <= ev.requested_energy
        for i, ev in enumerate(evs):
            row = np.zeros(num_vars)
            has_vars = False
            arr = max(0, min(ev.arrival, T - 1))
            dep = max(0, min(ev.departure, T))
            for t in range(arr, dep):
                key = (i, t)
                if key in ev_time_map:
                    row[ev_time_map[key]] = (voltage / 1000.0) * (period / 60.0)
                    has_vars = True
            if has_vars:
                A_ub.append(row)
                b_ub.append(float(ev.requested_energy))

        # Constraint 2: Peak Power Definition
        # For each time step t, total active power in kW <= P
        # sum_{i active at t} r_{i,t} * voltage / 1000 <= P
        # which is sum_{i active at t} r_{i,t} * voltage / 1000 - P <= 0
        for t in range(T):
            row = np.zeros(num_vars)
            has_vars = False
            for i, ev in enumerate(evs):
                key = (i, t)
                if key in ev_time_map:
                    row[ev_time_map[key]] = voltage / 1000.0
                    has_vars = True
            if has_vars:
                row[idx_P] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)

        # Constraint 3: Network physical constraints
        # For each time step t, and each constraint c in network:
        # sum_{i active at t} A_lin[c, station(i)] * r_{i,t} <= b_lim[c]
        for t in range(T):
            # Check which constraints are active or just enforce all
            for c_idx in range(num_constraints):
                row = np.zeros(num_vars)
                has_vars = False
                for i, ev in enumerate(evs):
                    key = (i, t)
                    if key in ev_time_map:
                        station_idx = station_ids.index(ev.station_id)
                        coeff = A_lin[c_idx, station_idx]
                        if coeff > 0:
                            row[ev_time_map[key]] = coeff
                            has_vars = True
                if has_vars:
                    A_ub.append(row)
                    b_ub.append(float(b_lim[c_idx]))

        if len(A_ub) == 0:
            # No sessions active today
            return {sid: np.zeros(T) for sid in station_ids}

        A_ub_mat = np.array(A_ub)
        b_ub_vec = np.array(b_ub)

        # Solve Linear Program
        res = linprog(c, A_ub=A_ub_mat, b_ub=b_ub_vec, bounds=bounds, method="highs")

        # Reconstruct schedule dict
        schedule = {sid: np.zeros(T) for sid in station_ids}
        if res.success:
            rates = res.x[:-1]
            for var in range(num_rate_vars):
                ev_idx, t = var_to_ev_time[var]
                ev = evs[ev_idx]
                schedule[ev.station_id][t] += rates[var]

        return schedule
