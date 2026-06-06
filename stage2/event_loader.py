"""Module to load charging sessions from local JSON files and convert them to ACN-Sim events with caching."""
import json
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from acnportal.acnsim import PluginEvent, EventQueue
from acnportal.acnsim.models.ev import EV
from acnportal.acnsim.models.battery import Battery

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "charging"

# Global memory cache for sessions to make loading and resets instantaneous
_sessions_cache: Dict[str, List[Dict[str, Any]]] = {}


def _get_cached_sessions(site: str) -> List[Dict[str, Any]]:
    """Load sessions from JSON and cache them in memory with pre-parsed datetime objects.

    Args:
        site (str): 'caltech' or 'jpl'.

    Returns:
        List[Dict[str, Any]]: Cached and pre-parsed session list.
    """
    site_key = site.lower()
    if site_key not in _sessions_cache:
        if site_key=='caltech':
            file_path = DATASET_DIR / f"{site_key}_sessions_full.json"
        elif site_key=='jpl':
            file_path = DATASET_DIR / f"{site_key}_sessions.json"
        else:
            raise ValueError(f"Invalid site: {site}")

        if not file_path.exists():
            raise FileNotFoundError(f"Session file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        sessions = raw_data.get("_items", [])
        date_fmt = "%a, %d %b %Y %H:%M:%S GMT"
        
        # Pre-parse dates to avoid repeated strptime calls in the loops
        for s in sessions:
            conn_str = s.get("connectionTime")
            disc_str = s.get("disconnectTime")
            if conn_str:
                s["connection_dt"] = datetime.strptime(conn_str, date_fmt).replace(tzinfo=timezone.utc)
            if disc_str:
                s["disconnect_dt"] = datetime.strptime(disc_str, date_fmt).replace(tzinfo=timezone.utc)

        _sessions_cache[site_key] = sessions
    return _sessions_cache[site_key]


def load_sessions_from_json(site: str, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
    """Load session data from cache and filter by date range.

    Args:
        site (str): 'caltech' or 'jpl'.
        start_dt (datetime): Start datetime (timezone-aware).
        end_dt (datetime): End datetime (timezone-aware).

    Returns:
        List[Dict[str, Any]]: List of filtered session documents.
    """
    sessions = _get_cached_sessions(site)
    filtered_sessions = []

    for session in sessions:
        conn_dt = session.get("connection_dt")
        if conn_dt and start_dt <= conn_dt < end_dt:
            filtered_sessions.append(session)

    return filtered_sessions


def get_date_ranges(site: str) -> Tuple[datetime, datetime]:
    """Get the full date range of sessions available in the local JSON.

    Args:
        site (str): 'caltech' or 'jpl'.

    Returns:
        Tuple[datetime, datetime]: (min_connection_dt, max_connection_dt).
    """
    sessions = _get_cached_sessions(site)
    dates = [s["connection_dt"] for s in sessions if "connection_dt" in s]

    if not dates:
        raise ValueError(f"No valid connection times found for site {site}")

    return min(dates), max(dates)


def split_train_test(site: str, train_ratio: float = 0.7) -> Tuple[List[datetime], List[datetime]]:
    """Split the available days into train and test sets chronologically.

    Args:
        site (str): 'caltech' or 'jpl'.
        train_ratio (float): Ratio of training days.

    Returns:
        Tuple[List[datetime], List[datetime]]: List of start_dts for train and test days.
    """
    min_date, max_date = get_date_ranges(site)
    
    start_day = datetime(min_date.year, min_date.month, min_date.day, tzinfo=timezone.utc)
    end_day = datetime(max_date.year, max_date.month, max_date.day, tzinfo=timezone.utc)
    
    all_days = []
    curr = start_day
    while curr <= end_day:
        all_days.append(curr)
        curr = curr + pd.Timedelta(days=1)

    split_idx = int(len(all_days) * train_ratio)
    train_days = all_days[:split_idx]
    test_days = all_days[split_idx:]
    
    return train_days, test_days


def sessions_to_event_queue(
    sessions: List[Dict[str, Any]],
    start_dt: datetime,
    period: int,
    voltage: float,
    max_battery_power: float,
    force_feasible: bool = False
) -> EventQueue:
    """Convert raw sessions to an ACN-Sim EventQueue.

    Args:
        sessions (List[Dict[str, Any]]): List of raw session dictionaries.
        start_dt (datetime): Start datetime of the simulation (timezone-aware UTC).
        period (int): Length of each simulation period in minutes.
        voltage (float): Operating voltage of the charging network.
        max_battery_power (float): Default maximum charging power for batteries (kW).
        force_feasible (bool): Capping energy requested by duration capability.

    Returns:
        EventQueue: EventQueue filled with PluginEvents.
    """
    events = []
    offset_ts = start_dt.timestamp() / (60 * period)

    for sess in sessions:
        conn_dt = sess.get("connection_dt")
        disc_dt = sess.get("disconnect_dt")
        
        if not conn_dt or not disc_dt:
            continue

        # Convert connection and disconnection times to simulation period indices
        arrival = int(conn_dt.timestamp() / (60 * period) - offset_ts)
        departure = int(disc_dt.timestamp() / (60 * period) - offset_ts)

        # Skip sessions that fall outside or end before the simulation window
        if arrival < 0 or departure <= arrival:
            continue

        delivered_energy = sess.get("kWhDelivered", 0)
        if delivered_energy <= 0:
            continue

        if force_feasible:
            # Maximum possible energy delivered in the connection window at max battery power
            duration_hrs = (departure - arrival) * (period / 60)
            delivered_energy = min(delivered_energy, max_battery_power * duration_hrs)

        session_id = sess.get("sessionID", "")
        station_id = sess.get("spaceID", "")

        # Create EV battery and vehicle models
        batt = Battery(capacity=delivered_energy, init_charge=0, max_power=max_battery_power)
        ev = EV(
            arrival=arrival,
            departure=departure,
            requested_energy=delivered_energy,
            station_id=station_id,
            session_id=session_id,
            battery=batt
        )
        
        events.append(PluginEvent(arrival, ev))

    events.sort(key=lambda x: x.timestamp)
    return EventQueue(events)
