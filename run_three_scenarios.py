"""One-file runner: three speed-conditioned Markov driver experiments.

Run: python run_three_scenarios.py
Change the short settings block below. No configuration files or old project
modules are required. Dependencies: numpy, scipy, matplotlib; h5py for MAT v7.3.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ==================== НАСТРОЙКИ ====================
BASE_DIR = Path(os.environ.get("MARKOV_DATA_DIR", "data"))
DATA_DIRS = [BASE_DIR]
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "runs"

# Имена переменных в MAT. Каждый сигнал — [время в секундах, значение].
MAT_SIGNALS = {
    "accelerator": "AccPedalPos", "brake": "BrakePedalPos",
    "parking": "ParkingBrakeSwitch", "selector": "DNR_Dashboard",
    "steering": "SteeringWheelAngle",  # radians, converted to degrees below
    "door1": "DoorsOpen_CC", "door2": "DoorsOpen_CD", "door3": "DoorsOpen_CE",
    "motor_left": "L_INV2TM_MotorSpeed", "motor_right": "R_INV2TM_MotorSpeed",
}

# Параметры водителя: pedal (газ/тормоз), parking (ручник), selector (селектор),
# steering (угол руля), doors (хотя бы одна из трёх дверей открыта).
# pedal = акселератор минус тормоз, в процентах.
# doors = (дверь1 == 1) ИЛИ (дверь2 == 1) ИЛИ (дверь3 == 1) после выравнивания.
# Окружающая среда во всех сценариях — скорость, км/ч.
# dt — секунды; n — удалить суммарные обучающие счётчики строго меньше n.
# pedal_step — проценты; speed_step — км/ч; steering_step — градусы.
SCENARIOS = [
    dict(name="01_pedals_parking_selector", driver_signals=["pedal", "parking", "selector"],
         dt=0.1, n=10, pedal_step=0.5, speed_step=0.5),
    dict(name="02_pedals_parking_selector_doors", driver_signals=["pedal", "parking", "selector", "doors"],
         dt=0.01, n=5, pedal_step=0.5, speed_step=0.5),
    dict(name="03_pedals_parking_selector_steering_doors",
         driver_signals=["pedal", "parking", "selector", "steering", "doors"],
         dt=0.01, n=5, pedal_step=0.5, speed_step=0.5, steering_step=10.0),
]

TEST_SHARE = 0.20      # 20% целых логов для оценки, остальные — для обучения.
RANDOM_SEED = 2026    # Одинаковое разбиение для всех трёх сценариев.
WHEEL_RADIUS_M = 0.472
GEAR_RATIO = 22.823
STEERING_MAX_TURNS = 3.5  # От нуля в каждую сторону: ±1260°. За пределами — выброс.
# ==================== КОНЕЦ НАСТРОЕК ==========================


try:
    import numpy as np
    import scipy.io as sio
    import matplotlib
    matplotlib.use("Agg")
except ImportError as exc:
    raise SystemExit("Install scientific libraries once: python -m pip install numpy scipy matplotlib h5py\n" + str(exc))


"""Vectorized signal alignment and disk-backed transition counting.

This part is concatenated into the single-file runner. Its public functions
use canonical source names; all project-specific MATLAB field names belong
to the runner's small settings section.
"""

import json
import math
import sqlite3
from pathlib import Path
import time

import numpy as np


DRIVER_SIGNAL_NAMES = ("pedal", "parking", "selector", "steering", "doors")


def _positive_number(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return value


def validate_scenario(scenario):
    """Validate the few human-facing controls without building a state tensor."""
    result = dict(scenario)
    result["dt"] = _positive_number(result["dt"], "dt")
    result["pedal_step"] = _positive_number(result.get("pedal_step", .5), "pedal_step")
    result["speed_step"] = _positive_number(result.get("speed_step", .5), "speed_step")
    result["steering_step"] = _positive_number(result.get("steering_step", 10), "steering_step")
    threshold = result.get("n", 0)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, np.integer)) or threshold < 0:
        raise ValueError("n must be a nonnegative integer")
    result["n"] = int(threshold)
    signals = result.get("driver_signals", [])
    if not signals or len(set(signals)) != len(signals) or any(s not in DRIVER_SIGNAL_NAMES for s in signals):
        raise ValueError("driver_signals must contain unique names from " + str(DRIVER_SIGNAL_NAMES))
    result["driver_signals"] = list(signals)
    return result


def required_sources(scenario):
    result = {"motor_left", "motor_right"}
    for name in scenario["driver_signals"]:
        if name == "pedal":
            result.update(("accelerator", "brake"))
        elif name == "doors":
            result.update(("door1", "door2", "door3"))
        else:
            result.add(name)
    return sorted(result)


def _source_arrays(signal):
    if isinstance(signal, dict):
        if "time" in signal and "value" in signal:
            t, v = signal["time"], signal["value"]
        elif "Time" in signal and "Data" in signal:
            t, v = signal["Time"], signal["Data"]
        else:
            raise ValueError("signal structure requires time/value or Time/Data")
    else:
        array = np.asarray(signal)
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError("signal must be a numeric N x 2 [seconds, value] array")
        t, v = array[:, 0], array[:, 1]
    t, v = np.asarray(t), np.asarray(v)
    if np.iscomplexobj(t) or np.iscomplexobj(v):
        raise ValueError("complex-valued signal is unsupported")
    t, v = np.asarray(t, dtype=np.float64).reshape(-1), np.asarray(v, dtype=np.float64).reshape(-1)
    if len(t) != len(v) or not len(t):
        raise ValueError("signal time and value must be equally sized and nonempty")
    if not np.all(np.isfinite(t)):
        raise ValueError("nonfinite source timestamp")
    return t, v


def prepare_raw_signal(signal, max_gap=1.0):
    """Stable timestamp sort, last duplicate retained, and continuity labels.

    Null/NaN values are retained as barriers. A long source interval is never
    crossed by interpolation, including between finite exact endpoints.
    """
    if max_gap is not None:
        max_gap = _positive_number(max_gap, "max_gap")
    t, v = _source_arrays(signal)
    # Typical vehicle logs are already ordered. Avoid argsort and two large
    # copies for each multi-million-sample motor/steering channel.
    if np.any(t[1:] < t[:-1]):
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
    retain = np.r_[t[1:] != t[:-1], True]
    removed = int(len(t) - np.count_nonzero(retain))
    if removed:
        t, v = t[retain], v[retain]
    finite = np.isfinite(v)
    starts = finite.copy()
    if len(t) > 1:
        gaps = np.zeros(len(t)-1, dtype=bool)
        if max_gap is not None:
            delta = np.diff(t)
            gaps = delta > max_gap + 1e-12 * np.maximum(1., np.abs(delta))
        starts[1:] &= ~finite[:-1] | gaps
    segments = np.cumsum(starts, dtype=np.int64)
    segments[~finite] = 0
    return {"time": t, "value": v, "segment": segments,
            "duplicates_removed": removed, "max_gap": max_gap}


def resample_signal(prepared, times, method="previous"):
    """Vectorized interpolation with no extrapolation or continuity bridging."""
    if method not in ("previous", "linear"):
        raise ValueError("interpolation must be previous or linear")
    source_t, source_v, source_s = prepared["time"], prepared["value"], prepared["segment"]
    times = np.asarray(times, dtype=np.float64)
    value = np.full(times.shape, np.nan, dtype=np.float64)
    segment = np.zeros(times.shape, dtype=np.int64)
    left = np.searchsorted(source_t, times, side="right") - 1
    in_range = (left >= 0) & (times <= source_t[-1])
    safe = np.clip(left, 0, len(source_t)-1)
    exact = in_range & (times == source_t[safe])
    if np.any(exact):
        value[exact] = source_v[safe[exact]]
        segment[exact] = source_s[safe[exact]]
    middle = in_range & ~exact & (left < len(source_t)-1)
    positions = np.flatnonzero(middle)
    if len(positions):
        ii = left[positions]
        span = source_t[ii+1] - source_t[ii]
        allowed = source_s[ii] > 0
        max_gap = prepared["max_gap"]
        if max_gap is not None:
            allowed &= span <= max_gap + 1e-12*np.maximum(1., np.abs(span))
        if method == "linear":
            allowed &= source_s[ii] == source_s[ii+1]
        positions, ii = positions[allowed], ii[allowed]
        if len(positions):
            if method == "previous":
                value[positions] = source_v[ii]
            else:
                weight = (times[positions]-source_t[ii])/(source_t[ii+1]-source_t[ii])
                value[positions] = source_v[ii] + weight*(source_v[ii+1]-source_v[ii])
            segment[positions] = source_s[ii]
    return value, segment


def state_key(ticks):
    """Portable SQLite state key; components are quantization ticks/categories."""
    return json.dumps([int(x) for x in ticks], separators=(",", ":"))


def state_ticks_to_values(ticks, scenario):
    values = np.asarray(ticks, dtype=np.float64).copy()
    if values.shape[-1] != len(scenario["driver_signals"]):
        raise ValueError("state component count differs from driver_signals")
    for col, name in enumerate(scenario["driver_signals"]):
        if name == "pedal":
            values[..., col] *= scenario["pedal_step"]
        elif name == "steering":
            values[..., col] *= scenario["steering_step"]
    return values


def _round_ticks(values, step):
    # Midpoint ties go toward the larger physical value, also for negatives.
    values = np.asarray(values, dtype=np.float64)
    safe = np.where(np.isfinite(values), values/step, 0.)
    if np.any(np.abs(safe) >= 2**52):
        raise ValueError("signal magnitude is too large for exact quantization")
    # rad->deg and decimal steps can move an exact mathematical midpoint by
    # one floating-point ulp. A capped numerical tolerance keeps ties stable
    # without moving measurements materially below a boundary into a new bin.
    tolerance = np.minimum(1e-8, 4*np.finfo(np.float64).eps*np.maximum(1., np.abs(safe)))
    return np.floor(safe + .5 + tolerance).astype(np.int64)


def iter_aligned_blocks(raw, scenario, settings):
    """Yield bounded-size grid blocks with exactly one overlapping endpoint.

    All requested sources use their common observed interval. State components
    are quantization ticks; state_values gives physical percentages/degrees.
    For occupancy, skip the first row when new_sample_start==1. Each valid
    transition is present in exactly one block, including chunk boundaries.
    """
    scenario = validate_scenario(scenario)
    chunk = int(settings.get("chunk_samples", 200000))
    if chunk < 1:
        raise ValueError("chunk_samples must be positive")
    max_gap = settings.get("max_gap", 1.)
    steering_limit = settings.get("steering_limit_degrees")
    if steering_limit is not None:
        steering_limit = _positive_number(steering_limit, "steering_limit_degrees")
    steering_outliers = 0
    prepared = {}
    for name in required_sources(scenario):
        if name not in raw:
            raise ValueError("missing source: " + name)
        signal = raw[name]
        if name == "steering" and steering_limit is not None:
            source_t, source_v = _source_arrays(signal)
            # Reject raw radians BEFORE interpolation. Keeping invalid samples
            # as NaN prevents transitions or interpolated states across spikes.
            outlier = np.isfinite(source_v) & (np.abs(source_v) > math.radians(steering_limit))
            steering_outliers = int(np.count_nonzero(outlier))
            if steering_outliers:
                source_v = source_v.copy()
                source_v[outlier] = np.nan
                signal = {"time": source_t, "value": source_v}
        # Invalid raw categories/physical pedal samples remain barriers even
        # when a coarse grid does not land on the bad original measurement.
        if name in ("parking", "selector", "door1", "door2", "door3", "accelerator", "brake"):
            source_t, source_v = _source_arrays(signal)
            if name.startswith("door"):
                allowed = np.isin(source_v, (0., 1.))
            elif name == "parking":
                allowed = np.isin(source_v, settings.get("parking_values", (0., 1.)))
            elif name == "selector":
                categories = settings.get("selector_values", (0., 1., 2.))
                allowed = np.isfinite(source_v) & (source_v == np.floor(source_v))
                if categories is not None:
                    allowed &= np.isin(source_v, categories)
            else:
                allowed = np.isfinite(source_v) & (source_v >= 0.) & (source_v <= 100.)
            if not np.all(allowed):
                source_v = source_v.copy()
                source_v[~allowed] = np.nan
                signal = {"time": source_t, "value": source_v}
        prepared[name] = prepare_raw_signal(signal, max_gap)
    lower = max(p["time"][0] for p in prepared.values())
    upper = min(p["time"][-1] for p in prepared.values())
    dt = scenario["dt"]
    count = max(0, int(math.floor((upper-lower)/dt + 1e-10))+1) if upper >= lower else 0
    if count == 0:
        return
    speed_factor = _positive_number(settings.get("speed_factor", math.pi*.472*3.6/(22.823*60.)), "speed_factor")
    drivers = scenario["driver_signals"]
    duplicates = sum(p["duplicates_removed"] for p in prepared.values())
    for start in range(0, max(1, count-1), chunk):
        end = min(count, start+chunk+1)
        times = lower + np.arange(start, end, dtype=np.float64)*dt
        tolerance = 1e-10*max(1., dt)
        if len(times) and times[-1] > upper:
            if times[-1]-upper <= tolerance:
                times[-1] = upper
            else:
                times = times[:-1]
        values, labels = {}, []
        for name, prepared_source in prepared.items():
            if name in ("parking", "selector", "door1", "door2", "door3"):
                method = "previous"
            elif name in ("accelerator", "brake"):
                method = settings.get("pedal_method", "previous")
            elif name == "steering":
                method = settings.get("steering_method", "linear")
            else:
                method = settings.get("speed_method", "previous")
            values[name], segment = resample_signal(prepared_source, times, method)
            labels.append(segment)
        if not len(times):
            continue
        finite_sources = np.logical_and.reduce([np.isfinite(v) for v in values.values()])
        state_valid = np.ones(len(times), dtype=bool)
        columns = []
        for name in drivers:
            if name == "pedal":
                physical = values["accelerator"] - values["brake"]
                state_valid &= np.isfinite(physical) & (physical >= -100.-1e-10) & (physical <= 100.+1e-10)
                columns.append(_round_ticks(physical, scenario["pedal_step"]))
            elif name == "steering":
                physical = np.rad2deg(values["steering"])
                state_valid &= np.isfinite(physical)
                if steering_limit is not None:
                    state_valid &= np.abs(physical) <= np.nextafter(steering_limit, np.inf)
                columns.append(_round_ticks(physical, scenario["steering_step"]))
            elif name == "doors":
                door_values = [values["door"+str(i)] for i in (1, 2, 3)]
                door_valid = np.logical_and.reduce([np.isin(v, (0., 1.)) for v in door_values])
                state_valid &= door_valid
                columns.append(np.logical_or.reduce([v == 1. for v in door_values]).astype(np.int64))
            else:
                physical = values[name]
                allowed = np.isfinite(physical) & (physical == np.floor(physical))
                categories = settings.get(name+"_values", (0, 1) if name == "parking" else None)
                if categories is not None:
                    allowed &= np.isin(physical, categories)
                state_valid &= allowed
                columns.append(np.where(allowed, physical, 0.).astype(np.int64))
        state = np.column_stack(columns)
        speed = speed_factor*(values["motor_left"] + values["motor_right"])
        finite_speed = np.isfinite(speed)
        speed_ticks = np.where(finite_speed, speed/scenario["speed_step"], 0.)
        if np.any(np.abs(speed_ticks) >= 2**52):
            raise ValueError("speed magnitude is too large for exact binning")
        environment = np.floor(speed_ticks).astype(np.int64)
        segments = np.column_stack(labels)
        valid_samples = state_valid & finite_speed & finite_sources & np.all(segments > 0, axis=1)
        transition_valid = (state_valid[:-1] & state_valid[1:] & finite_speed[:-1]
                            & np.all(segments[:-1] > 0, axis=1)
                            & np.all(segments[:-1] == segments[1:], axis=1))
        # Finite quantized future controls may not be paired across an invalid
        # source endpoint; matching positive continuity labels enforce that.
        yield {"time": times, "state": state, "state_values": state_ticks_to_values(state, scenario),
               "environment": environment, "speed": speed,
               "state_valid": state_valid, "valid_samples": valid_samples,
               "transition_valid": transition_valid, "source_segments": segments,
               "new_sample_start": 0 if start == 0 else 1,
               "duplicates_removed": duplicates if start == 0 else 0,
               "steering_outliers_removed": steering_outliers if start == 0 else 0}


def _meta_set(connection, key, value):
    connection.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                       (key, json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)))


def open_training_db(path, scenario=None, settings=None):
    """Open a resumable model database. Existing counts are never cleared."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=120.)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS states(id INTEGER PRIMARY KEY,value_key TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS edges(e INTEGER NOT NULL,i INTEGER NOT NULL,j INTEGER NOT NULL,c INTEGER NOT NULL CHECK(c>0),
            PRIMARY KEY(e,i,j)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS initial(i INTEGER PRIMARY KEY,c INTEGER NOT NULL CHECK(c>0));
        CREATE TABLE IF NOT EXISTS logs(path TEXT PRIMARY KEY,status TEXT NOT NULL,grid_samples INTEGER NOT NULL DEFAULT 0,
            valid_transitions INTEGER NOT NULL DEFAULT 0,invalid_samples INTEGER NOT NULL DEFAULT 0,
            duplicate_timestamps INTEGER NOT NULL DEFAULT 0,steering_outliers INTEGER NOT NULL DEFAULT 0,
            seconds REAL NOT NULL DEFAULT 0,error TEXT) WITHOUT ROWID;
    """)
    if "steering_outliers" not in {row[1] for row in connection.execute("PRAGMA table_info(logs)")}:
        connection.execute("ALTER TABLE logs ADD COLUMN steering_outliers INTEGER NOT NULL DEFAULT 0")
    if scenario is not None:
        scenario = validate_scenario(scenario)
        stored = connection.execute("SELECT value FROM meta WHERE key='scenario'").fetchone()
        if stored is not None and json.loads(stored[0]) != scenario:
            connection.close()
            raise ValueError("existing database uses a different scenario; choose a new output folder")
        _meta_set(connection, "scenario", scenario)
    if settings is not None:
        relevant = {key: value for key, value in settings.items() if key not in ("chunk_samples",)}
        stored = connection.execute("SELECT value FROM meta WHERE key='alignment_settings'").fetchone()
        if stored is not None and json.loads(stored[0]) != relevant:
            connection.close()
            raise ValueError("existing database uses different preprocessing settings; choose a new output folder")
        _meta_set(connection, "alignment_settings", relevant)
    _meta_set(connection, "schema", "vasiliy-simple-three-scenarios-v1")
    _meta_set(connection, "state_encoding", "JSON integer ticks; pedal ticks*pedal_step percent; steering ticks*steering_step degrees; categories exact")
    _meta_set(connection, "fallback", "hold current driver state for an unseen or pruned-empty row")
    connection.commit()
    return connection


def log_is_complete(connection, path):
    row = connection.execute("SELECT status FROM logs WHERE path=?", (str(path),)).fetchone()
    return row is not None and row[0] in ("ok", "no_valid_transitions")


def record_log_error(connection, path, error):
    """Record a recoverable ingestion/alignment failure without changing counts."""
    if connection.in_transaction:
        connection.rollback()
    connection.execute("INSERT OR REPLACE INTO logs(path,status,error) VALUES(?,?,?)",
                       (str(path), "error", str(error)))
    connection.commit()


def _state_ids(connection, state, mask, cache):
    result = np.zeros(len(state), dtype=np.int64)
    if not np.any(mask):
        return result
    unique, inverse = np.unique(state[mask], axis=0, return_inverse=True)
    ids = np.empty(len(unique), dtype=np.int64)
    for q, ticks in enumerate(unique):
        key = state_key(ticks)
        identifier = cache.get(key)
        if identifier is None:
            connection.execute("INSERT OR IGNORE INTO states(value_key) VALUES(?)", (key,))
            identifier = int(connection.execute("SELECT id FROM states WHERE value_key=?", (key,)).fetchone()[0])
            if len(cache) >= 100000:
                cache.clear()
            cache[key] = identifier
        ids[q] = identifier
    result[mask] = ids[inverse]
    return result


def train_one_log(connection, path, raw, scenario, settings, state_cache=None):
    """Atomically add one log. Retry after interruption never double-counts it."""
    path = str(path)
    if log_is_complete(connection, path):
        return {"path": path, "status": "already_complete", "valid_transitions": 0}
    if connection.in_transaction:
        raise RuntimeError("train_one_log requires no pre-existing database transaction")
    scenario = validate_scenario(scenario)
    cache = {} if state_cache is None else state_cache
    report = dict(path=path, status="ok", grid_samples=0, valid_transitions=0,
                  invalid_samples=0, duplicate_timestamps=0, steering_outliers=0, seconds=0.)
    started = time.perf_counter()
    connection.execute("BEGIN IMMEDIATE")
    try:
        # Another process may have completed this log after our fast resume
        # check and before we acquired SQLite's write lock. Recheck inside
        # the transaction so two accidental launches cannot count it twice.
        if log_is_complete(connection, path):
            connection.rollback()
            return {"path": path, "status": "already_complete", "valid_transitions": 0}
        for block in iter_aligned_blocks(raw, scenario, settings):
            start = block["new_sample_start"]
            report["grid_samples"] += len(block["time"])-start
            report["invalid_samples"] += int(np.count_nonzero(~block["valid_samples"][start:]))
            report["duplicate_timestamps"] += block["duplicates_removed"]
            report["steering_outliers"] += block["steering_outliers_removed"]
            valid_pairs = block["transition_valid"]
            # Include both endpoints of accepted transitions even if a future
            # environment value itself will not be used for conditioning.
            state_mask = block["valid_samples"].copy()
            state_mask[:-1] |= valid_pairs
            state_mask[1:] |= valid_pairs
            ids = _state_ids(connection, block["state"], state_mask, cache)
            if np.any(valid_pairs):
                observations = np.column_stack((block["environment"][:-1][valid_pairs],
                                                ids[:-1][valid_pairs], ids[1:][valid_pairs]))
                unique, counts = np.unique(observations, axis=0, return_counts=True)
                payload = ((int(e), int(i), int(j), int(c)) for (e, i, j), c in zip(unique, counts))
                connection.executemany("INSERT INTO edges(e,i,j,c) VALUES(?,?,?,?) "
                    "ON CONFLICT(e,i,j) DO UPDATE SET c=c+excluded.c", payload)
                report["valid_transitions"] += int(np.count_nonzero(valid_pairs))
            initial_mask = block["valid_samples"].copy()
            if len(initial_mask)>1:
                initial_mask[1:] &= ~valid_pairs
            if start:
                initial_mask[0] = False
            if np.any(initial_mask):
                initial_ids, counts = np.unique(ids[initial_mask], return_counts=True)
                connection.executemany("INSERT INTO initial(i,c) VALUES(?,?) "
                    "ON CONFLICT(i) DO UPDATE SET c=c+excluded.c",
                    ((int(i), int(c)) for i, c in zip(initial_ids, counts)))
        report["status"] = "ok" if report["valid_transitions"] else "no_valid_transitions"
        report["seconds"] = time.perf_counter()-started
        connection.execute("INSERT OR REPLACE INTO logs(path,status,grid_samples,valid_transitions,invalid_samples,"
            "duplicate_timestamps,steering_outliers,seconds,error) VALUES(?,?,?,?,?,?,?,?,NULL)",
            tuple(report[k] for k in ("path", "status", "grid_samples", "valid_transitions", "invalid_samples", "duplicate_timestamps", "steering_outliers", "seconds")))
        _meta_set(connection, "training_complete", False)
        connection.commit()
    except Exception:
        connection.rollback()
        # IDs inserted into a rolled-back transaction must never remain cached.
        cache.clear()
        raise
    return report


def finalize_model(connection, n=None):
    """Apply the global threshold only after all training logs are aggregated.

    Raw counts remain unchanged. retained_edges is a view, so changing n does
    not duplicate a second large transition table. Probabilities normalize
    retained counts per (environment, source) row; empty rows are implicit holds.
    """
    if n is None:
        stored = connection.execute("SELECT value FROM meta WHERE key='scenario'").fetchone()
        if stored is None:
            raise ValueError("n or stored scenario is required")
        n = json.loads(stored[0])["n"]
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    n = int(n)
    connection.execute("DROP VIEW IF EXISTS retained_edges")
    connection.execute("CREATE VIEW retained_edges AS SELECT e,i,j,c FROM edges WHERE c >= " + str(n))
    connection.execute("DROP TABLE IF EXISTS row_totals")
    connection.execute("CREATE TABLE row_totals(e INTEGER NOT NULL,i INTEGER NOT NULL,total INTEGER NOT NULL,"
                       "out_degree INTEGER NOT NULL,PRIMARY KEY(e,i)) WITHOUT ROWID")
    connection.execute("INSERT INTO row_totals SELECT e,i,SUM(c),COUNT(*) FROM retained_edges GROUP BY e,i")
    raw_edges, raw_events = connection.execute("SELECT COUNT(*),COALESCE(SUM(c),0) FROM edges").fetchone()
    kept_edges, kept_events = connection.execute("SELECT COUNT(*),COALESCE(SUM(c),0) FROM retained_edges").fetchone()
    raw_rows = int(connection.execute("SELECT COUNT(*) FROM (SELECT 1 FROM edges GROUP BY e,i)").fetchone()[0])
    kept_rows = int(connection.execute("SELECT COUNT(*) FROM row_totals").fetchone()[0])
    states = int(connection.execute("SELECT COUNT(*) FROM states").fetchone()[0])
    environments = int(connection.execute("SELECT COUNT(DISTINCT e) FROM edges").fetchone()[0])
    logs = int(connection.execute("SELECT COUNT(*) FROM logs WHERE status='ok'").fetchone()[0])
    steering_outliers = int(connection.execute("SELECT COALESCE(SUM(steering_outliers),0) FROM logs").fetchone()[0])
    report = dict(min_count=n, states=states, speed_bins=environments, accepted_training_logs=logs,
                  steering_raw_outlier_samples=steering_outliers,
                  raw_edges=int(raw_edges), kept_edges=int(kept_edges), raw_events=int(raw_events),
                  kept_events=int(kept_events), removed_events=int(raw_events-kept_events),
                  removed_event_fraction=(raw_events-kept_events)/raw_events if raw_events else 0.,
                  raw_active_rows=raw_rows, retained_active_rows=kept_rows,
                  rows_emptied_by_pruning=raw_rows-kept_rows,
                  fallback="hold current driver state; fallback events are reported separately")
    _meta_set(connection, "n", n)
    _meta_set(connection, "training_summary", report)
    _meta_set(connection, "training_complete", True)
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return report


"""Evaluation and figures for the single-file three-scenario runner.

This source part is concatenated with training.py and the small launch section.
It does not import the earlier article project.
"""
import collections
import csv
import itertools
import json
import math
from pathlib import Path
import random
import sqlite3
import time
from contextlib import closing

import numpy as np


_EV_PAIR_DTYPE = np.dtype([('e', '<i8'), ('i', '<i8')])
_EV_EDGE_DTYPE = np.dtype([('e', '<i8'), ('i', '<i8'), ('j', '<i8')])


def _ev_pairs(e, i):
    result = np.empty(len(i), dtype=_EV_PAIR_DTYPE)
    result['e'], result['i'] = e, i
    return result


def _ev_edges(e, i, j):
    result = np.empty(len(i), dtype=_EV_EDGE_DTYPE)
    result['e'], result['i'], result['j'] = e, i, j
    return result


def _ev_aggregate(e, i, j, c=None):
    keys = _ev_edges(e, i, j)
    if not len(keys):
        return np.empty((0, 4), dtype=np.int64)
    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    weights = np.ones(len(keys), dtype=np.int64) if c is None else np.asarray(c, dtype=np.int64)[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    return np.column_stack((keys['e'][starts], keys['i'][starts], keys['j'][starts],
                            np.add.reduceat(weights, starts)))


class _EvRows:
    """Compressed *observed* rows: no allocation proportional to N*K."""
    def __init__(self, edges):
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 4)
        if len(edges):
            if np.any(edges[:, 3] <= 0):
                raise ValueError('Transition counts must be positive')
            edges = _ev_aggregate(*edges[:, :3].T, edges[:, 3])
        self.edges = edges
        self.keys = _ev_edges(*edges[:, :3].T)
        if len(edges):
            starts = np.r_[0, np.flatnonzero(np.any(edges[1:, :2] != edges[:-1, :2], axis=1)) + 1]
            self.rowkeys = _ev_pairs(edges[starts, 0], edges[starts, 1])
            self.indptr = np.r_[starts, len(edges)].astype(np.int64)
            self.totals = np.add.reduceat(edges[:, 3], starts)
            repeated_totals = np.repeat(self.totals, np.diff(self.indptr))
            self.probability = edges[:, 3].astype(float) / repeated_totals
            self.sumsq = np.add.reduceat(self.probability ** 2, starts)
            self.entropy = np.add.reduceat(-self.probability * np.log(self.probability), starts)
            row_number = np.repeat(np.arange(len(starts)), np.diff(self.indptr))
            rank = np.lexsort((edges[:, 2], -self.probability, row_number))
            ordered_rows = row_number[rank]
            first_rank = np.r_[0, np.flatnonzero(ordered_rows[1:] != ordered_rows[:-1]) + 1]
            ordinal = np.arange(len(rank)) - np.repeat(first_rank, np.diff(np.r_[first_rank, len(rank)]))
            self.top5 = np.full((len(starts), 5), np.iinfo(np.int64).min, dtype=np.int64)
            selected = ordinal < 5
            self.top5[ordered_rows[selected], ordinal[selected]] = edges[rank[selected], 2]
            self.winner = self.top5[:, 0]
            self.confidence = self.probability[rank[first_rank]]
        else:
            self.rowkeys = np.empty(0, dtype=_EV_PAIR_DTYPE)
            self.indptr = np.zeros(1, dtype=np.int64)
            self.totals = self.probability = self.sumsq = self.entropy = self.confidence = np.empty(0)
            self.winner = np.empty(0, dtype=np.int64)
            self.top5 = np.empty((0, 5), dtype=np.int64)
        self._sampling_cache = {}

    def row_positions(self, e, i):
        keys = _ev_pairs(e, i)
        positions = np.searchsorted(self.rowkeys, keys)
        found = positions < len(self.rowkeys)
        if len(self.rowkeys):
            found &= self.rowkeys[np.minimum(positions, len(self.rowkeys)-1)] == keys
        return positions, found

    def scores(self, e, i, j, fallback_i=None):
        i, j = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
        e = np.broadcast_to(e, i.shape).astype(np.int64, copy=False)
        fallback_i = i if fallback_i is None else np.asarray(fallback_i, dtype=np.int64)
        positions, found = self.row_positions(e, i)
        p = (fallback_i == j).astype(float)
        winner = fallback_i.copy()
        sumsq = np.ones(len(i))
        confidence = np.ones(len(i))
        top5 = fallback_i == j
        support = np.zeros(len(i), dtype=np.int64)
        if np.any(found):
            rr = positions[found]
            winner[found], sumsq[found] = self.winner[rr], self.sumsq[rr]
            confidence[found] = self.confidence[rr]
            top5[found] = np.any(self.top5[rr] == j[found, None], axis=1)
            support[found] = self.totals[rr]
            query = _ev_edges(e[found], i[found], j[found])
            where = np.searchsorted(self.keys, query)
            exists = where < len(self.keys)
            exists &= self.keys[np.minimum(where, len(self.keys)-1)] == query
            values = np.zeros(len(query))
            values[exists] = self.probability[where[exists]]
            p[found] = values
        return dict(p=p, winner=winner, sumsq=sumsq, confidence=confidence,
                    top5=top5, fallback=~found, support=support)

    def sample(self, e, i, rng):
        key = (int(e), int(i))
        cached = self._sampling_cache.get(key)
        if cached is None:
            query = np.array(key, dtype=_EV_PAIR_DTYPE)
            row = int(np.searchsorted(self.rowkeys, query))
            if row >= len(self.rowkeys) or self.rowkeys[row] != query:
                cached = (None, None)
            else:
                lo, hi = self.indptr[row:row+2]
                cdf = np.cumsum(self.probability[lo:hi])
                cdf[-1] = 1.0
                cached = (self.edges[lo:hi, 2], cdf)
            self._sampling_cache[key] = cached
        targets, cdf = cached
        if targets is None:
            return int(i), True
        pos = min(int(np.searchsorted(cdf, rng.random(), side='right')), len(targets)-1)
        return int(targets[pos]), False


class _EvStates:
    def __init__(self, rows, scenario):
        self.lookup = {}
        self.ticks = {}
        self.scenario = scenario
        self.next_unknown = -1
        for state, key in rows:
            tick = tuple(int(x) for x in json.loads(key))
            self.lookup[tick] = int(state)
            self.ticks[int(state)] = tick
        self.training_state_ids = np.array(sorted(self.ticks), dtype=np.int64)

    def ids(self, ticks):
        if not len(ticks):
            return np.empty(0, dtype=np.int64)
        unique, inverse = np.unique(ticks, axis=0, return_inverse=True)
        ids = []
        for row in unique:
            key = tuple(int(x) for x in row)
            state = self.lookup.get(key)
            if state is None:
                state = self.next_unknown
                self.next_unknown -= 1
                self.lookup[key] = state
                self.ticks[state] = key
            ids.append(state)
        return np.asarray(ids, dtype=np.int64)[inverse]

    def values(self, state):
        row = self.ticks[int(state)]
        return tuple(float(v) * (self.scenario.get('pedal_step', .5) if name == 'pedal'
                                 else self.scenario.get('steering_step', 10.) if name == 'steering' else 1.)
                     for name, v in zip(self.scenario['driver_signals'], row))


def _ev_record(identity=''):
    return dict(log=identity, events=0, zero=0, positive_nll=0., clipped_nll=0., brier=0.,
                correct=0, top5=0, fallback=0, tp=0, fp=0, fn=0, tn=0)


def _ev_add(record, score, current, target, counts, epsilon=1e-12):
    counts = np.asarray(counts, dtype=np.int64)
    p = score['p']
    record['events'] += int(counts.sum())
    record['zero'] += int(counts[p == 0].sum())
    nonzero = p > 0
    record['positive_nll'] += float(np.dot(counts[nonzero], -np.log(p[nonzero])))
    record['clipped_nll'] += float(np.dot(counts, -np.log(np.maximum(p, epsilon))))
    record['brier'] += float(np.dot(counts, np.maximum(0., 1 - 2*p + score['sumsq'])))
    record['correct'] += int(counts[score['winner'] == target].sum())
    record['top5'] += int(counts[score['top5']].sum())
    record['fallback'] += int(counts[score['fallback']].sum())
    actual, predicted = target != current, score['winner'] != current
    for key, mask in [('tp', actual & predicted), ('fp', ~actual & predicted),
                      ('fn', actual & ~predicted), ('tn', ~actual & ~predicted)]:
        record[key] += int(counts[mask].sum())


def _ev_summary(records):
    result = _ev_record()
    for record in records:
        for key in result:
            if key != 'log':
                result[key] += record[key]
    total = result['events']
    if not total:
        return dict(events=0, status='No valid held-out transitions')
    tp, fp, fn = result['tp'], result['fp'], result['fn']
    return dict(events=total, strict_nll_nats=None if result['zero'] else result['positive_nll']/total,
                strict_nll_is_infinite=bool(result['zero']), clipped_nll_nats=result['clipped_nll']/total,
                brier=result['brier']/total, top1_accuracy=result['correct']/total,
                top5_accuracy=result['top5']/total, zero_probability_fraction=result['zero']/total,
                fallback_fraction=result['fallback']/total,
                change_precision=tp/(tp+fp) if tp+fp else None,
                change_recall=tp/(tp+fn) if tp+fn else None,
                change_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                change_confusion={key: result[key] for key in ('tp', 'fp', 'fn', 'tn')})


def _ev_write_json(path, data):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    default=convert, allow_nan=False), encoding='utf-8')


def _ev_write_csv(path, rows):
    if isinstance(rows, (list,tuple)):
        if not rows:
            return
        fields = list(dict.fromkeys(key for row in rows for key in row))
    else:
        iterator=iter(rows)
        first=next(iterator,None)
        if first is None:
            return
        fields=list(first)
        rows=itertools.chain([first],iterator)
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ev_structural(raw, n_states, db_path, scenario, output):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    thresholds = sorted(set((0, int(scenario['n']), 2*int(scenario['n']), 5*int(scenario['n']))))
    raw_rows = _EvRows(raw)
    contexts = len(np.unique(raw[:, 0])) if len(raw) else 0
    total = int(raw[:, 3].sum()) if len(raw) else 0
    summaries, structure_plot = [], None
    selected_order = np.arange(n_states, dtype=np.int64)
    for threshold in thresholds:
        edges = raw[raw[:, 3] >= threshold]
        retained_rows = _EvRows(edges)
        if len(edges):
            i, j = edges[:, 1]-1, edges[:, 2]-1
            if np.any(i < 0) or np.any(i >= n_states) or np.any(j < 0) or np.any(j >= n_states):
                raise ValueError('SQLite state IDs must be contiguous, one-based IDs')
            graph = coo_matrix((np.ones(len(i), dtype=bool), (i, j)), shape=(n_states, n_states)).tocsr()
            graph = graph + graph.T
            graph.setdiag(0); graph.eliminate_zeros()
            order = reverse_cuthill_mckee(graph, symmetric_mode=True)
            inverse = np.empty(n_states, dtype=np.int64); inverse[order] = np.arange(n_states)
            lower0, upper0 = max(0, int(np.max(i-j))), max(0, int(np.max(j-i)))
            lower, upper = max(0, int(np.max(inverse[i]-inverse[j]))), max(0, int(np.max(inverse[j]-inverse[i])))
            if lower+upper > lower0+upper0:
                order = inverse = np.arange(n_states)
                lower, upper = lower0, upper0
            # A band position is invertible for every retained entry; no entry
            # is discarded and no multi-gigabyte dense band is allocated.
            offset = inverse[j]-inverse[i]
            assert np.all((-lower <= offset) & (offset <= upper))
            assert np.array_equal(order[inverse[i]+offset], j)
        else:
            i = j = np.empty(0, dtype=np.int64)
            order = inverse = np.arange(n_states)
            lower0 = upper0 = lower = upper = 0
        pos, found = retained_rows.row_positions(raw_rows.rowkeys['e'], raw_rows.rowkeys['i'])
        kept_counts = np.zeros(len(raw_rows.rowkeys), dtype=np.int64)
        kept_counts[found] = retained_rows.totals[pos[found]]
        removed_fraction = 1-kept_counts/raw_rows.totals if len(raw_rows.totals) else np.empty(0)
        active = len(retained_rows.rowkeys)
        csr = len(edges)*16 + (active+1)*8 + active*16
        band = int(n_states)*contexts*(lower+upper+1)*8
        summary = dict(n=threshold, observations=total, retained_observations=int(edges[:, 3].sum()),
            removed_observations=total-int(edges[:, 3].sum()),
            removed_observation_fraction=(total-int(edges[:, 3].sum()))/total if total else 0.,
            raw_edge_types=len(raw), retained_edge_types=len(edges), observed_state_count=n_states,
            observed_speed_bins=contexts, active_rows=active,
            rows_emptied=int(np.sum(kept_counts == 0)),
            worst_row_removed_fraction=float(np.max(removed_fraction)) if len(removed_fraction) else 0.,
            lower_before=lower0, upper_before=upper0, lower_after=lower, upper_after=upper,
            exact_band_probability_bytes=band, compressed_sparse_probability_bytes=csr,
            ordering_bytes=16*n_states, dense_probability_bytes=int(n_states)**2*contexts*8,
            band_coordinate_roundtrip='PASS for every retained transition',
            convention='float64 probabilities + int64 target IDs/pointers; compressed sparse includes 2 int64 row keys; implicit fallback; counts, object overhead and state decoder excluded')
        summaries.append(summary)
        if threshold == int(scenario['n']):
            selected_order = order
            stride = max(1, math.ceil(len(i)/100000))
            structure_plot = (i[::stride], j[::stride], inverse[i[::stride]], inverse[j[::stride]])
    _ev_write_csv(output/'threshold_memory.csv', summaries)
    _ev_write_json(output/'structural_metrics.json', dict(thresholds=summaries,
        sqlite_file_bytes=Path(db_path).stat().st_size,
        graph_note='One shared undirected union graph over all observed speed contexts; RCM is heuristic; original ordering retained if narrower.',
        observed_state_note='Only actually observed driver tuples receive IDs; no Cartesian state grid or N*K row table is allocated.',
        threshold_note='Structural sensitivity only. These thresholds are not selected using held-out test scores.'))
    np.savez_compressed(output/'band_ordering.npz', new_to_old=selected_order+1,
                        old_to_new=np.argsort(selected_order)+1)
    return summaries, structure_plot


def _ev_distribution(a, b, ordered=False):
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return dict(tv=None, js_nats=None, wasserstein=None)
    keys = sorted(a.keys() | b.keys())
    p = np.array([a.get(x, 0)/na for x in keys]); q = np.array([b.get(x, 0)/nb for x in keys])
    mixture = (p+q)/2
    positive_p, positive_q = p > 0, q > 0
    js = .5*(np.sum(p[positive_p]*np.log(p[positive_p]/mixture[positive_p])) +
             np.sum(q[positive_q]*np.log(q[positive_q]/mixture[positive_q])))
    w1 = float(np.sum(np.abs(np.cumsum(p-q)[:-1])*np.diff(keys))) if ordered else None
    return dict(tv=float(np.sum(np.abs(p-q))/2), js_nats=float(js), wasserstein=w1)


def _ev_complete_dwells(states, valid_pair):
    """Return complete internal residence runs; segment-edge runs are censored."""
    states = np.asarray(states)
    if not len(states):
        return []
    cuts = np.r_[0, np.flatnonzero(~np.asarray(valid_pair, dtype=bool))+1, len(states)]
    complete = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        seq = states[lo:hi]
        changes = np.r_[0, np.flatnonzero(seq[1:] != seq[:-1])+1, len(seq)]
        complete.extend(np.diff(changes)[1:-1].tolist())
    return complete


def _ev_survival(counter, dt):
    if not counter:
        return [], []
    values = np.array(sorted(counter), dtype=float)
    counts = np.array([counter[x] for x in sorted(counter)], dtype=float)
    return np.r_[0., values*dt].tolist(), np.r_[1., 1-np.cumsum(counts)/counts.sum()].tolist()


def _ev_acf(values, valid_pair, lags):
    values = np.asarray(values, dtype=float)
    invalid_prefix = np.r_[0, np.cumsum(~np.asarray(valid_pair, dtype=bool))]
    result = []
    for lag in lags:
        if lag >= len(values):
            result.append(None); continue
        valid = invalid_prefix[lag:] == invalid_prefix[:-lag]
        x, y = values[:-lag][valid], values[lag:][valid]
        good = np.isfinite(x) & np.isfinite(y); x, y = x[good], y[good]
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            result.append(None)
        else:
            result.append(float(np.corrcoef(x, y)[0, 1]))
    return result


def _ev_bootstrap(per_model, repeats, seed):
    names = list(per_model)
    if not names:
        return {}
    indices = [z for z, r in enumerate(per_model[names[0]]) if r['events']]
    if len(indices) < 2 or not repeats:
        return dict(status='At least two nonempty held-out logs are required for a log-bootstrap interval.')
    rng = np.random.default_rng(seed)
    fields = ('clipped_nll_nats', 'brier', 'top1_accuracy')
    samples = {name: {field: [] for field in fields} for name in names}
    paired = {name: {field: [] for field in fields} for name in names if name != 'conditional'}
    for _ in range(repeats):
        draw = rng.choice(indices, size=len(indices), replace=True)
        scores = {name: _ev_summary([records[z] for z in draw]) for name, records in per_model.items()}
        for name in names:
            for field in fields:
                samples[name][field].append(scores[name][field])
                if name != 'conditional':
                    paired[name][field].append(scores['conditional'][field]-scores[name][field])
    ci = lambda values: np.quantile(values, [.025, .975]).tolist()
    return dict(replicates=repeats, log_count=len(indices),
        note='Whole-log bootstrap with the same draw for every model. Requires independent logs; correlated fragments or repeated-driver logs can understate uncertainty.',
        ci95={name: {field: ci(values) for field, values in scores.items()} for name, scores in samples.items()},
        conditional_minus_baseline_ci95={name: {field: ci(values) for field, values in scores.items()} for name, scores in paired.items()})


def _ev_collect_replay(raw, scenario, settings, block_iterator):
    max_samples = int(settings.get('max_replay_samples', 100000))
    pieces = {'time': [], 'state': [], 'environment': [], 'speed': [], 'valid_samples': [], 'transition_valid': []}
    remaining = max_samples
    for block in block_iterator(raw, scenario, settings):
        start = int(block.get('new_sample_start', 0))
        count = min(remaining, len(block['time'])-start)
        if count <= 0:
            break
        end = start+count
        for key in ('time', 'state', 'environment', 'speed', 'valid_samples'):
            pieces[key].append(np.asarray(block[key])[start:end])
        edge_start = max(0, start-1)
        pieces['transition_valid'].append(np.asarray(block['transition_valid'])[edge_start:max(edge_start, end-1)])
        remaining -= count
        if not remaining:
            break
    return {key: np.concatenate(value, axis=0) if value else np.empty((0, len(scenario['driver_signals'])), dtype=np.int64) if key == 'state' else np.empty(0)
            for key, value in pieces.items()}


def _ev_replay(rows, state_map, test_paths, scenario, settings, read_raw, block_iterator):
    selected = list(test_paths)
    selection_rng = random.Random(int(settings.get('seed', 2026)))
    selection_rng.shuffle(selected)
    selected = selected[:int(settings.get('max_replay_files', 10))]
    repeats = int(settings.get('eval_rollouts', 3))
    rng = np.random.default_rng(int(settings.get('seed', 2026)))
    observed, generated = collections.Counter(), collections.Counter()
    obs_dw, gen_dw = collections.Counter(), collections.Counter()
    names = scenario['driver_signals']
    obs_signal = {name: collections.Counter() for name in names}
    gen_signal = {name: collections.Counter() for name in names}
    logs, preview = [], None
    fallback = decisions = 0
    for path in selected:
        try:
            data = _ev_collect_replay(read_raw(path, scenario, settings), scenario, settings, block_iterator)
            if not len(data['time']):
                logs.append(dict(path=str(path), status='no_common_grid')); continue
            actual = state_map.ids(data['state'])
            valid_samples = np.asarray(data['valid_samples'], dtype=bool)
            valid_pair = np.asarray(data['transition_valid'], dtype=bool)
            valid_pair &= valid_samples[:-1] & valid_samples[1:]
            obs_dw.update(_ev_complete_dwells(actual, valid_pair))
            ids, counts = np.unique(actual[valid_samples], return_counts=True)
            observed.update(dict(zip(ids.tolist(), counts.tolist())))
            for state, count in zip(ids, counts):
                for name, value in zip(names, state_map.values(state)):
                    obs_signal[name][value] += int(count)
            first_generated = None
            for repeat in range(repeats):
                rollout = actual.copy()
                for z in range(len(actual)-1):
                    if valid_pair[z]:
                        rollout[z+1], missing = rows.sample(int(data['environment'][z]), int(rollout[z]), rng)
                        fallback += int(missing); decisions += 1
                    else:
                        rollout[z+1] = actual[z+1]
                ids, counts = np.unique(rollout[valid_samples], return_counts=True)
                generated.update(dict(zip(ids.tolist(), counts.tolist())))
                for state, count in zip(ids, counts):
                    for name, value in zip(names, state_map.values(state)):
                        gen_signal[name][value] += int(count)
                gen_dw.update(_ev_complete_dwells(rollout, valid_pair))
                if first_generated is None:
                    first_generated = rollout
            logs.append(dict(path=str(path), status='ok', grid_samples=len(actual),
                             valid_samples=int(valid_samples.sum()), repeats=repeats,
                             truncation_limit=int(settings.get('max_replay_samples', 100000))))
            if preview is None and first_generated is not None:
                limit = min(20000, len(actual))
                actual_values = np.array([state_map.values(s) for s in actual[:limit]])
                generated_values = np.array([state_map.values(s) for s in first_generated[:limit]])
                actual_values[~valid_samples[:limit]] = np.nan
                generated_values[~valid_samples[:limit]] = np.nan
                preview = dict(time=data['time'][:limit], observed=actual_values, generated=generated_values,
                    environment=data['environment'][:limit], speed=data['speed'][:limit], valid_pair=valid_pair[:max(0, limit-1)],
                    path=str(path))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            logs.append(dict(path=str(path), status='error', error=str(exc)))
    marginals = {}
    for name in names:
        metric = _ev_distribution(obs_signal[name], gen_signal[name], name in ('pedal', 'steering'))
        metric['observed'] = sorted(obs_signal[name].items())
        metric['generated'] = sorted(gen_signal[name].items())
        marginals[name] = metric
    acf = {}
    if preview:
        dt = float(scenario['dt'])
        lags = sorted(set([1]+[max(1, round(value/dt)) for value in (.1, .5, 1, 2, 5, 10)]))
        for col, name in enumerate(names):
            if name == 'selector':
                continue
            acf[name] = dict(lags_s=[lag*dt for lag in lags],
                observed=_ev_acf(preview['observed'][:, col], preview['valid_pair'], lags),
                generated=_ev_acf(preview['generated'][:, col], preview['valid_pair'], lags))
    result = dict(mode='Recorded-speed replay. Vehicle dynamics are not simulated; this is not a closed-loop driving test.',
        replay_logs=logs, repeats=repeats, seed=int(settings.get('seed', 2026)),
        selected_log_limit=int(settings.get('max_replay_files', 10)),
        samples_per_log_limit=int(settings.get('max_replay_samples', 100000)),
        observed_samples=sum(observed.values()), generated_samples=sum(generated.values()),
        fallback_fraction=fallback/decisions if decisions else None,
        state_occupancy=_ev_distribution(observed, generated), signal_marginals=marginals,
        complete_dwell_observed_count=sum(obs_dw.values()), complete_dwell_generated_count=sum(gen_dw.values()),
        mean_complete_dwell_observed_s=float(scenario['dt'])*sum(k*v for k,v in obs_dw.items())/sum(obs_dw.values()) if obs_dw else None,
        mean_complete_dwell_generated_s=float(scenario['dt'])*sum(k*v for k,v in gen_dw.items())/sum(gen_dw.values()) if gen_dw else None,
        dwell_note='First and last residence runs of every uninterrupted segment are censored and excluded. Survival curves describe complete internal runs only.',
        dwell_observed_survival=_ev_survival(obs_dw, float(scenario['dt'])),
        dwell_generated_survival=_ev_survival(gen_dw, float(scenario['dt'])),
        autocorrelation_preview=acf,
        acf_note='Gap-aware lagged Pearson correlation of at most 20000 samples from the first selected usable log, not a full-corpus estimate.')
    return result, preview


def _ev_plots(output, scenario, structural, structure, selected_rows, metrics, replay, preview, raw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'savefig.dpi': 180})
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(output/(name+'.'+ext), bbox_inches='tight')
        plt.close(fig)
    x = [row['n'] for row in structural]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for key, label in [('compressed_sparse_probability_bytes', 'Compressed sparse'), ('exact_band_probability_bytes', 'Exact band')]:
        axes[0].plot(x, [max(1, row[key])/1024**2 for row in structural], 'o-', label=label)
    axes[0].set(xlabel='Minimum count n', ylabel='Probability buffers (MiB)', yscale='log'); axes[0].legend()
    axes[1].plot(x, [100*row['removed_observation_fraction'] for row in structural], 'o-', label='All observations')
    axes[1].plot(x, [100*row['worst_row_removed_fraction'] for row in structural], 's--', label='Worst row')
    axes[1].set(xlabel='Minimum count n', ylabel='Removed empirical mass (%)'); axes[1].legend()
    save(fig, 'threshold_memory_and_mass')
    if structure:
        ii, jj, pi, pj = structure
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        for ax, a, b, title in zip(axes, (jj, pj), (ii, pi), ('Observed-state numbering', 'Shared union-graph ordering')):
            ax.scatter(a, b, s=.3, rasterized=True); ax.invert_yaxis()
            ax.set(xlabel='Target', ylabel='Source', title=title)
        save(fig, 'transition_sparsity')
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    if len(selected_rows.rowkeys):
        axes[0].hist(np.diff(selected_rows.indptr), bins=40)
        axes[1].hist(selected_rows.entropy, bins=40)
    axes[0].set(xlabel='Positive outgoing destinations', ylabel='Observed rows')
    axes[1].set(xlabel='Row entropy (nats)', ylabel='Observed rows')
    save(fig, 'row_support_entropy')
    speed_bins, speed_inverse = np.unique(raw[:, 0], return_inverse=True) if len(raw) else ([], [])
    if len(speed_bins):
        counts = np.bincount(speed_inverse, weights=raw[:, 3])
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar((speed_bins+.5)*scenario['speed_step'], counts/max(1, counts.sum()), width=scenario['speed_step']*.85)
        ax.set(xlabel='Speed context (km/h)', ylabel='Training event fraction')
        save(fig, 'training_speed_coverage')
    nonempty = {name: value for name,value in metrics['models'].items() if value.get('events')}
    if nonempty:
        names = list(nonempty)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
        for ax, field, label in zip(axes, ('clipped_nll_nats', 'brier', 'top1_accuracy'), ('Clipped NLL (nats)', 'Brier score', 'Top-1 accuracy')):
            ax.bar(range(len(names)), [nonempty[name][field] for name in names])
            ax.set_xticks(range(len(names)), names, rotation=35, ha='right'); ax.set_ylabel(label)
        save(fig, 'predictive_baselines')
        cal = [row for row in metrics['calibration'] if row['events']]
        fig, ax = plt.subplots(figsize=(4, 3.7))
        ax.plot([0,1], [0,1], 'k--')
        ax.plot([row['mean_confidence'] for row in cal], [row['accuracy'] for row in cal], 'o-')
        ax.set(xlabel='Top-1 predicted confidence', ylabel='Top-1 empirical accuracy', xlim=(0,1), ylim=(0,1))
        save(fig, 'calibration')
        by_speed = metrics['by_speed']
        if by_speed:
            fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            sx = [row['speed_center_kmh'] for row in by_speed]
            axes[0].plot(sx, [row['clipped_nll_nats'] for row in by_speed], '.-')
            axes[0].set_ylabel('Clipped NLL (nats)')
            axes[1].plot(sx, [100*row['fallback_fraction'] for row in by_speed], '.-', label='Fallback')
            axes[1].plot(sx, [100*row['zero_probability_fraction'] for row in by_speed], '.-', label='Zero probability')
            axes[1].set(xlabel='Speed context center (km/h)', ylabel='Test events (%)'); axes[1].legend()
            save(fig, 'predictive_by_speed')
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.bar(range(len(names)), [nonempty[name]['change_f1'] if nonempty[name]['change_f1'] is not None else np.nan for name in names])
        ax.set_xticks(range(len(names)), names, rotation=30, ha='right')
        ax.set_ylabel('State-change F1 (argmax decision)')
        save(fig, 'state_change_f1')
    if replay is None or not replay.get('generated_samples'):
        return
    names = scenario['driver_signals']
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.5*len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        for key, label in [('observed', 'Observed'), ('generated', 'Replay')]:
            points = replay['signal_marginals'][name][key]
            total = sum(c for value, c in points)
            ax.plot([value for value,c in points], [c/total for value,c in points], 'o-', ms=2.5, label=label)
        ax.set(xlabel=name+(' (deg)' if name=='steering' else ' (%)' if name=='pedal' else ''), ylabel='Probability'); ax.legend()
    save(fig, 'replay_signal_distributions')
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for key, label in [('dwell_observed_survival', 'Observed'), ('dwell_generated_survival', 'Replay')]:
        xx, yy = replay[key]
        if xx:
            ax.step(xx, yy, where='post', label=label)
    ax.set(xlabel='Complete internal state residence (s)', ylabel='P(D > duration)')
    if replay['complete_dwell_observed_count'] or replay['complete_dwell_generated_count']:
        ax.legend()
    else:
        ax.text(.5,.5,'No uncensored internal residence runs',ha='center',transform=ax.transAxes)
    save(fig, 'replay_dwell_survival')
    acf = replay['autocorrelation_preview']
    if acf:
        fig, axes = plt.subplots(len(acf), 1, figsize=(7, 2.3*len(acf)), squeeze=False)
        for ax, (name, row) in zip(axes[:, 0], acf.items()):
            ax.plot(row['lags_s'], row['observed'], 'o-', label='Observed')
            ax.plot(row['lags_s'], row['generated'], 's--', label='Replay')
            ax.set(xlabel='Lag (s)', ylabel=name+' correlation'); ax.legend()
        save(fig, 'replay_autocorrelation')
    if preview:
        length = min(len(preview['time']), max(2, round(120/scenario['dt'])))
        fig, axes = plt.subplots(len(names)+1, 1, figsize=(10, 2*(len(names)+1)), sharex=True)
        tt = preview['time'][:length]-preview['time'][0]
        axes[0].step(tt, preview['speed'][:length], where='post')
        axes[0].set_ylabel('Speed (km/h)')
        for col, (ax, name) in enumerate(zip(axes[1:], names)):
            ax.step(tt, preview['observed'][:length, col], where='post', label='Observed', alpha=.8)
            ax.step(tt, preview['generated'][:length, col], where='post', label='Replay', alpha=.7)
            ax.set_ylabel(name+(' (deg)' if name=='steering' else ' (%)' if name=='pedal' else '')); ax.legend(loc='upper right')
        axes[-1].set_xlabel('Time since replay segment start (s)')
        save(fig, 'replay_example')


def evaluate_scenario(db_path, test_paths, scenario, settings, read_raw, output_dir, *, block_iterator=None):
    """Evaluate held-out logs completely, then bounded recorded-speed replay.

    ``read_raw(path,scenario,settings)`` is supplied by the launch section.
    Training's block iterator is looked up after concatenating source parts.
    """
    started = time.perf_counter()
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if block_iterator is None:
        block_iterator = iter_aligned_blocks
    with closing(sqlite3.connect(str(db_path))) as connection:
        batches=[]
        cursor=connection.execute('SELECT e,i,j,c FROM edges ORDER BY e,i,j')
        while True:
            batch=cursor.fetchmany(100000)
            if not batch:
                break
            batches.append(np.asarray(batch,dtype=np.int64))
        raw=np.concatenate(batches,axis=0) if batches else np.empty((0,4),dtype=np.int64)
        del batches
        state_map = _EvStates(connection.execute('SELECT id,value_key FROM states ORDER BY id'), scenario)
        training_paths = {str(row[0]) for row in connection.execute("SELECT path FROM logs WHERE status='ok'")}
        initial_entries=np.asarray(list(connection.execute('SELECT i,c FROM initial ORDER BY i')),dtype=np.int64).reshape(-1,2)
        training_logs=[dict(zip(('path','status','grid_samples','valid_transitions','invalid_samples','seconds','error'),row))
            for row in connection.execute('SELECT path,status,grid_samples,valid_transitions,invalid_samples,seconds,error FROM logs ORDER BY path')]
    test_paths = list(test_paths)
    normalized = [str(Path(path).resolve()).casefold() for path in test_paths]
    if len(set(normalized)) != len(normalized):
        raise ValueError('Repeated test log path')
    trained = {str(Path(path).resolve()).casefold() for path in training_paths}
    if trained & set(normalized):
        raise ValueError('A log appears in both training and test sets')
    known_ids = state_map.training_state_ids
    if len(known_ids) and not np.array_equal(known_ids, np.arange(1,len(known_ids)+1)):
        raise ValueError('Model state IDs must be contiguous and one-based')
    rows_raw = _EvRows(raw)
    rows = rows_raw if int(scenario['n']) <= 1 else _EvRows(raw[raw[:,3] >= int(scenario['n'])])
    unconditional = _EvRows(_ev_aggregate(np.zeros(len(raw), dtype=np.int64), raw[:,1], raw[:,2], raw[:,3]))
    context = _EvRows(_ev_aggregate(raw[:,0], np.zeros(len(raw),dtype=np.int64), raw[:,2], raw[:,3]))
    occupancy = _EvRows(_ev_aggregate(np.zeros(len(raw), dtype=np.int64), np.zeros(len(raw),dtype=np.int64), raw[:,2], raw[:,3]))
    models = ('conditional','conditional_unpruned','unconditional','context_only','persistence','occupancy')
    per_model = {name: [] for name in models}
    speed_records = {}
    cal_count, cal_confidence, cal_correct = np.zeros(10), np.zeros(10), np.zeros(10)
    input_audit = []
    for file_number, path in enumerate(test_paths, 1):
        print(f"  Test [{file_number}/{len(test_paths)}] {Path(path).name}", flush=True)
        records = {name: _ev_record(str(path)) for name in models}
        log_cal_count,log_cal_confidence,log_cal_correct=np.zeros(10),np.zeros(10),np.zeros(10)
        log_speed_records={}
        success=False
        try:
            raw_signals = read_raw(path, scenario, settings)
            for block in block_iterator(raw_signals, scenario, settings):
                ids = state_map.ids(block['state'])
                valid = np.asarray(block['transition_valid'], dtype=bool)
                if not np.any(valid):
                    continue
                events = _ev_aggregate(np.asarray(block['environment'])[:-1][valid], ids[:-1][valid], ids[1:][valid])
                e, i, j, counts = events.T
                zero = np.zeros(len(i), dtype=np.int64)
                definitions = [('conditional', rows, e, i), ('conditional_unpruned', rows_raw, e, i),
                    ('unconditional', unconditional, zero, i), ('context_only', context, e, zero),
                    ('occupancy', occupancy, zero, zero)]
                for name, lookup, query_e, query_i in definitions:
                    score = lookup.scores(query_e, query_i, j, fallback_i=i)
                    _ev_add(records[name], score, i, j, counts)
                    if name == 'conditional':
                        bins = np.minimum(9, (score['confidence']*10).astype(int))
                        log_cal_count += np.bincount(bins, weights=counts, minlength=10)
                        log_cal_confidence += np.bincount(bins, weights=counts*score['confidence'], minlength=10)
                        log_cal_correct += np.bincount(bins, weights=counts*(score['winner']==j), minlength=10)
                        for speed in np.unique(e):
                            mask = e == speed
                            subset = {key: value[mask] for key,value in score.items()}
                            _ev_add(log_speed_records.setdefault(int(speed), _ev_record(str(speed))), subset,
                                    i[mask], j[mask], counts[mask])
                persistence = dict(p=(i==j).astype(float), winner=i, sumsq=np.ones(len(i)),
                    confidence=np.ones(len(i)), top5=i==j, fallback=np.zeros(len(i), dtype=bool))
                _ev_add(records['persistence'], persistence, i, j, counts)
            input_audit.append(dict(path=str(path),status='ok',valid_transitions=records['conditional']['events']))
            success=True
        except (ValueError, TypeError, KeyError, OSError) as exc:
            # A failed log contributes no partial score to the aggregate.
            records = {name: _ev_record(str(path)) for name in models}
            input_audit.append(dict(path=str(path),status='error',error=str(exc)))
        if success:
            cal_count+=log_cal_count;cal_confidence+=log_cal_confidence;cal_correct+=log_cal_correct
            for speed,record in log_speed_records.items():
                target=speed_records.setdefault(speed,_ev_record(str(speed)))
                for key in record:
                    if key!='log':
                        target[key]+=record[key]
        for name in models:
            per_model[name].append(records[name])
    scores = {name: _ev_summary(records) for name,records in per_model.items()}
    for name, records in per_model.items():
        singles = [_ev_summary([record]) for record in records if record['events']]
        if singles:
            scores[name]['macro_per_log'] = {key: float(np.mean([row[key] for row in singles]))
                for key in ('clipped_nll_nats','brier','top1_accuracy','fallback_fraction')}
    calibration = [dict(bin_left=z/10, bin_right=(z+1)/10, events=int(cal_count[z]),
        mean_confidence=float(cal_confidence[z]/cal_count[z]) if cal_count[z] else None,
        accuracy=float(cal_correct[z]/cal_count[z]) if cal_count[z] else None) for z in range(10)]
    ece = sum(row['events']*abs(row['mean_confidence']-row['accuracy']) for row in calibration if row['events'])/sum(cal_count) if sum(cal_count) else None
    metrics = dict(models=scores, calibration=calibration, top1_ece=ece,
        by_speed=[dict(speed_bin=e,speed_center_kmh=(e+.5)*scenario['speed_step'], **_ev_summary([record]))
                  for e,record in sorted(speed_records.items()) if record['events']],
        test_logs=input_audit, test_file_count=len(test_paths),
        baseline_note='All trainable baselines use raw training counts. Conditional uses globally pruned counts; persistence is deterministic holding.',
        nll_note='Strict NLL is +infinity if any observed outcome has zero probability (JSON null plus infinite flag). Clipped NLL separately uses epsilon=1e-12.',
        support_note='Unknown test driver tuples have distinct negative diagnostic IDs; empty rows use an explicit self-loop. They are never added to training counts.',
        split_note='Whole logs are held out. This does not establish driver/route/day independence if log fragments share an origin.',
        cross_scenario_note='NLL and exact-state accuracy are not directly comparable across different dt or output dimensions. Also compare common pedal marginals and coverage.',
        bootstrap=_ev_bootstrap(per_model,int(settings.get('bootstrap',200)),int(settings.get('seed',2026))))
    metrics['training_coverage']=dict(logs=len(training_logs),accepted_logs=sum(row['status']=='ok' for row in training_logs),
        rejected_logs=sum(row['status']=='error' for row in training_logs),
        empty_logs=sum(row['status']=='no_valid_transitions' for row in training_logs),
        grid_samples=sum(row['grid_samples'] for row in training_logs),
        invalid_samples=sum(row['invalid_samples'] for row in training_logs),
        valid_transition_duration_s=sum(row['valid_transitions'] for row in training_logs)*float(scenario['dt']))
    structural, structure = _ev_structural(raw, len(known_ids), db_path, scenario, output)
    replay, preview = (None,None)
    if test_paths and int(settings.get('eval_rollouts',3)) > 0:
        replay, preview = _ev_replay(rows,state_map,test_paths,scenario,settings,read_raw,block_iterator)
        _ev_write_json(output/'replay_metrics.json',replay)
    _ev_write_json(output/'metrics.json', metrics)
    _ev_write_csv(output/'model_scores.csv', [dict(model=name,**{key:value for key,value in row.items() if not isinstance(value,dict)}) for name,row in scores.items()])
    _ev_write_csv(output/'scores_by_speed.csv', metrics['by_speed'])
    _ev_write_csv(output/'test_log_audit.csv', input_audit)
    _ev_write_csv(output/'training_log_audit.csv',training_logs)
    for name, records in per_model.items():
        _ev_write_csv(output/('per_log_'+name+'.csv'),[dict(path=record['log'],**_ev_summary([record])) for record in records])
    ordered_ticks = np.array([state_map.ticks[int(s)] for s in known_ids],dtype=np.int64)
    physical_states=np.array([state_map.values(s) for s in known_ids],dtype=float)
    from scipy.io import savemat
    model_export=dict(state_ids=known_ids,state_ticks=ordered_ticks,state_values=physical_states,
        driver_signal_names=np.asarray(scenario['driver_signals'],dtype=object),
        row_speed_bin=rows.rowkeys['e'],row_current_state=rows.rowkeys['i'],
        row_start_1based=rows.indptr+1,target_state=rows.edges[:,2],probability=rows.probability,
        retained_count=rows.edges[:,3],dt=float(scenario['dt']),minimum_count=int(scenario['n']),
        speed_step_kmh=float(scenario['speed_step']),pedal_step_percent=float(scenario['pedal_step']),
        steering_step_degrees=float(scenario.get('steering_step',10.)),
        configuration_json=json.dumps(scenario,ensure_ascii=False),schema='vasiliy-observed-state-csr-v1',
        initial_state_ids=initial_entries[:,0],initial_state_counts=initial_entries[:,1],
        convention='IDs and row_start are one-based. Speed bin e denotes [e*step,(e+1)*step). Only observed rows are stored. Missing rows hold the current state.')
    if settings.get('steering_limit_degrees') is not None:
        model_export['steering_limit_degrees']=float(settings['steering_limit_degrees'])
    savemat(str(output/'model_compact.mat'),model_export,do_compression=True,oned_as='column')
    if settings.get('save_npz',False):
        np.savez_compressed(output/'transition_model.npz',state_ids=known_ids,state_ticks=ordered_ticks,
            state_values=physical_states,row_speed_bin=rows.rowkeys['e'],
            row_current_state=rows.rowkeys['i'],indptr=rows.indptr,target_state=rows.edges[:,2],
            probability=rows.probability,configuration=np.array(json.dumps(scenario,ensure_ascii=False)))
    _ev_write_json(output/'model_format.json',dict(schema='vasiliy-observed-state-csr-v1',
        file='model_compact.mat',state_indexing='one-based observed state IDs',
        row_start_indexing='one-based offsets; last offset is number of edges + 1',
        speed_bins='e corresponds to [e*speed_step_kmh,(e+1)*speed_step_kmh)',
        fallback='hold current driver state if its (speed_bin,state_id) row is absent',
        instructions='Find row with matching row_speed_bin and row_current_state. Its edges are row_start_1based(r):row_start_1based(r+1)-1; sample target_state using probability. Decode state_values(target_state,:).'))
    _ev_write_csv(output/'state_decoder.csv',(dict(state_id=int(s),**dict(zip(scenario['driver_signals'],state_map.values(s)))) for s in known_ids))
    if settings.get('make_figures',True):
        _ev_plots(output,scenario,structural,structure,rows,metrics,replay,preview,raw)
    selected = next(row for row in structural if row['n']==int(scenario['n']))
    summary = dict(scenario=scenario.get('name',scenario.get('id','')),driver_signals=', '.join(scenario['driver_signals']),
        dt=scenario['dt'],n=scenario['n'],training_observations=selected['observations'],
        retained_observations=selected['retained_observations'],states=len(known_ids),
        raw_edge_types=len(raw),retained_edge_types=len(rows.edges),
        removed_observation_fraction=selected['removed_observation_fraction'],
        compressed_sparse_probability_mib=selected['compressed_sparse_probability_bytes']/1024**2,
        exact_band_probability_mib=selected['exact_band_probability_bytes']/1024**2,
        test_events=scores['conditional'].get('events',0),
        strict_nll_nats=scores['conditional'].get('strict_nll_nats'),
        strict_nll_is_infinite=scores['conditional'].get('strict_nll_is_infinite'),
        clipped_nll_nats=scores['conditional'].get('clipped_nll_nats'),
        brier=scores['conditional'].get('brier'),top1_accuracy=scores['conditional'].get('top1_accuracy'),
        test_fallback_fraction=scores['conditional'].get('fallback_fraction'),
        evaluation_seconds=time.perf_counter()-started,output_directory=str(output))
    _ev_write_json(output/'scenario_summary.json',summary)
    return summary


"""MAT input, reproducible file split, and sequential experiment launch."""
import argparse
import csv
import gc
import hashlib
import logging
import traceback
import random
from datetime import datetime


class InputChangedError(RuntimeError):
    """A frozen source log changed during a multi-hour experiment."""


def freeze_input_reader(manifest, reader=None):
    expected = {str(Path(row['path']).resolve()).casefold(): (row['bytes'], row['mtime_ns']) for row in manifest}
    loader = read_raw if reader is None else reader
    def frozen(path, scenario, settings):
        p = Path(path)
        key = str(p.resolve()).casefold()
        if key not in expected:
            raise InputChangedError('Input is absent from the frozen manifest: '+str(p))
        def check():
            state = p.stat()
            if (state.st_size,state.st_mtime_ns) != expected[key]:
                raise InputChangedError('Source MAT changed during the run: '+str(p)+'. Restart the script to create a new consistent run.')
        check()
        raw = loader(path,scenario,settings)
        check()
        return raw
    return frozen


def _simple_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _simple_csv(path, records):
    records = list(records)
    if not records:
        return
    keys = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _nested_mat_value(obj, path):
    for name in path.split("."):
        if isinstance(obj, dict):
            obj = obj[name]
        elif hasattr(obj, name):
            obj = getattr(obj, name)
        else:
            obj = obj[name]
    return obj


def _numeric_mat_signal(value, label):
    if isinstance(value, dict):
        if "time" in value and "value" in value:
            return np.column_stack((np.asarray(value["time"]).reshape(-1), np.asarray(value["value"]).reshape(-1)))
        if "Time" in value and "Data" in value:
            return np.column_stack((np.asarray(value["Time"]).reshape(-1), np.asarray(value["Data"]).reshape(-1)))
    array = np.asarray(value)
    if array.ndim == 1 and array.size == 2:
        array = array.reshape(1, 2)  # scipy simplify_cells squeezes one-row channels
    if array.ndim != 2 or array.shape[1] < 2 or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(label + ": expected a real numeric N x 2 [seconds,value] signal")
    return np.asarray(array[:, :2], dtype=np.float64)


def read_raw(path, scenario, settings):
    """Load only the sources required by this scenario and preserve numpy arrays."""
    names = required_sources(scenario)
    fields = {name: MAT_SIGNALS[name] for name in names}
    path = Path(path)
    try:
        import h5py
    except ImportError:
        h5py = None
    raw = {}
    if h5py is not None and h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            for name, field in fields.items():
                node = handle
                for component in field.split("."):
                    node = node[component]
                if not isinstance(node, h5py.Dataset) or h5py.check_dtype(ref=node.dtype) is not None:
                    raise ValueError(field + ": object/cell MATLAB v7.3 channels must be exported as numeric [time,value] arrays")
                array = np.asarray(node)
                if node.attrs.get("MATLAB_class") is not None and array.ndim > 1:
                    array = array.transpose(tuple(reversed(range(array.ndim))))
                raw[name] = _numeric_mat_signal(array, field)
    else:
        loaded = sio.loadmat(path, variable_names=sorted({f.split(".")[0] for f in fields.values()}), simplify_cells=True)
        for name, field in fields.items():
            try:
                raw[name] = _numeric_mat_signal(_nested_mat_value(loaded, field), field)
            except KeyError as exc:
                raise ValueError("Missing MATLAB signal: " + field) from exc
    return raw


def discover_mat_files(directories):
    files, seen = [], set()
    for location in directories:
        location = Path(location)
        if location.is_file() and location.suffix.lower() == ".mat":
            candidates = [location]
        elif location.is_dir():
            candidates = sorted(location.glob("*.mat"), key=lambda p: str(p).casefold())
        else:
            logging.warning("Input unavailable: %s", location)
            continue
        for path in candidates:
            resolved = path.resolve()
            key = str(resolved).casefold()
            if key not in seen:
                files.append(resolved)
                seen.add(key)
    return sorted(files, key=lambda p: str(p).casefold())


def make_file_split(files, test_share, seed):
    """Identical log basenames in different folders always share a partition."""
    if not 0 < test_share < 1:
        raise ValueError("TEST_SHARE must be between zero and one")
    groups = sorted({p.stem.casefold() for p in files})
    if len(groups) < 2:
        raise ValueError("At least two differently named MAT logs are required for separate training and testing. The supplied single log is sufficient for --check-sample only.")
    random.Random(seed).shuffle(groups)
    count = min(len(groups)-1, max(1, int(round(len(groups)*test_share))))
    test_groups = set(groups[:count])
    train = [p for p in files if p.stem.casefold() not in test_groups]
    test = [p for p in files if p.stem.casefold() in test_groups]
    return train, test


def _experiment_settings(seed, fast=False):
    return dict(max_gap=1.0, chunk_samples=200000,
        pedal_method="previous", speed_method="previous", steering_method="linear",
        speed_factor=math.pi*WHEEL_RADIUS_M*3.6/(GEAR_RATIO*60.),
        steering_limit_degrees=_positive_number(360*STEERING_MAX_TURNS, "STEERING_MAX_TURNS"),
        seed=seed, bootstrap=20 if fast else 200,
        eval_rollouts=1 if fast else 3, max_replay_files=2 if fast else 10,
        max_replay_samples=5000 if fast else 100000)


def _configure_logging(folder):
    folder.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_log = logging.FileHandler(folder/"run.log", encoding="utf-8")
    file_log.setFormatter(formatter)
    root_logger.addHandler(console)
    root_logger.addHandler(file_log)


def _export_training_progress(connection, folder, ordered_paths):
    columns = [row[1] for row in connection.execute("PRAGMA table_info(logs)")]
    records = [dict(zip(columns, row)) for row in connection.execute("SELECT * FROM logs ORDER BY path")]
    _simple_csv(folder/"training_logs.csv", records)
    by_path = {r["path"]: r for r in records}
    cumulative, total = [], 0
    for q, path in enumerate(ordered_paths, 1):
        row = by_path.get(str(path), {})
        total += row.get("valid_transitions", 0)
        cumulative.append(dict(processed_log=q, counted_transitions=total, path=str(path), status=row.get("status", "not_processed")))
    _simple_csv(folder/"training_progress.csv", cumulative)
    if cumulative:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot([r["processed_log"] for r in cumulative], [r["counted_transitions"] for r in cumulative])
        ax.set(xlabel="Processed training logs", ylabel="Cumulative transition observations")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(folder/("training_progress."+suffix), dpi=180)
        plt.close(fig)


def run_scenario(scenario, train_paths, test_paths, settings, folder, reader=read_raw):
    scenario = validate_scenario(scenario)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _simple_json(folder/"settings.json", dict(scenario=scenario, preprocessing=settings, matlab_signals=MAT_SIGNALS))
    db_path = folder/"model.sqlite"
    connection = open_training_db(db_path, scenario, settings)
    begun = time.perf_counter()
    cache = {}
    try:
        for q, path in enumerate(train_paths, 1):
            if log_is_complete(connection, str(path)):
                logging.info("[%s] train %d/%d already saved: %s", scenario["name"], q, len(train_paths), Path(path).name)
                continue
            tick = time.perf_counter()
            raw = None
            try:
                raw = reader(path, scenario, settings)
                report = train_one_log(connection, str(path), raw, scenario, settings, state_cache=cache)
                logging.info("[%s] train %d/%d %s: %s transitions, %.1fs", scenario["name"], q, len(train_paths), Path(path).name, report["valid_transitions"], time.perf_counter()-tick)
            except (KeyboardInterrupt, InputChangedError):
                raise
            except Exception as exc:
                record_log_error(connection, str(path), exc)
                logging.warning("[%s] skipped %s: %s", scenario["name"], Path(path).name, exc)
            finally:
                raw = None
            gc.collect()
        training = finalize_model(connection, scenario["n"])
        _simple_json(folder/"training_summary.json", training)
        _export_training_progress(connection, folder, train_paths)
        if not training["raw_events"]:
            raise ValueError("No valid training transitions; see training_logs.csv")
    finally:
        connection.close()
    logging.info("[%s] evaluating %d held-out logs; structural analysis and plots", scenario["name"], len(test_paths))
    evaluated = evaluate_scenario(db_path, test_paths, scenario, settings, reader, folder)
    summary = dict(evaluated)
    status = "complete" if summary.get("test_events", 0) else "trained_without_valid_test"
    summary.update(name=scenario["name"], dt=scenario["dt"], n=scenario["n"],
        driver_signals=",".join(scenario["driver_signals"]),
        train_files=len(train_paths), test_files=len(test_paths),
        elapsed_seconds=time.perf_counter()-begun, status=status)
    _simple_json(folder/"scenario_summary.json", summary)
    logging.info("[%s] COMPLETE: %s", scenario["name"], folder)
    if status != "complete":
        logging.warning("[%s] Model trained, but no usable held-out transitions; inspect test_log_audit.csv",scenario["name"])
    return summary


def _flatten_summary(summary):
    flat = {}
    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, part in value.items():
                walk(part, prefix+"."+key if prefix else str(key))
        elif not isinstance(value, (list, tuple)):
            flat[prefix] = value
    walk(summary)
    return flat


def check_sample(path, output):
    """Software smoke check on separated excerpts of ONE log, not a holdout study."""
    output = Path(output)
    _configure_logging(output)
    settings = _experiment_settings(RANDOM_SEED, fast=True)
    summaries = []
    for scenario in SCENARIOS:
        scenario = validate_scenario(scenario)
        logging.info("Sample check: %s", scenario["name"])
        raw = read_raw(path, scenario, settings)
        left = max(float(a[0, 0]) for a in raw.values())
        right = min(float(a[-1, 0]) for a in raw.values())
        if right-left < 70:
            raise ValueError("Sample needs at least 70 seconds of common signals")
        # Keep surrounding source records for interpolation, but don't use the
        # same records or adjacent endpoints in train and test windows.
        train_end = left+min(120., (right-left)*.35)
        test_begin = train_end+max(5., settings["max_gap"]*3)
        test_end = min(right, test_begin+120.)
        def window(begin, end):
            return {name: a[(a[:, 0]>=begin)&(a[:, 0]<=end)].copy() for name, a in raw.items()}
        train_raw, test_raw = window(left, train_end), window(test_begin, test_end)
        del raw
        source_path = Path(path).resolve()
        train_id = str(source_path.with_name(source_path.stem+"__software_check_train_excerpt.mat"))
        test_id = str(source_path.with_name(source_path.stem+"__software_check_test_excerpt.mat"))
        def reader(identity, scenario, settings):
            return train_raw if str(identity)==train_id else test_raw
        summary = run_scenario(scenario, [train_id], [test_id], settings, output/scenario["name"], reader)
        summary["evidence"] = "Software check on separate excerpts of one MAT file; not independent-trip validation."
        summaries.append(summary)
    _simple_json(output/"sample_check.json", dict(status="PASS", source=str(path), scenarios=summaries,
        evidence="One-log software check; scores must not be presented as independent operational results."))
    _simple_csv(output/"comparison.csv", [_flatten_summary(x) for x in summaries])
    print("Sample software check completed:", output)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the three configured Markov driver scenarios in sequence.")
    parser.add_argument("--data", nargs="+", type=Path, help="Optional override: MAT files or folders; otherwise use DATA_DIRS at the top.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--check-sample", type=Path, help="Software check on excerpts of one local MAT; no network access or operational validation claim.")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.check_sample:
        return check_sample(args.check_sample, args.output/("sample_check_"+datetime.now().strftime("%Y%m%d_%H%M%S")))
    _configure_logging(args.output)
    files = discover_mat_files(args.data or DATA_DIRS)
    if not files:
        raise SystemExit("No MAT files found. Pass --data FOLDER or set DATA_DIRS at the top.")
    snapshots = {}
    for path in files:
        try:
            state = path.stat()
            snapshots[path] = (state.st_size,state.st_mtime_ns)
        except OSError as exc:
            logging.warning("Cannot inspect input; excluded before splitting: %s: %s",path,exc)
    files = [path for path in files if path in snapshots]
    train_paths, test_paths = make_file_split(files, TEST_SHARE, RANDOM_SEED)
    scenarios = [validate_scenario(s) for s in SCENARIOS]
    settings = _experiment_settings(RANDOM_SEED)
    train_set = set(train_paths)
    manifest = [dict(path=str(p), group_id=p.stem.casefold(), split="train" if p in train_set else "test",
                     bytes=snapshots[p][0], mtime_ns=snapshots[p][1]) for p in files]
    frozen_reader = freeze_input_reader(manifest)
    program_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = dict(scenarios=scenarios, settings=settings, source_names=MAT_SIGNALS,
                    program_sha256=program_sha256, manifest=manifest)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    run_dir = args.output/("run_"+signature)
    _configure_logging(run_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir/"_matplotlib_cache"))
    _simple_json(run_dir/"run_settings.json", identity)
    _simple_csv(run_dir/"split_manifest.csv", manifest)
    logging.info("%d MAT files: train=%d, held-out=%d. All scenarios use the same split.", len(files), len(train_paths), len(test_paths))
    logging.info("Output %s. Completed training logs are reused if this same run is restarted.", run_dir)
    summaries = []
    for scenario in scenarios:
        try:
            result = run_scenario(scenario, train_paths, test_paths, settings, run_dir/scenario["name"], frozen_reader)
        except KeyboardInterrupt:
            logging.warning("Interrupted. Already committed training logs are safe. Run the same command again to continue.")
            raise SystemExit(130)
        except Exception as exc:
            logging.exception("Scenario failed: %s", scenario["name"])
            result = dict(name=scenario["name"], status="failed", error=str(exc))
        summaries.append(result)
        _simple_json(run_dir/"comparison.json", summaries)
        _simple_csv(run_dir/"comparison.csv", [_flatten_summary(x) for x in summaries])
        gc.collect()
    _simple_json(args.output/"last_run.json", dict(output=str(run_dir), scenarios=[dict(name=x["name"], status=x["status"]) for x in summaries]))
    logging.info("ALL SCENARIOS FINISHED. Comparison: %s", run_dir/"comparison.csv")
    if any(r["status"]!="complete" for r in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
