"""
FEATURE BUILDER  (still part of Box 2: Understanding)

A single sensor reading tells you almost nothing. sensor_11 = 47.6 — is that bad?
You cannot answer without context. Three kinds of context matter:

  1. WHERE IS IT NOW      -> the raw reading (and a smoothed version of it)
  2. HOW FAR HAS IT MOVED -> the reading minus THIS engine's own healthy baseline.
                             This is the literal definition of degradation: drift
                             from where this specific machine started.
  3. HOW FAST IS IT MOVING-> the slope over a recent window. An engine sliding
                             quickly is in more trouble than one sliding slowly.

Feature #2 is the important one, and it is why we can compare engines that were
built slightly differently: we never compare engine A to engine B, we compare
engine A to *its own younger self*.

TRANSFERS DIRECTLY to churn: a customer ordering twice a month is fine — unless
they used to order eight times a month. Level means nothing. Drift means everything.
"""

import numpy as np
import pandas as pd

from data import SENSOR_COLS

# The 12 survivors from the Stage-2 triage (src/understand.py).
# Hard-coded on purpose: this is a decision we made and can defend, not a
# number that silently changes every time we rerun the code.
USEFUL_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_11",
    "sensor_12", "sensor_13", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

BASELINE_CYCLES = 5   # an engine's first 5 cycles = "healthy, factory-fresh"
WINDOW = 20           # look back 20 cycles for smoothing + slope


def _rolling_slope(y: np.ndarray, w: int = WINDOW) -> np.ndarray:
    """
    Least-squares slope over a trailing window of length w, for every position.

    Vectorised. The naive version (polyfit inside a rolling.apply) is ~1000x
    slower and made the whole pipeline time out — worth the extra 10 lines.

    Least-squares slope with x = 0,1,...,w-1 is a fixed dot product:
        slope = sum(k_i * y_i)   where   k = (x - mean(x)) / sum((x - mean(x))^2)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.zeros(n)
    if n < 2:
        return out

    # positions with a full window: one dot product each
    if n >= w:
        x = np.arange(w, dtype=float)
        k = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
        windows = np.lib.stride_tricks.sliding_window_view(y, w)
        out[w - 1:] = windows @ k

    # the first w-1 positions don't have a full window: use what's available
    for i in range(1, min(w - 1, n - 1) + 1):
        x = np.arange(i + 1, dtype=float)
        xc = x - x.mean()
        denom = (xc ** 2).sum()
        out[i] = (xc @ (y[: i + 1] - y[: i + 1].mean())) / denom if denom else 0.0
    return out


def build_features(df: pd.DataFrame, sensors=USEFUL_SENSORS) -> pd.DataFrame:
    """
    Row-wise features. Everything is computed using ONLY the past
    (expanding/rolling backwards) — never the future. A model that peeks at
    future cycles will score brilliantly and be useless in production.
    """
    out = df[["unit", "cycle"]].copy()
    out["cycle"] = df["cycle"]

    g = df.groupby("unit")

    for s in sensors:
        # 1. where is it now (raw + smoothed)
        out[f"{s}"] = df[s]
        out[f"{s}_roll_mean"] = g[s].transform(
            lambda x: x.rolling(WINDOW, min_periods=1).mean()
        )
        out[f"{s}_roll_std"] = g[s].transform(
            lambda x: x.rolling(WINDOW, min_periods=1).std().fillna(0.0)
        )

        # 2. how far has it drifted from THIS engine's own healthy baseline
        baseline = g[s].transform(lambda x: x.head(BASELINE_CYCLES).mean())
        out[f"{s}_drift"] = df[s] - baseline

        # 3. how fast is it moving
        out[f"{s}_slope"] = g[s].transform(lambda x: pd.Series(_rolling_slope(x.to_numpy()), index=x.index))

    return out


def feature_columns(sensors=USEFUL_SENSORS):
    cols = ["cycle"]
    for s in sensors:
        cols += [s, f"{s}_roll_mean", f"{s}_roll_std", f"{s}_drift", f"{s}_slope"]
    return cols


# The crude baseline feature set: just the raw sensors + how long it's been running.
# No cleverness at all. This is the bar the fancy models must clear.
def baseline_columns(sensors=USEFUL_SENSORS):
    return ["cycle"] + list(sensors)
