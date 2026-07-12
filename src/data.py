"""
DATA LAYER  —  Reality -> Data

Turns raw NASA C-MAPSS FD001 text files into a clean table with a target
we can actually learn from.

Three variable families are kept deliberately separate:
  1. OPERATING CONDITIONS (op_setting_1..3) — context. How hard the engine is
     being pushed. NOT health. A hot day is not a broken engine.
  2. SENSORS (sensor_1..21) — the observable state of the machine.
     Several of these are flat and carry zero signal (Stage 2 proves which).
  3. TARGET (RUL) — Remaining Useful Life, in cycles.

TURBOFAN-SPECIFIC (swap this out for the DTC version):
  - the column names / file format
  - the RUL cap of 125
Everything else (unit_id + time + features + target) is the generic shape of
any "entity degrading over time" problem: engine -> customer, cycle -> week.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema. The raw files are space-delimited with no header.
# ---------------------------------------------------------------------------
INDEX_COLS = ["unit", "cycle"]
OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + OP_COLS + SENSOR_COLS

# Locked decision: piecewise-linear RUL.
# An engine that has 300 cycles left is not "more healthy" than one with 200.
# Both are simply fine. Degradation only becomes visible near the end of life,
# so we flatten the target above this cap and let the model spend its capacity
# learning the part that actually matters: the decline.
RUL_CAP = 125

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _read_raw(path: Path) -> pd.DataFrame:
    """Read one space-delimited C-MAPSS file. Trailing whitespace -> 2 junk cols."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, : len(ALL_COLS)]
    df.columns = ALL_COLS
    return df


def load_train(cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Training engines run to failure. So for any row, the true RUL is simply
    "how many cycles until this engine's last recorded cycle".
    """
    df = _read_raw(DATA_DIR / "train_FD001.txt")
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = last_cycle - df["cycle"]          # raw, uncapped
    df["rul_capped"] = df["rul"].clip(upper=cap)  # the actual training target
    return df


def load_test(cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Test engines are TRUNCATED — they stop mid-life. The number of cycles each
    one had left after its final recorded reading lives in RUL_FD001.txt
    (one number per engine, in unit order). So:

        rul(row) = (last_recorded_cycle - this_cycle) + true_remaining_after_end
    """
    df = _read_raw(DATA_DIR / "test_FD001.txt")
    truth = pd.read_csv(DATA_DIR / "RUL_FD001.txt", sep=r"\s+", header=None)
    truth.columns = ["rul_at_end"]
    truth["unit"] = np.arange(1, len(truth) + 1)

    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.merge(truth, on="unit", how="left")
    df["rul"] = (last_cycle - df["cycle"]) + df["rul_at_end"]
    df["rul_capped"] = df["rul"].clip(upper=cap)
    return df.drop(columns=["rul_at_end"])


def last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """The scoring view: one row per engine — its most recent reading.

    This is what a real maintenance system sees: 'here is each engine as of
    today'. Test-set accuracy is judged here, not on every historical row.
    """
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def split_by_unit(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """
    Locked decision: split by ENGINE UNIT, never by row.

    Rows from the same engine are almost identical to their neighbours. Splitting
    by row would put cycle 100 in train and cycle 101 in validation — the model
    would look brilliant and be worthless. Whole engines go one side or the other.
    """
    units = np.sort(df["unit"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = int(round(len(units) * val_frac))
    val_units = set(shuffled[:n_val])
    train_units = set(shuffled[n_val:])
    return (
        df[df["unit"].isin(train_units)].reset_index(drop=True),
        df[df["unit"].isin(val_units)].reset_index(drop=True),
    )


if __name__ == "__main__":
    train, test = load_train(), load_test()
    print(f"train rows={train.shape[0]:,} cols={train.shape[1]}  engines={train.unit.nunique()}")
    print(f"test  rows={test.shape[0]:,} cols={test.shape[1]}  engines={test.unit.nunique()}")
    print("\nrun-to-failure check (min RUL per training engine should be 0):")
    print("  unique min-RUL values:", sorted(train.groupby("unit")["rul"].min().unique()))
    print(train[["unit", "cycle", "sensor_2", "sensor_11", "rul", "rul_capped"]].head())
