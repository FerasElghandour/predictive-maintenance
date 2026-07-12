"""
UNDERSTANDING LAYER  —  Data -> Understanding

Before modelling anything, look at the machine. Two questions:

  Q1. Which sensors actually MOVE as an engine ages, and which are dead weight?
  Q2. Do they move in a consistent direction across different engines?
      (A sensor that drifts up on engine 3 and down on engine 47 is noise,
       not a health signal.)

The output is a SENSOR TRIAGE: keep / drop. This is the step people skip, and
it is the reason their model is 21 columns of confusion instead of 14 columns
of signal.

TURBOFAN-SPECIFIC: the sensor names. The METHOD (does this variable trend with
the outcome, consistently, across entities?) transfers to any domain.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_train, SENSOR_COLS, OP_COLS

REPORTS = Path(__file__).resolve().parents[1] / "reports"
REPORTS.mkdir(exist_ok=True)


def triage_sensors(train: pd.DataFrame) -> pd.DataFrame:
    """
    Score every sensor on two things:

      variability  — does it move at all? (std across the whole dataset)
      trend        — does it move WITH degradation, the same way, on every engine?
                     Measured as the correlation between the sensor and RUL,
                     computed per engine, then averaged. Consistency is the point:
                     we also report how many engines agree on the direction.
    """
    rows = []
    for s in SENSOR_COLS:
        std = train[s].std()

        # per-engine correlation with true (uncapped) RUL
        cors = (
            train.groupby("unit")[[s, "rul"]]
            .apply(lambda g: g[s].corr(g["rul"]))
            .dropna()
        )
        mean_cor = cors.mean() if len(cors) else np.nan
        # what fraction of engines agree on the sign of the trend?
        agreement = (np.sign(cors) == np.sign(mean_cor)).mean() if len(cors) else np.nan

        rows.append(
            dict(
                sensor=s,
                std=std,
                mean_corr_with_rul=mean_cor,
                abs_corr=abs(mean_cor) if pd.notna(mean_cor) else 0.0,
                direction_agreement=agreement,
            )
        )

    t = pd.DataFrame(rows)
    # A sensor is USEFUL if it moves (std > 0) and its trend is both strong
    # and consistent across engines.
    t["useful"] = (t["std"] > 1e-8) & (t["abs_corr"] > 0.30) & (t["direction_agreement"] > 0.90)
    t["verdict"] = np.where(
        t["std"] <= 1e-8, "FLAT — constant, zero information",
        np.where(t["useful"], "KEEP — drifts consistently with age", "DROP — moves, but no reliable trend"),
    )
    return t.sort_values("abs_corr", ascending=False).reset_index(drop=True)


def plot_sensor_lifetimes(train, sensors, units, fname, title):
    """Plot chosen sensors across the full life of a few engines.
    X axis = RUL counting DOWN to zero, so all engines line up at failure."""
    n = len(sensors)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, s in zip(axes, sensors):
        for u in units:
            g = train[train.unit == u]
            ax.plot(g["rul"], g[s], lw=0.9, alpha=0.85, label=f"engine {u}")
        ax.invert_xaxis()  # life runs left->right toward failure at RUL=0
        ax.axvline(125, color="grey", ls="--", lw=0.8)
        ax.set_title(s, fontsize=10)
        ax.grid(alpha=0.2)
    for ax in axes[n:]:
        ax.set_visible(False)

    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(title + "\n(x-axis = cycles remaining, counting DOWN to failure at 0. "
                 "Dashed line = the 125-cycle cap.)", fontsize=12)
    fig.supxlabel("Remaining Useful Life (cycles)")
    fig.tight_layout()
    out = REPORTS / fname
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_op_settings(train, units, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    for ax, c in zip(axes, OP_COLS):
        for u in units:
            g = train[train.unit == u]
            ax.plot(g["rul"], g[c], lw=0.8, alpha=0.85)
        ax.invert_xaxis()
        ax.set_title(c, fontsize=10)
        ax.grid(alpha=0.2)
    fig.suptitle("Operating conditions — FD001 is a SINGLE operating condition. "
                 "These are flat noise, not health.", fontsize=12)
    fig.supxlabel("Remaining Useful Life (cycles)")
    fig.tight_layout()
    out = REPORTS / fname
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    train = load_train()
    t = triage_sensors(train)
    t.to_csv(REPORTS / "sensor_triage.csv", index=False)

    keep = t[t.useful].sensor.tolist()
    flat = t[t.verdict.str.startswith("FLAT")].sensor.tolist()
    drop = t[t.verdict.str.startswith("DROP")].sensor.tolist()

    units = [1, 24, 69, 100]
    plot_sensor_lifetimes(train, keep[:9], units, "sensors_that_drift.png",
                          "SIGNAL: sensors that drift consistently as the engine ages")
    plot_sensor_lifetimes(train, (flat + drop)[:9], units, "sensors_that_are_flat.png",
                          "NOISE: sensors that are flat or carry no reliable trend")
    plot_op_settings(train, units, "operating_conditions.png")

    print(t.to_string(index=False))
    print(f"\nKEEP ({len(keep)}): {keep}")
    print(f"FLAT ({len(flat)}): {flat}")
    print(f"DROP ({len(drop)}): {drop}")
