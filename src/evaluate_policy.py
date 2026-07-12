"""
POLICY EVALUATION — the honest test.

Box 3 scored the MODEL: "given this engine today, how close is the guess?"
That is the easy test, and it is the one everybody reports.

This file scores the SYSTEM: "if we had actually run this rule, every day, on
every engine — would a human have had enough warning to act?"

Method: replay each test engine cycle by cycle. Run the model at every cycle.
Fire the rule the FIRST time predicted health drops below the threshold. Then
look up how much TRUE life the engine had left at that exact moment.

Four outcomes per engine:
  TOO LATE     — we fired, but with less than LEAD_TIME cycles of runway.
                 The crew could not have acted. This is the outcome that hurts.
  NEVER WARNED — the engine entered the danger zone and we never fired at all.
  FALSE ALARM  — we fired while the engine still had more life than the rule
                 intended to catch. We waste good engine life.
  GOOD CALL    — we fired inside the intended window, with enough runway.

These are the numbers that belong in the README. Not MAE.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from data import load_test, RUL_CAP
from features import build_features
from decide import MAINTENANCE_THRESHOLD, LEAD_TIME_CYCLES

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def score_timeline(test_scored: pd.DataFrame, threshold: float, lead_time: int):
    danger_rul = threshold * RUL_CAP     # what the rule is trying to catch
    n = test_scored["unit"].nunique()
    too_late = never = false_alarm = good = 0
    runways, wasted = [], []

    for _, g in test_scored.groupby("unit"):
        g = g.sort_values("cycle")
        fired = g[g["health"] < threshold]
        end_rul = g["rul"].iloc[-1]      # true life left when the record stops

        if len(fired) == 0:
            if end_rul < danger_rul:
                never += 1               # it was sick and we said nothing
            continue

        true_at_flag = float(fired["rul"].iloc[0])
        runways.append(true_at_flag)

        if true_at_flag < lead_time:
            too_late += 1
        elif true_at_flag > danger_rul:
            false_alarm += 1
            wasted.append(true_at_flag - danger_rul)
        else:
            good += 1

    return dict(
        threshold_pct=int(threshold * 100),
        lead_time_cycles=lead_time,
        fires_below_cycles=round(danger_rul, 1),
        engines_flagged=too_late + false_alarm + good,
        pct_too_late=round(100 * too_late / n, 1),
        pct_never_warned=round(100 * never / n, 1),
        pct_false_alarm=round(100 * false_alarm / n, 1),
        pct_good_call=round(100 * good / n, 1),
        median_runway_cycles=int(np.median(runways)) if runways else 0,
        median_life_wasted_cycles=int(np.median(wasted)) if wasted else 0,
    )


def main():
    blob = joblib.load(ART / "rf_model.joblib")
    rf, fcols = blob["model"], blob["features"]

    test = load_test()
    X = build_features(test)
    test["pred_rul"] = np.clip(rf.predict(X[fcols]), 0, RUL_CAP)
    test["health"] = test["pred_rul"] / RUL_CAP
    test.to_csv(ART / "test_timeline.csv", index=False)

    chosen = score_timeline(test, MAINTENANCE_THRESHOLD, LEAD_TIME_CYCLES)

    # the full sweep — this is the evidence for WHY we picked 35%
    sweep = [score_timeline(test, t, LEAD_TIME_CYCLES) for t in (0.30, 0.35, 0.40, 0.45)]
    sensitivity = {
        str(L): [score_timeline(test, t, L) for t in (0.30, 0.35, 0.40, 0.45)]
        for L in (10, 20, 30)
    }

    out = dict(chosen=chosen, sweep=sweep, lead_time_sensitivity=sensitivity)
    json.dump(out, open(ART / "policy_metrics.json", "w"), indent=2)

    print("CHOSEN POLICY —", f"{chosen['threshold_pct']}% health,",
          f"{chosen['lead_time_cycles']}-cycle lead time\n")
    for k in ("pct_too_late", "pct_never_warned", "pct_false_alarm", "pct_good_call",
              "median_runway_cycles", "median_life_wasted_cycles"):
        print(f"  {k:<28} {chosen[k]}")
    print("\nFULL SWEEP (the evidence for the threshold choice):")
    print(pd.DataFrame(sweep)[
        ["threshold_pct", "pct_too_late", "pct_never_warned",
         "pct_false_alarm", "pct_good_call", "median_runway_cycles"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
