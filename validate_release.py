"""Engine-disjoint development, validation-selected policy, and censored replay.

This is a corrected protocol on a previously explored public benchmark, not
a claim that FD001 has become a new untouched external validation cohort.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data import load_train, load_test, split_by_unit, RUL_CAP
from features import build_features, feature_columns


def replay(frame, threshold, lead_time=20):
    """Count observed eligibility separately from right-censored trajectories."""
    counts = dict(engines=0, eligible_observed=0, censored_before_danger=0,
                  warned=0, good=0, late=0, early=0, missed_observed=0)
    lead = []
    for _, g in frame.groupby("unit"):
        g = g.sort_values("cycle")
        counts["engines"] += 1
        eligible = bool((g.rul <= threshold * RUL_CAP).any())
        counts["eligible_observed" if eligible else "censored_before_danger"] += 1
        fired = g[g.pred_rul < threshold * RUL_CAP]
        if fired.empty:
            counts["missed_observed"] += int(eligible)
            continue
        counts["warned"] += 1
        remaining = float(fired.rul.iloc[0]); lead.append(remaining)
        counts["late" if remaining < lead_time else "early" if remaining > threshold * RUL_CAP else "good"] += 1
    return {**counts, "threshold": float(threshold), "lead_time": lead_time,
            "median_warning_cycles": float(np.median(lead)) if lead else None,
            "miss_rate_observed_eligible": counts["missed_observed"] / counts["eligible_observed"] if counts["eligible_observed"] else None}


def snapshot_mae(model, frame):
    ix = frame.groupby("unit").cycle.idxmax()
    f = build_features(frame)
    pred = np.clip(model.predict(f.loc[ix, feature_columns()]), 0, RUL_CAP)
    return float(mean_absolute_error(frame.loc[ix, "rul_capped"], pred))


def main():
    train, val = split_by_unit(load_train(), val_frac=.2, seed=42)
    assert set(train.unit).isdisjoint(val.unit)
    x = build_features(train)[feature_columns()]
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                                max_features=.4, n_jobs=2, random_state=42),
    }
    validation = {}
    for name, model in models.items():
        model.fit(x, train.rul_capped)
        validation[name] = snapshot_mae(model, val)
    winner = min(validation, key=validation.get)
    model = models[winner]
    val = val.copy()
    val["pred_rul"] = np.clip(model.predict(build_features(val)[feature_columns()]), 0, RUL_CAP)
    sweep = [replay(val, t) for t in [.30, .35, .40, .45]]
    # Predefined lexicographic objective: avoid late/missed observed calls,
    # then reduce early interventions; final tie-break chooses lower threshold.
    selected = min(sweep, key=lambda r: (r["late"] + r["missed_observed"], r["early"], r["threshold"]))
    test = load_test()
    test_mae = snapshot_mae(model, test)
    test["pred_rul"] = np.clip(model.predict(build_features(test)[feature_columns()]), 0, RUL_CAP)
    import joblib
    artifact = ROOT / "artifacts/trained_model.joblib"
    joblib.dump({"model":model,"features":feature_columns(),"rul_cap":RUL_CAP,"threshold":selected["threshold"]},artifact)
    restored=joblib.load(artifact)
    restored_predictions=np.clip(restored["model"].predict(build_features(test)[restored["features"]]),0,RUL_CAP)
    if not np.array_equal(restored_predictions,test["pred_rul"].to_numpy()):raise AssertionError("Checkpoint reload changed predictions")
    test[["unit","cycle","pred_rul"]].to_csv(ROOT/"artifacts/test_predictions.csv",index=False)
    report = {
        "checkpoint_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "checkpoint_reload_exact":True,
        "protocol": "engine-disjoint development and validation; historical public test cohort; observed-window policy replay",
        "train_engines": sorted(int(n) for n in train.unit.unique()),
        "validation_engines": sorted(int(n) for n in val.unit.unique()),
        "validation_model_mae": validation, "selected_model": winner,
        "validation_policy_sweep": sweep, "frozen_threshold": selected["threshold"],
        "test_snapshot_mae": test_mae, "test_replay": replay(test, selected["threshold"]),
        "limits": ["Test trajectories stop before failure.", "Public FD001 was inspected in earlier development.",
                   "Observed-window rates do not establish eventual failure coverage or certified safety."],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT/"src/features.py", ROOT/"src/data.py"]},
        "input_sha256": {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"data/raw").glob("*.txt"))},
    }
    target = ROOT / "artifacts/release_validation.json"
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ["selected_model","frozen_threshold","test_snapshot_mae","test_replay"]},indent=2))

if __name__ == "__main__":
    main()
