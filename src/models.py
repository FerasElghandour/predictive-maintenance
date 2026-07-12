"""
BOX 3 — PREDICTION LAYER

Job of this box:  12 sensors  ->  "how many cycles does this engine have left?"

Three models, in deliberate order. The order IS the lesson:

  1. LINEAR        — the dumbest thing that could work. Raw sensors, straight line.
                     This SETS THE BAR. If a fancy model can't beat this, the fancy
                     model is decoration.
  1b. LINEAR + features — same dumb model, better inputs. Isolates how much of any
                     improvement comes from FEATURE WORK vs from MODEL COMPLEXITY.
                     Most people never run this and so never learn which one paid.
  2. RANDOM FOREST — many decision trees voting. Handles curves and interactions,
                     and tells us which features it actually leaned on.
  3. LSTM          — a neural net that reads a SEQUENCE of 30 cycles rather than a
                     single snapshot. It can see the shape of the decline, not just
                     the current position.

HOW WE SCORE (the part people get wrong):
  Train on 100 engines we watched die.
  Predict on 100 DIFFERENT engines, never seen, frozen mid-life.
  Compare to NASA's true answer key (RUL_FD001.txt).
  ONE prediction per engine — its most recent reading. That is exactly what a real
  maintenance system sees: "here is each engine, as of today."

  MAE = on average, how many cycles are we off by. Lower is better. This is the
        number that goes in the README.

RUN:  python models.py snapshot   (linear + forest)
      python models.py lstm       (neural net; resumable — safe to run repeatedly)
      python models.py report     (compare everything, pick winner, save outputs)
      python models.py all
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import load_train, load_test, last_cycle_per_unit, split_by_unit, RUL_CAP
from features import build_features, feature_columns, baseline_columns, USEFUL_SENSORS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
ARTIFACTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

SEQ_LEN = 30        # the LSTM reads the last 30 cycles
MAX_EPOCHS = 30
EPOCHS_PER_RUN = 6  # train in chunks so a long job can't be killed mid-way


def score(y_true, y_pred, name):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, RUL_CAP)
    err = y_pred - y_true          # negative = predicted EARLY (cautious/safe)
    return dict(
        model=name,
        MAE=round(mean_absolute_error(y_true, y_pred), 2),
        RMSE=round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        pct_early=round(float((err <= 0).mean() * 100), 1),
        pct_late=round(float((err > 0).mean() * 100), 1),
        worst_late_call=round(float(err.max()), 1),   # our most dangerous miss
    )


def _truth():
    test = load_test()
    snap = last_cycle_per_unit(test).sort_values("unit")
    return test, snap, snap["rul_capped"].to_numpy()


# ---------------------------------------------------------------------------
# 1 + 2 — snapshot models
# ---------------------------------------------------------------------------
def run_snapshot():
    train = load_train()
    test, snap, y = _truth()

    Xtr = build_features(train)
    Xte = build_features(test)
    ytr = train["rul_capped"].to_numpy()
    te_idx = test.groupby("unit")["cycle"].idxmax()      # last reading per engine
    te_idx = test.loc[te_idx].sort_values("unit").index  # keep unit order 1..100

    results, preds = [], {}
    fcols, bcols = feature_columns(), baseline_columns()

    lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr[bcols], ytr)
    preds["linear"] = lin.predict(Xte.loc[te_idx, bcols])
    results.append(score(y, preds["linear"], "1. Linear (baseline)"))

    lin2 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr[fcols], ytr)
    preds["linear_feat"] = lin2.predict(Xte.loc[te_idx, fcols])
    results.append(score(y, preds["linear_feat"], "1b. Linear + features"))

    rf = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, max_features=0.4,
        n_jobs=-1, random_state=42,
    ).fit(Xtr[fcols], ytr)
    preds["rf"] = rf.predict(Xte.loc[te_idx, fcols])
    results.append(score(y, preds["rf"], "2. RandomForest"))

    imp = (pd.DataFrame({"feature": fcols, "importance": rf.feature_importances_})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)

    joblib.dump({"model": rf, "features": fcols}, ARTIFACTS / "rf_model.joblib")
    np.savez(ARTIFACTS / "preds_snapshot.npz", **{k: np.clip(v, 0, RUL_CAP) for k, v in preds.items()})
    json.dump(results, open(ARTIFACTS / "results_snapshot.json", "w"), indent=2)

    print(pd.DataFrame(results).to_string(index=False))
    print("\nTop features the forest leaned on:")
    print(imp.head(8).to_string(index=False))
    return results


# ---------------------------------------------------------------------------
# 3 — LSTM
# ---------------------------------------------------------------------------
def make_sequences(df, sensors=USEFUL_SENSORS, seq_len=SEQ_LEN, last_only=False):
    """One sample = the last `seq_len` cycles of one engine (a 2-D window)."""
    Xs, ys, units = [], [], []
    for u, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        arr = g[sensors].to_numpy(dtype=np.float32)
        rul = g["rul_capped"].to_numpy(dtype=np.float32)
        n = len(g)
        if n < seq_len:  # short engine: left-pad with its first reading
            arr = np.vstack([np.repeat(arr[:1], seq_len - n, axis=0), arr])
            rul = np.concatenate([np.repeat(rul[:1], seq_len - n), rul])
            n = seq_len
        rows = [n - 1] if last_only else range(seq_len - 1, n)
        for i in rows:
            Xs.append(arr[i - seq_len + 1: i + 1])
            ys.append(rul[i])
            units.append(u)
    return np.asarray(Xs), np.asarray(ys), np.asarray(units)


def run_lstm():
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    torch.manual_seed(42)

    train = load_train()
    test, snap, y = _truth()
    S = USEFUL_SENSORS

    mu = train[S].mean().to_numpy(np.float32)
    sd = train[S].std().replace(0, 1).to_numpy(np.float32)

    tr_in, val_in = split_by_unit(train, val_frac=0.15, seed=7)   # split by ENGINE
    Xtr, ytr, _ = make_sequences(tr_in)
    Xva, yva, _ = make_sequences(val_in)
    Xte, _, ute = make_sequences(test, last_only=True)
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    class Net(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.lstm = nn.LSTM(n_in, 64, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1]).squeeze(-1)

    net = Net(len(S))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.L1Loss()

    ck = ARTIFACTS / "lstm_ckpt.pt"
    state = {"epoch": 0, "best": 1e9, "patience": 0}
    if ck.exists():   # resume — long trainings shouldn't have to start over
        blob = torch.load(ck, weights_only=False)
        net.load_state_dict(blob["net"]); opt.load_state_dict(blob["opt"])
        state = blob["state"]
        print(f"  resuming from epoch {state['epoch']} (best val MAE {state['best']:.2f})")

    vX, vY = torch.tensor(Xva), torch.tensor(yva)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=256, shuffle=True,
    )

    done = False
    for _ in range(EPOCHS_PER_RUN):
        if state["epoch"] >= MAX_EPOCHS or state["patience"] >= 5:
            done = True
            break
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            lossf(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(vX), vY).item()
        state["epoch"] += 1
        if vl < state["best"] - 0.1:
            state["best"], state["patience"] = vl, 0
            torch.save(net.state_dict(), ARTIFACTS / "lstm_best.pt")
        else:
            state["patience"] += 1
        print(f"  epoch {state['epoch']:2d}  val MAE {vl:6.2f}  (best {state['best']:.2f})")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "state": state}, ck)

    if state["epoch"] >= MAX_EPOCHS or state["patience"] >= 5:
        done = True

    if not done:
        print(f"  -> paused at epoch {state['epoch']}/{MAX_EPOCHS}. Run again to continue.")
        return None

    net.load_state_dict(torch.load(ARTIFACTS / "lstm_best.pt", weights_only=True))
    net.eval()
    with torch.no_grad():
        p = net(torch.tensor(Xte)).numpy()
    p = np.clip(p[np.argsort(ute)], 0, RUL_CAP)     # align to unit order 1..100

    res = score(y, p, "3. LSTM")
    np.savez(ARTIFACTS / "preds_lstm.npz", lstm=p)
    json.dump([res], open(ARTIFACTS / "results_lstm.json", "w"), indent=2)
    print("\n" + pd.DataFrame([res]).to_string(index=False))
    print("  LSTM training COMPLETE.")
    return res


# ---------------------------------------------------------------------------
# report — compare, pick a winner, save what the dashboard needs
# ---------------------------------------------------------------------------
def run_report():
    test, snap, y = _truth()
    results = json.load(open(ARTIFACTS / "results_snapshot.json"))
    preds = dict(np.load(ARTIFACTS / "preds_snapshot.npz"))

    if (ARTIFACTS / "results_lstm.json").exists():
        results += json.load(open(ARTIFACTS / "results_lstm.json"))
        preds.update(dict(np.load(ARTIFACTS / "preds_lstm.npz")))
    else:
        print("NOTE: LSTM not finished — excluded from the comparison rather than faked.")

    table = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    table.to_csv(REPORTS / "model_comparison.csv", index=False)

    key = {"1. Linear (baseline)": "linear", "1b. Linear + features": "linear_feat",
           "2. RandomForest": "rf", "3. LSTM": "lstm"}
    winner = table.iloc[0]["model"]
    wkey = key[winner]

    fleet = snap[["unit", "cycle"]].copy()
    fleet["true_rul"] = y
    for k, v in preds.items():
        fleet[f"pred_{k}"] = np.round(v, 1)
    fleet["pred_rul"] = np.round(preds[wkey], 1)
    fleet.to_csv(ARTIFACTS / "fleet_predictions.csv", index=False)

    json.dump({"winner": winner, "winner_key": wkey, "rul_cap": RUL_CAP,
               "results": results}, open(ARTIFACTS / "metrics.json", "w"), indent=2)

    print("=" * 82)
    print("SCORED ON 100 UNSEEN TEST ENGINES  vs  NASA's TRUE ANSWER KEY")
    print("=" * 82)
    print(table.to_string(index=False))
    print(f"\nWINNER: {winner}   MAE = {table.iloc[0]['MAE']} cycles")
    return table


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("snapshot", "all"):
        print("--- Snapshot models (Linear, Linear+features, RandomForest) ---")
        run_snapshot()
    if cmd in ("lstm", "all"):
        print("\n--- LSTM ---")
        run_lstm()
    if cmd in ("report", "all"):
        print()
        run_report()
