# Predicting Jet Engine Failure Before It Happens

### 🔗 **[View the live dashboard →](https://FerasElghandour.github.io/predictive-maintenance/)**

---

**The problem.** Jet engines don't fail on a schedule. In this fleet, identical engines lasted anywhere from **128 to 362 flight cycles**. That range is the entire problem. Replace every engine early and you throw away millions in usable life. Wait too long and one fails in the air. Airlines have historically had to pick which mistake to make.

**What this system does.** It reads 12 sensors off each engine, predicts how many cycles it has left, and converts that into a decision a maintenance chief can act on today: **fix it now, watch it, or leave it alone.** Then it ranks the whole fleet by urgency on a single screen.

**The result.** Tested by replaying **100 engines the system had never seen**, cycle by cycle, against their true failure times:

| | |
|---|---|
| **0%** | of failing engines were flagged **too late to act** |
| **0%** | of failing engines were **missed entirely** |
| **10%** | false alarms — each wasting a median of just **7 cycles** of good engine life |
| **40 cycles** | median warning given before the engine hit its danger zone |

It caught **every engine that was going to fail**, and gave the crew over a month of flying time to plan the fix.

---

## The fleet, right now

| Status | Engines | Meaning |
|---|---|---|
| 🔴 **Maintenance required** | **28** | Below 35% health. Schedule now. |
| 🟡 **Monitor** | **21** | Degrading. Watch the trend. |
| 🟢 **Healthy** | **51** | No action. |

The five most urgent, with NASA's sealed answer key beside the prediction:

| Engine | Predicted cycles left | **Actually** had left | Health |
|---|---|---|---|
| 81 | 5.7 | **8** | 4.6% |
| 34 | 7.1 | **7** | 5.7% |
| 76 | 9.1 | **10** | 7.3% |
| 82 | 10.3 | **9** | 8.2% |
| 35 | 10.9 | **11** | 8.7% |

---

## How it works

The system is a chain of six components. The machine learning is **one** of them — and it's the least important.

```
Reality  ->  Data  ->  Understanding  ->  Prediction  ->  Decision  ->  Action
             │         │                  │              │             │
             │         │                  │              │             └─ ranked dashboard
             │         │                  │              └─ health % -> fix / watch / leave
             │         │                  └─ RandomForest, MAE 10.3 cycles
             │         └─ 21 sensors -> the 12 that carry real signal
             └─ 100 run-to-failure engines + a piecewise-linear target
```

**Run the whole thing with one command:**

```bash
pip install -r requirements.txt
python run.py          # data -> features -> models -> decisions -> dashboard
open docs/index.html
```

---

## Decision log — why I made each choice

**Why a dumb baseline first, before any fancy model.**
A linear regression on raw sensors was the first thing built. It scored **17.4 cycles** of average error. That number's only job is to be a bar. Without it, you have no idea whether a complicated model is earning its complexity or just wearing it. Most projects skip this step and never find out.

**Why the RandomForest beat the LSTM — and why I shipped the forest.**

| Model | Avg error (cycles) |
|---|---|
| **RandomForest** ✅ | **10.3** |
| LSTM (neural network) | 10.7 |
| Linear + engineered features | 13.1 |
| Linear on raw sensors | 17.4 |

The neural network — the most sophisticated model here, and the one that would look best in a headline — **lost**. I shipped the forest. If I'd started with the LSTM I'd have shipped it, called it AI, and never known it was worse. **The baseline is what turns that into a knowable fact instead of a guess.**

**Why feature engineering mattered more than model choice.**
Look at rows 3 and 4 above. Same linear model, same data — the only difference is *better inputs* (drift from each engine's own healthy baseline, trend direction, smoothed averages). That alone cut error from **17.4 → 13.1**. Upgrading all the way from that to a RandomForest bought **13.1 → 10.3**.

**The feature work delivered a bigger gain than the model upgrade did.** That's the norm, and it's the opposite of where most people spend their time.

The single most useful feature wasn't a raw sensor reading — it was **`sensor_11_drift`: how far this engine has moved from where *it* started.** Health isn't a value. It's a *deviation*.

**Why the safety margin lives in the decision layer, not the model.**
The model has a real flaw: it **over-estimates** remaining life 54% of the time. It leans optimistic. For a jet engine, optimistic is the dangerous direction.

I did not fix this in the model. I absorbed it in the **rule** — by firing the alarm on the *first* moment predicted health crosses the line, rather than waiting for confirmation. That single design choice converts a late-leaning model into an early-warning system.

The 35% threshold wasn't taste. It came from simulating four candidate policies against the answer key:

| Threshold | Too late | Missed | False alarms | Median warning |
|---|---|---|---|---|
| 30% | 0% | 1% | 9% | 33 cycles |
| **35%** ✅ | **0%** | **0%** | 10% | 40 cycles |
| 40% | 0% | 1% | 14% | 47 cycles |
| 45% | 0% | 2% | 15% | 54 cycles |

35% is the only setting with **both** zero missed engines and zero late calls. It rests on one assumption worth stating out loud: **that a crew needs at least 20 cycles of notice to act.** At a 30-cycle lead time, 35% starts to fray and you'd move to 40%. That number comes from the airline, not from the data.

**Why the two mistakes aren't equal — and why that's the whole design.**
> A false alarm means you service a healthy engine. **Cost: money.**
> A missed call means the engine fails in flight. **Cost: not money.**

Everything above follows from that asymmetry.

**Why the dashboard is a static file, not a live server.**
Predictions are computed in a batch. Nothing here needs a Python process running to answer a request. So the dashboard is one self-contained HTML file: it loads instantly, never sleeps, costs nothing, and can't fall over. Matching the architecture to the actual requirement — instead of to what looks impressive — *is* the engineering.

---

## Honest limitations

- **This is a benchmark dataset, not a novel problem.** NASA's C-MAPSS FD001 is one of the most-studied datasets in machine learning. I chose it deliberately: the answers are known, so the *method* can be trusted when pointed at data where they aren't. This repo is a demonstration of **end-to-end system architecture**, not a research contribution.
- **FD001 is the easy variant** — one operating condition, one fault mode. Real fleets have many of both.
- **There is no feedback loop.** A real system would learn from whether its calls were right. A static dataset has nothing to feed back. This is a genuine gap, not an oversight.
- **Three sensors I discarded are probably recoverable.** Sensors 9 and 14 look like noise raw, because each engine sits at its own baseline — but normalised per-engine they'd likely carry signal. I left the improvement on the table to ship the loop end-to-end first.

---

## Why this transfers

The six boxes are the reusable part. Swap the nouns and the same pipeline is a churn model:

| Turbofan | DTC / consumer brand |
|---|---|
| engine | customer |
| flight cycle | week |
| 21 sensors | opens, sessions, orders, returns |
| cycles until failure | weeks until churn |
| "fix this engine" | "send this customer a win-back offer" |

Same architecture. Same failure modes. Same triage of *"which of these signals is actually telling me something?"*

**With one crucial flip: the cost asymmetry inverts.** Here, a false alarm costs money and a miss costs lives — so we lean cautious and happily eat 10% false alarms. In churn, a false alarm means **discounting a customer who was never going to leave** — you burn margin on someone who'd have paid full price. False alarms get *expensive*, and the same threshold sweep lands somewhere much stricter.

**The reusable asset is the method, not the number 35%.**

---

## Repo structure

```
├── data/raw/               NASA C-MAPSS FD001 (train, test, sealed answer key)
├── src/
│   ├── data.py             Box 1 — load, piecewise RUL target, leakage-safe split
│   ├── understand.py       Box 2 — sensor triage (21 -> 12)
│   ├── features.py         Box 2 — drift from baseline, rolling stats, slope
│   ├── models.py           Box 3 — Linear -> RandomForest -> LSTM
│   ├── decide.py           Box 4 — RUL -> health % -> action
│   ├── evaluate_policy.py  Box 4 — the honest test: replay the policy cycle by cycle
│   └── dashboard.py        Box 5 — build the static dashboard
├── reports/                sensor triage, model comparison, feature importance, plots
├── docs/index.html         the deployed dashboard (GitHub Pages)
└── run.py                  one command, whole system
```

**Data:** NASA Prognostics Data Repository — C-MAPSS Turbofan Engine Degradation Simulation, subset FD001.

---

### About

Built by **Feras Elghandour** — part of a series on building intelligent systems that solve real-world problems.

🔗 [GitHub](https://github.com/FerasElghandour) · [Site](https://tenx-gc.com)
