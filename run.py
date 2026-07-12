"""
One command, whole system:  python run.py

Reality -> Data -> Understanding -> Prediction -> Decision -> Action

Each stage writes its output to artifacts/ or reports/, so stages are cached and
independently inspectable. The LSTM trains in resumable chunks; re-run to continue.
"""
import subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).parent / "src"
STAGES = [
    ("Box 1+2  Data & sensor triage",      ["understand.py"]),
    ("Box 3    Linear + RandomForest",     ["models.py", "snapshot"]),
    ("Box 3    LSTM (resumable)",          ["models.py", "lstm"]),
    ("Box 3    Model comparison",          ["models.py", "report"]),
    ("Box 4    Decision layer",            ["decide.py"]),
    ("Box 4    Policy replay (honest test)", ["evaluate_policy.py"]),
    ("Box 5    Build dashboard",           ["dashboard.py"]),
]

for name, cmd in STAGES:
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    t = time.time()
    r = subprocess.run([sys.executable, "-u", *cmd], cwd=SRC)
    if r.returncode != 0:
        sys.exit(f"FAILED at: {name}")
    print(f"  ...{time.time()-t:.1f}s")

print("\n\nDone. Open docs/index.html in a browser.")
