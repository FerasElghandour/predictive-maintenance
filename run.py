"""
One command, whole system:  python run.py

    Reality -> Data -> Understanding -> Prediction -> Decision -> Action

Each stage writes to artifacts/ or reports/, so stages are cached and
independently inspectable. Re-running is cheap.

The LSTM trains in resumable 6-epoch chunks (checkpointed every epoch, so a
killed job never loses work). This runner keeps calling it until it converges.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
LSTM_DONE = ROOT / "artifacts" / "results_lstm.json"
MAX_LSTM_CHUNKS = 12   # hard stop; each chunk is 6 epochs


def stage(name, *cmd):
    print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
    t = time.time()
    r = subprocess.run([sys.executable, "-u", *cmd], cwd=SRC)
    if r.returncode != 0:
        sys.exit(f"\nFAILED at: {name}")
    print(f"  ...{time.time() - t:.1f}s")


stage("Box 1+2   Data layer & sensor triage (21 sensors -> 12)", "understand.py")
stage("Box 3     Linear baseline + RandomForest", "models.py", "snapshot")

# LSTM: keep going until it converges or we hit the cap.
for i in range(MAX_LSTM_CHUNKS):
    if LSTM_DONE.exists():
        break
    stage(f"Box 3     LSTM (chunk {i + 1}, resumable)", "models.py", "lstm")
else:
    print("\nNOTE: LSTM hit the chunk cap without converging. It will be excluded "
          "from the comparison rather than reported dishonestly.")

stage("Box 3     Model comparison — pick the winner", "models.py", "report")
stage("Box 4     Decision layer — RUL -> health -> action", "decide.py")
stage("Box 4     Policy replay — the honest test", "evaluate_policy.py")
stage("Box 5     Build the dashboard", "dashboard.py")

print("\n\n" + "=" * 70)
print("  DONE.  Open docs/index.html in a browser.")
print("=" * 70)
