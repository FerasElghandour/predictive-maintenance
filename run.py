"""Run the corrected release evaluation and rebuild its interactive report."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    for script in ['validate_release.py','build_report.py']:
        subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True)
