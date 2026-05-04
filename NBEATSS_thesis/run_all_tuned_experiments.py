"""
Combined runner: N-BEATS-S (24 runs) then N-HiTS-S (24 runs) = 48 total.
Launches sequentially so they don't compete for GPU.
"""
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
PYTHON_EXE = str(BASE_DIR.parent / ".venv" / "Scripts" / "python.exe")

def run_script(name, script_path):
    print(f"\n{'#'*70}")
    print(f"  STARTING: {name}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")
    
    start = time.time()
    result = subprocess.run(
        [PYTHON_EXE, str(script_path)],
        cwd=str(BASE_DIR),
        env={**__import__('os').environ, "WANDB_MODE": "offline", "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.time() - start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    
    status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print(f"\n  {name}: {status} in {h:02d}:{m:02d}:{s:02d}")
    return result.returncode

def main():
    overall_start = time.time()
    print(f"\n{'='*70}")
    print(f"  ALL TUNED EXPERIMENTS — 48 TOTAL")
    print(f"  N-BEATS-S: hidden=256, blocks=20 (Van Belle et al., 2023)")
    print(f"  N-HiTS-S:  hidden=512, blocks=10 (M4 HP-tuned)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Phase 1: N-BEATS-S (24 runs)
    rc1 = run_script("N-BEATS-S Tuned Experiments (24 runs)", BASE_DIR / "run_experiments_tuned.py")
    
    # Phase 2: N-HiTS-S (24 runs)
    rc2 = run_script("N-HiTS-S Tuned Experiments (24 runs)", BASE_DIR / "run_nhits_experiments_tuned.py")
    
    total = time.time() - overall_start
    h, rem = divmod(int(total), 3600)
    m, s = divmod(rem, 60)
    print(f"\n{'='*70}")
    print(f"  ALL DONE — Total time: {h:02d}:{m:02d}:{s:02d}")
    print(f"  N-BEATS-S: {'OK' if rc1 == 0 else 'FAILED'}")
    print(f"  N-HiTS-S:  {'OK' if rc2 == 0 else 'FAILED'}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
