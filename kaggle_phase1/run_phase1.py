"""Kaggle runner for the Phase 1 data check. Clones the repo + runs the check."""
import subprocess
import sys

REPO = "https://github.com/Vinyas1/FL-fedpest-LoRA.git"


def sh(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


sh(["git", "clone", "--depth", "1", REPO])
sh([sys.executable, "-m", "pip", "install", "-q", "-r",
    "FL-fedpest-LoRA/requirements.txt"])
sh([sys.executable, "FL-fedpest-LoRA/scripts/phase1_check_data.py"])
print("\n[kernel] done.", flush=True)
