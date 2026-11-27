"""Kaggle kernel runner for the Phase 0 smoke test.

Pushed to Kaggle via the CLI; runs on their GPU. Clones the public repo,
installs deps, and runs the smoke test. Output is captured in the kernel log,
which we pull back with `kaggle kernels output`.
"""
import subprocess
import sys

REPO = "https://github.com/Vinyas1/FL-fedpest-LoRA.git"


def sh(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


sh(["git", "clone", "--depth", "1", REPO])
sh([sys.executable, "-m", "pip", "install", "-q", "-r",
    "FL-fedpest-LoRA/requirements.txt"])
sh([sys.executable, "FL-fedpest-LoRA/scripts/phase0_smoke_test.py"])
print("\n[kernel] done.", flush=True)
