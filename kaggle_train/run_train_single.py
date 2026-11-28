"""Kaggle GPU runner: isolated single-client LoRA training (baselines A & B)."""
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/Vinyas1/FL-fedpest-LoRA.git"
WORK = "/tmp/repo"          # clone off /kaggle/working so result pull stays small


def sh(cmd, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


sh(["git", "clone", "--depth", "1", REPO, WORK])
sh([sys.executable, "-m", "pip", "install", "-q", "-r", f"{WORK}/requirements.txt"])
sh([sys.executable, "scripts/train_single.py", "--images", "2000", "--epochs", "3"],
   cwd=WORK)

# copy just the metrics json back for a clean pull
src = f"{WORK}/outputs/single_client_metrics.json"
if os.path.exists(src):
    shutil.copy(src, "/kaggle/working/")
print("\n[kernel] done.", flush=True)
