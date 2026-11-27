"""Single entry point for every experiment.

    python scripts/run.py --config configs/<name>.yaml

Routes to the right method based on cfg.federated.method. Phase 0 only wires
the config + device; the trainer is filled in from Phase 4 on.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedpest.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[run] method={cfg.federated.method} seed={cfg.seed} "
          f"clients={cfg.federated.clients}")

    from fedpest.federated import run_federated
    run_federated(cfg)          # NotImplementedError until Phase 4


if __name__ == "__main__":
    main()
