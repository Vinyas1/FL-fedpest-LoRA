"""Plain-loop federated trainer.  (Phase 4 / 5)

No networking — a simple sequential simulation, which is standard and far easier
to debug than Flower for 3 clients on one GPU:

    for round in rounds:
        for client in clients:
            train LoRA locally on that client's data
        aggregate the GLOBAL adapter (FedAvg); keep LOCAL adapters private

Methods (cfg.federated.method):
    zero_shot    - no training (baseline A)
    isolated     - each client trains, never aggregates (baseline B)
    global_only  - aggregate the single shared adapter (baseline C, Poudel-style)
    global_local - shared global adapter + private per-client local adapter ( OURS, D)
"""
from __future__ import annotations


def fedavg(state_dicts: list[dict]) -> dict:
    """Uniform average of adapter tensors across clients."""
    keys = state_dicts[0].keys()
    out = {}
    for k in keys:
        stacked = sum(sd[k] for sd in state_dicts) / len(state_dicts)
        out[k] = stacked
    return out


def run_federated(cfg):
    """Entry point used by scripts/run.py.

    TODO (Phase 4): build per-client models (shared global LoRA), train a round,
    FedAvg the global adapter, repeat. (Phase 5): add the private local adapter
    and the shared_b vs independent_local switch. Save adapters + metrics.json.
    """
    raise NotImplementedError("run_federated: Phase 4/5")
