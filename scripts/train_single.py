"""Phase 2/3: isolated single-client LoRA training.

For each hospital: measure zero-shot AUROC (baseline A), train a LoRA adapter on
that hospital's own train split, then measure AUROC again (baseline B = isolated
local LoRA). No federation yet — this validates that LoRA training actually helps
and gives us baselines A and B for every hospital.

    python scripts/train_single.py --config configs/base.yaml --images 2000 --epochs 3

Writes outputs/single_client_metrics.json for pulling structured numbers.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedpest.config import load_config
from fedpest.model import load_backbone, attach_lora, count_trainable
from fedpest.data import build_client_datasets, SHARED_LABELS
from fedpest.train import text_prototypes, train_epochs, predict, make_loader
from fedpest.evaluate import per_disease_auroc, macro_auroc


def report(tag, y, p):
    pd_ = per_disease_auroc(y, p, SHARED_LABELS)
    macro = macro_auroc(y, p, SHARED_LABELS)
    print(f"  [{tag}] macro AUROC = {macro:.4f}")
    for k, v in pd_.items():
        print(f"      {k:14s} {v:.4f}")
    return {"macro": macro, "per_disease": pd_}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--images", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.images:
        cfg.data.images_per_client = args.images
    epochs = args.epochs or cfg.federated.local_epochs
    lr = args.lr if args.lr is not None else cfg.optim.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[single] device={device} images/client={cfg.data.images_per_client} "
          f"epochs={epochs} lr={lr} rank={cfg.model.lora_rank}")

    results = {}
    for client in cfg.federated.clients:
        print("\n" + "=" * 68)
        print(f"CLIENT: {client}")

        # fresh frozen backbone + prototypes (text is never adapted here)
        model, preprocess, tokenizer = load_backbone(cfg.model.backbone, device)
        logit_scale = model.logit_scale.exp().item()
        protos = text_prototypes(model, tokenizer, SHARED_LABELS, device)

        model = attach_lora(model, cfg).to(device)
        trn, tot = count_trainable(model)
        print(f"  trainable params: {trn:,} / {tot:,} ({100*trn/tot:.3f}%)")

        train_ds, test_ds = build_client_datasets(client, cfg, preprocess)
        print(f"  train={len(train_ds)}  test={len(test_ds)}")
        train_loader = make_loader(train_ds, cfg.data.batch_size, shuffle=True)
        test_loader = make_loader(test_ds, cfg.data.batch_size)

        # Baseline A: zero-shot (LoRA starts as a no-op, so this is frozen zero-shot)
        y, p = predict(model, test_loader, protos, logit_scale, device)
        res_a = report("A: zero-shot", y, p)

        # Train isolated LoRA, evaluating the test set after every epoch so we can
        # see the trajectory (does more training help or overfit?).
        trajectory = []

        def on_epoch_end(ep, mean_loss):
            yy, pp = predict(model, test_loader, protos, logit_scale, device)
            m = macro_auroc(yy, pp, SHARED_LABELS)
            trajectory.append({"epoch": ep, "loss": mean_loss, "macro": m})
            print(f"    -> test macro AUROC = {m:.4f}", flush=True)

        train_epochs(model, train_loader, protos, logit_scale, device,
                     epochs=epochs, lr=lr, weight_decay=cfg.optim.weight_decay,
                     on_epoch_end=on_epoch_end)

        # Baseline B: final-epoch isolated LoRA (+ best epoch for reference)
        y, p = predict(model, test_loader, protos, logit_scale, device)
        res_b = report("B: isolated-LoRA (final)", y, p)
        best = max(trajectory, key=lambda t: t["macro"])
        print(f"  best epoch = {best['epoch']} (macro {best['macro']:.4f})")

        results[client] = {"zero_shot": res_a, "isolated_lora": res_b,
                           "trajectory": trajectory, "best_epoch": best,
                           "n_train": len(train_ds), "n_test": len(test_ds)}
        del model
        torch.cuda.empty_cache()

    # Summary: worst-hospital macro AUROC for each method
    print("\n" + "=" * 68)
    print("SUMMARY (macro AUROC per hospital):")
    print(f"  {'client':10s}  {'zero-shot':>10s}  {'isolated':>10s}")
    for c, r in results.items():
        print(f"  {c:10s}  {r['zero_shot']['macro']:>10.4f}  "
              f"{r['isolated_lora']['macro']:>10.4f}")
    worst_a = min(r['zero_shot']['macro'] for r in results.values())
    worst_b = min(r['isolated_lora']['macro'] for r in results.values())
    print(f"  {'WORST':10s}  {worst_a:>10.4f}  {worst_b:>10.4f}")

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    (outdir / "single_client_metrics.json").write_text(json.dumps(results, indent=2))
    print("\n[single] wrote outputs/single_client_metrics.json")


if __name__ == "__main__":
    main()
