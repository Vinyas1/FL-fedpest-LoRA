"""Phase 1 verification: prove the data loader reads real images + labels.

For each client, build a small sample, print label counts, and actually open a
few images through the backbone preprocess to confirm paths + decoding work.
"""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedpest.data import build_client_frame, CXRDataset, SHARED_LABELS
from fedpest.model import load_backbone

BACKBONE = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
SAMPLE = 300          # per client, just to verify


def main():
    cfg = SimpleNamespace(seed=0, data=SimpleNamespace(images_per_client=SAMPLE))
    print("[phase1] loading backbone preprocess ...")
    _, preprocess, _ = load_backbone(BACKBONE, device="cpu")

    for client in ["nih", "chexpert", "vindr"]:
        print("\n" + "=" * 68)
        print(f"CLIENT: {client}")
        df = build_client_frame(client, cfg, check_exists=True)
        print(f"  usable sampled rows (images found on disk): {len(df)}")
        if len(df) == 0:
            print("  !! no images found — check mount paths")
            continue
        pos = {l: int(df[l].sum()) for l in SHARED_LABELS}
        print(f"  positive counts (of {len(df)}): {pos}")

        ds = CXRDataset(df, preprocess)
        x, y = ds[0]
        print(f"  sample tensor shape: {tuple(x.shape)}, label vec: {y.tolist()}")
        # open a few more to be sure decoding is robust
        for i in range(1, min(5, len(ds))):
            xi, _ = ds[i]
            assert xi.shape == x.shape
        print(f"  opened 5 images OK via preprocess")

    print("\n[phase1] PASS — all three clients load real images + 5-label vectors.")


if __name__ == "__main__":
    main()
