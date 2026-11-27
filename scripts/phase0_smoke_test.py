"""Phase 0 smoke test.

Proves the environment works end to end:
  1. BiomedCLIP loads
  2. we can embed an image and a set of text prompts
  3. zero-shot similarity between them is sane

Run (Kaggle or any machine with the deps):
    python scripts/phase0_smoke_test.py

Self-contained: with no image path it synthesizes one in-memory, so it needs
no dataset attached and no network beyond the model download. This is the thing
to run FIRST on Kaggle's T4. (Pass a real image path as argv[1] to use one.)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedpest.model import load_backbone, embed_image, embed_text, label_prompts

BACKBONE = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
LABELS = ["Cardiomegaly", "Effusion", "Pneumothorax",
          "Atelectasis", "Consolidation", "Edema"]


def get_sample_image(path: str | None) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    # Synthetic grayscale image — scores are meaningless, we only prove the
    # image->embedding path runs. No external fetch, so nothing to be blocked.
    print("[smoke] synthesizing a test image (no download) ...")
    rng = np.random.default_rng(0)
    arr = (rng.normal(0.5, 0.15, size=(224, 224)).clip(0, 1) * 255).astype("uint8")
    return Image.fromarray(arr).convert("RGB")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device}")

    print("[smoke] loading BiomedCLIP (first run downloads weights) ...")
    model, preprocess, tokenizer = load_backbone(BACKBONE, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] backbone loaded: {n_params/1e6:.1f}M params, all frozen")

    img = get_sample_image(sys.argv[1] if len(sys.argv) > 1 else None)
    img_feat = embed_image(model, preprocess, img, device)         # [1, D]
    txt_feat = embed_text(model, tokenizer, label_prompts(LABELS), device)  # [L, D]

    sims = (img_feat @ txt_feat.T).squeeze(0)                      # [L]
    print(f"[smoke] image embedding dim = {img_feat.shape[-1]}")
    print("[smoke] zero-shot similarity to each label prompt:")
    for lab, s in sorted(zip(LABELS, sims.tolist()), key=lambda x: -x[1]):
        print(f"         {lab:14s} {s:+.3f}")

    print("\n[smoke] PASS — model loads, embeds image+text, produces scores.")


if __name__ == "__main__":
    main()
