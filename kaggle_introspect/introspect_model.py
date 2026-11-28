"""Phase 2 prep: discover BiomedCLIP's internal module names.

We need the exact names of the Linear layers inside the attention blocks so
LoRA (peft) can target them. Prints the top-level structure and a compact
summary of every nn.Linear, grouped by a normalized name, for vision vs text.
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "open_clip_torch", "peft"], check=True)

import re
from collections import Counter
import torch.nn as nn
import open_clip

NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
print("loading", NAME)
model, _, _ = open_clip.create_model_and_transforms(NAME)

print("\n=== TOP-LEVEL CHILDREN ===")
for n, _ in model.named_children():
    print("  ", n)

print("\n=== LINEAR LAYERS (normalized name -> count, example) ===")
# normalize by stripping block indices so we see the repeating pattern
norm = {}
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear):
        key = re.sub(r"\.\d+\.", ".N.", name)
        norm.setdefault(key, {"count": 0, "example": name,
                              "shape": (mod.in_features, mod.out_features)})
        norm[key]["count"] += 1

for key in sorted(norm):
    info = norm[key]
    branch = "VISION" if "visual" in key else ("TEXT" if "text" in key else "OTHER")
    print(f"  [{branch}] {key}   x{info['count']}  "
          f"(in={info['shape'][0]}, out={info['shape'][1]})  e.g. {info['example']}")

# Also show MultiheadAttention presence (peft can't LoRA these directly)
print("\n=== nn.MultiheadAttention modules (if any) ===")
mha = [n for n, m in model.named_modules() if isinstance(m, nn.MultiheadAttention)]
print("  count:", len(mha), "| examples:", mha[:3])

print("\n[introspect] done.")
