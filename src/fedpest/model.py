"""BiomedCLIP backbone + LoRA adapters.

Phase 0 goal: load the frozen backbone and embed one image + one text prompt.
Later phases attach LoRA via `peft` and expose the adapter state_dict for the
federated loop.

BiomedCLIP is an open_clip model hosted on HuggingFace:
    hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
It ships its own tokenizer and image preprocessing.
"""
from __future__ import annotations
import torch
import open_clip


def load_backbone(name: str, device: str = "cuda"):
    """Return (model, preprocess, tokenizer), frozen, in eval mode.

    `model` has .encode_image(pixel_values) and .encode_text(token_ids),
    both returning L2-normalizable feature vectors.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(name)
    tokenizer = open_clip.get_tokenizer(name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)          # freeze everything; LoRA adds the only trainables
    return model, preprocess, tokenizer


@torch.no_grad()
def embed_image(model, preprocess, pil_image, device: str = "cuda") -> torch.Tensor:
    px = preprocess(pil_image).unsqueeze(0).to(device)
    feat = model.encode_image(px)
    return torch.nn.functional.normalize(feat, dim=-1)


@torch.no_grad()
def embed_text(model, tokenizer, texts, device: str = "cuda") -> torch.Tensor:
    toks = tokenizer(texts).to(device)
    feat = model.encode_text(toks)
    return torch.nn.functional.normalize(feat, dim=-1)


def label_prompts(labels: list[str]) -> list[str]:
    """Label-derived text prompts (no real radiology reports).

    Two prompts per label (present / absent) is the standard CLIP zero-shot
    framing; here we keep it simple with a 'present' template per label.
    """
    return [f"a chest x-ray showing {lab.lower()}" for lab in labels]


# --- Phase 3+: LoRA attachment (stub, implemented when Phase 2 baseline runs) ---
def attach_lora(model, cfg):
    """Wrap attention projections with LoRA adapters via peft.

    TODO (Phase 3): build a LoraConfig(r=cfg.lora_rank, ...), apply to the
    visual + text transformer blocks named in cfg.target_modules, and return
    the peft-wrapped model. `factorization == 'independent_local'` is the
    research switch that gives each client its own A and B (see README).
    """
    raise NotImplementedError("attach_lora: implemented in Phase 3")
