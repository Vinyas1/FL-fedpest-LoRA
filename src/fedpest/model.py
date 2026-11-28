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


# --- LoRA attachment (Phase 2) ---
def attach_lora(model, cfg):
    """Attach LoRA adapters to the vision tower's attention projections.

    BiomedCLIP's vision tower is a timm ViT-B/16 whose attention uses a single
    fused `qkv` Linear per block (names like visual.trunk.blocks.N.attn.qkv).
    We adapt only the vision encoder — hospitals differ in imaging, and the text
    class prototypes stay fixed — which keeps the method parameter-efficient and
    the federated story simple. Target names confirmed by kaggle_introspect.

    Only LoRA params end up trainable; everything else stays frozen.
    """
    from peft import LoraConfig, get_peft_model

    targets = list(getattr(cfg.model, "target_modules", ["qkv"]))
    lconf = LoraConfig(
        r=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=targets,
        bias="none",
    )
    peft_model = get_peft_model(model, lconf)
    return peft_model


def lora_state_dict(model) -> dict:
    """The LoRA tensors only — what crosses the wire in federated rounds."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if "lora_" in k}


def load_lora_state_dict(model, sd: dict):
    model.load_state_dict(sd, strict=False)


def count_trainable(model) -> tuple[int, int]:
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return trn, tot
