"""Training + evaluation core.  (Phase 2/3)

Classifier = CLIP-style. We build 5 frozen text "prototype" embeddings from the
label prompts once, then score an image by cosine similarity to each prototype
(scaled by the model's logit_scale). Multi-label BCE trains the LoRA params so
image embeddings align with the correct label prototypes. No extra classifier
head -> nothing extra to aggregate in the federated setting.

Only the LoRA params are trainable (the backbone + text prototypes are frozen).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import embed_text, label_prompts


@torch.no_grad()
def text_prototypes(model, tokenizer, labels, device):
    """[L, D] L2-normalized, frozen prototypes — one prompt per label."""
    return embed_text(model, tokenizer, label_prompts(labels), device)


def image_logits(model, pixel_values, prototypes, logit_scale):
    """[B, L] similarity logits for a batch of images."""
    feat = model.encode_image(pixel_values)
    feat = F.normalize(feat, dim=-1)
    return logit_scale * feat @ prototypes.T


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def train_epochs(model, loader, prototypes, logit_scale, device,
                 epochs=1, lr=5e-4, weight_decay=0.0, log_every=20,
                 on_epoch_end=None):
    """Train the trainable (LoRA) params. One optimizer spans all epochs.

    on_epoch_end(epoch, mean_loss) is called after each epoch (e.g. to evaluate
    the test set and record the trajectory). Returns the last mean loss.
    """
    opt = torch.optim.AdamW(trainable_params(model), lr=lr, weight_decay=weight_decay)
    last = 0.0
    for ep in range(epochs):
        model.train()
        running, n = 0.0, 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits = image_logits(model, x, prototypes, logit_scale)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(x)
            n += len(x)
            if log_every and i % log_every == 0:
                print(f"    ep{ep} step{i} loss={loss.item():.4f}", flush=True)
        last = running / max(n, 1)
        print(f"    [epoch {ep}] mean loss={last:.4f}", flush=True)
        if on_epoch_end is not None:
            on_epoch_end(ep, last)
    return last


@torch.no_grad()
def predict(model, loader, prototypes, logit_scale, device):
    """Return (y_true[N,L], y_score[N,L]).

    y_score is the RAW logit (scaled cosine similarity), not sigmoid. AUROC/AUPRC
    only need a monotonic ranking score, and the large CLIP logit_scale (~100)
    saturates sigmoid to exactly 1.0 in float32 -> constant probs -> AUROC 0.5.
    Raw logits rank correctly and avoid that.
    """
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = image_logits(model, x, prototypes, logit_scale)
        ps.append(logits.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def make_loader(dataset, batch_size=64, shuffle=False, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)
