"""Data pipeline for the 3 simulated hospitals.  (Phase 1)

Each client is one real dataset. We map each dataset's native label names onto
the SHARED_LABELS, keep frontal images only, collapse multi-annotation formats
to image-level multi-hot vectors, subsample, and wrap with the backbone's
`preprocess` transform.

Shared labels = the 5 pathologies common to ALL THREE sources. Edema is dropped
because VinDr-CXR does not annotate it (NIH and CheXpert do). Decided from the
data (see kaggle_inspect), not assumed.

Verified dataset facts (Kaggle mounts):
  NIH      /kaggle/input/datasets/nih-chest-xrays/data
             Data_Entry_2017.csv: 'Image Index', 'Finding Labels' (pipe-sep),
             'View Position' (PA/AP = both frontal). Images in images_001..012/images/*.png
  CheXpert /kaggle/input/datasets/ashery/chexpert
             train.csv: 'Path' (prefixed 'CheXpert-v1.0-small/'), 'Frontal/Lateral',
             one column per pathology with values {1, 0, -1, blank}. Images jpg.
  VinDr    labels: /kaggle/input/datasets/hiuvquc/vindr-cxr-csv/train.csv  (LONG format:
             one row per radiologist annotation; 'image_id', 'class_name')
           images: /kaggle/input/datasets/sonnlp/vindr-cxr-images/images/train/<image_id>.jpg
"""
from __future__ import annotations
import os
import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

SHARED_LABELS = ["Cardiomegaly", "Effusion", "Pneumothorax",
                 "Atelectasis", "Consolidation"]

# Default Kaggle mount roots. Override via build_client_dataset(roots=...).
ROOTS = {
    "nih_dir":   "/kaggle/input/datasets/nih-chest-xrays/data",
    "chex_dir":  "/kaggle/input/datasets/ashery/chexpert",
    "vindr_csv": "/kaggle/input/datasets/hiuvquc/vindr-cxr-csv/train.csv",
    "vindr_img": "/kaggle/input/datasets/sonnlp/vindr-cxr-images/images/train",
}

# Native label name -> shared label. Only the 5 shared names are kept.
NIH_MAP = {  # NIH native names already equal the shared names
    "Cardiomegaly": "Cardiomegaly", "Effusion": "Effusion",
    "Pneumothorax": "Pneumothorax", "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
}
CHEX_MAP = {
    "Cardiomegaly": "Cardiomegaly", "Pleural Effusion": "Effusion",
    "Pneumothorax": "Pneumothorax", "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
}
VINDR_MAP = {
    "Cardiomegaly": "Cardiomegaly", "Pleural effusion": "Effusion",
    "Pneumothorax": "Pneumothorax", "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
}


# ----------------------------------------------------------------------------
# Per-dataset builders: each returns a DataFrame with ['path'] + SHARED_LABELS
# (label columns are 0.0/1.0 floats), image-level, frontal only.
# ----------------------------------------------------------------------------
def _nih_image_index(nih_dir: str) -> dict:
    """filename -> full path across images_001..012/images/."""
    idx = {}
    for sub in sorted(glob.glob(os.path.join(nih_dir, "images_*"))):
        imgdir = os.path.join(sub, "images")
        if os.path.isdir(imgdir):
            for fn in os.listdir(imgdir):
                idx[fn] = os.path.join(imgdir, fn)
    return idx


def build_nih(roots: dict) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(roots["nih_dir"], "Data_Entry_2017.csv"))
    labels = df["Finding Labels"].fillna("").str.split("|")
    out = pd.DataFrame()
    for shared in SHARED_LABELS:
        out[shared] = labels.apply(lambda L: 1.0 if shared in L else 0.0)
    idx = _nih_image_index(roots["nih_dir"])
    out.insert(0, "path", df["Image Index"].map(idx))
    out = out.dropna(subset=["path"]).reset_index(drop=True)  # NIH is all frontal
    return out


def build_chexpert(roots: dict) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(roots["chex_dir"], "train.csv"))
    df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
    rel = df["Path"].str.replace(r"^CheXpert-v1\.0-small/", "", regex=True)
    out = pd.DataFrame({"path": [os.path.join(roots["chex_dir"], r) for r in rel]})
    for native, shared in CHEX_MAP.items():
        col = df[native]
        out[shared] = (col == 1.0).astype(float).values   # U-Zero: only 1 is positive
    return out[["path"] + SHARED_LABELS]


def build_vindr(roots: dict) -> pd.DataFrame:
    df = pd.read_csv(roots["vindr_csv"])
    df["shared"] = df["class_name"].map(VINDR_MAP)
    present = (df.dropna(subset=["shared"])
                 .groupby("image_id")["shared"].apply(set))
    rows = []
    for iid in df["image_id"].unique():
        s = present.get(iid, set())
        row = {"path": os.path.join(roots["vindr_img"], f"{iid}.jpg")}
        for shared in SHARED_LABELS:
            row[shared] = 1.0 if shared in s else 0.0
        rows.append(row)
    return pd.DataFrame(rows)[["path"] + SHARED_LABELS]


_BUILDERS = {"nih": build_nih, "chexpert": build_chexpert, "vindr": build_vindr}


# ----------------------------------------------------------------------------
class CXRDataset(Dataset):
    """(preprocessed image tensor, multi-hot[5]) from a built DataFrame."""

    def __init__(self, df: pd.DataFrame, preprocess):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.preprocess(img)
        y = torch.tensor([row[l] for l in SHARED_LABELS], dtype=torch.float32)
        return x, y


def build_client_frame(client: str, cfg, roots: dict | None = None,
                       check_exists: bool = True, seed: int | None = None) -> pd.DataFrame:
    """Build, subsample, and (optionally) existence-check one client's frame."""
    roots = roots or ROOTS
    if client not in _BUILDERS:
        raise ValueError(f"unknown client '{client}', expected one of {list(_BUILDERS)}")
    df = _BUILDERS[client](roots)

    n = getattr(cfg.data, "images_per_client", 3000)
    seed = cfg.seed if seed is None else seed
    if len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    if check_exists:
        mask = df["path"].apply(os.path.exists)
        df = df[mask].reset_index(drop=True)
    return df


def build_client_dataset(client: str, cfg, preprocess, roots: dict | None = None):
    """Return a torch Dataset of (image, multi_hot[5]) for one hospital."""
    df = build_client_frame(client, cfg, roots=roots)
    return CXRDataset(df, preprocess)
