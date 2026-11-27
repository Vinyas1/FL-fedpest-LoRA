"""Data pipeline for the 3 simulated hospitals.  (Phase 1)

Each client is one real dataset. We map each dataset's native label names onto
our 6 shared labels, keep frontal images only, subsample to cfg.images_per_client,
and yield (image_tensor, multi_hot_label) batches.

TODO (Phase 1): fill in the per-dataset CSV readers + label maps below once the
Kaggle datasets are attached. Keep rare findings OUT of the shared set; evaluate
them separately (see README / survey Sec VI.B).
"""
from __future__ import annotations

SHARED_LABELS = ["Cardiomegaly", "Effusion", "Pneumothorax",
                 "Atelectasis", "Consolidation", "Edema"]

# Map each dataset's native column names -> our shared label.
# Filled in Phase 1 after inspecting each dataset's CSV.
LABEL_MAPS = {
    "nih": {
        "Cardiomegaly": "Cardiomegaly", "Effusion": "Effusion",
        "Pneumothorax": "Pneumothorax", "Atelectasis": "Atelectasis",
        "Consolidation": "Consolidation", "Edema": "Edema",
    },
    "chexpert": {
        "Cardiomegaly": "Cardiomegaly", "Pleural Effusion": "Effusion",
        "Pneumothorax": "Pneumothorax", "Atelectasis": "Atelectasis",
        "Consolidation": "Consolidation", "Edema": "Edema",
    },
    "vindr": {
        # TODO: confirm VinDr column names in Phase 1
        "Cardiomegaly": "Cardiomegaly", "Pleural effusion": "Effusion",
        "Pneumothorax": "Pneumothorax", "Atelectasis": "Atelectasis",
        "Consolidation": "Consolidation", "Pulmonary edema": "Edema",
    },
}


def build_client_dataset(client: str, cfg):
    """Return a torch Dataset of (image, multi_hot[6]) for one hospital.

    TODO (Phase 1): read the dataset's CSV, filter frontal, apply LABEL_MAPS,
    subsample, and wrap with the backbone's `preprocess` transform.
    """
    raise NotImplementedError(f"build_client_dataset({client}): Phase 1")
