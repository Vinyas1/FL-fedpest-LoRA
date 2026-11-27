# Fed-PEST

**Parameter-Efficient Federated Personalization for Chest X-Ray Diagnostics**

A frozen BiomedCLIP backbone + tiny LoRA adapters, trained across simulated
"hospitals" (federated), where each hospital keeps a **private local adapter**
in addition to the shared global one.

## The one claim this project makes

> Does giving each hospital a small **private** LoRA adapter improve the
> **worst-performing hospital's** AUROC compared to a global-only federated
> adapter, at the same parameter budget — and does personalization stay stable
> after aggregation?

Everything in this repo exists to answer that. No scanner claims, no privacy
"guarantees", no ">98%" numbers until they are *measured*.

## What we compare (same data / backbone / budget)

| # | Model | What it is |
|---|-------|-----------|
| A | Frozen BiomedCLIP, zero-shot | floor |
| B | Isolated local LoRA (no federation) | per-hospital, no sharing |
| C | Global-only federated LoRA | the Poudel-style baseline |
| D | Global + private-local LoRA | **our method** |

Then ablations: remove local branch; shared-B vs independent-local factorization;
vary rank `r`.

## Metrics (report exactly these)

- Macro AUROC + per-disease AUROC/AUPRC
- **Per-hospital AUROC, and worst-hospital AUROC** ← the money metric
- Leave-one-hospital-out transfer
- Communication: MB/round with honest byte accounting
- 3 seeds + bootstrap confidence intervals

## Data (3 simulated hospitals)

| Client | Dataset | Kaggle source |
|--------|---------|---------------|
| 0 | NIH ChestX-ray14 | `nih-chest-xrays/data` |
| 1 | CheXpert | `ashery/chexpert` (or official) |
| 2 | VinDr-CXR | `awsaf49/vinbigdata-...` |

6 shared labels (all three support): **Cardiomegaly, Pleural Effusion,
Pneumothorax, Atelectasis, Consolidation, Edema**. Label-derived text prompts
(no real radiology reports). ~2–4k frontal images/client.

## Compute

Kaggle free GPU (T4, 16 GB). Code lives here; a thin Kaggle kernel clones the
repo and runs `scripts/run.py --config configs/<name>.yaml`.

## Layout

```
src/fedpest/      core package (config, model, data, federated, evaluate)
scripts/          phase0_smoke_test.py, run.py
configs/          yaml configs, one per experiment
kaggle/           how to run on Kaggle's GPU
```

## Status: Phase 0 — load model + smoke test. See `scripts/phase0_smoke_test.py`.
