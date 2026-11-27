# Running Fed-PEST on Kaggle's free GPU

The efficient loop: **code lives in GitHub**, the Kaggle kernel just clones it
and runs a config. You (or Claude Code via the Kaggle CLI) push the kernel; it
runs on the T4; you pull the output back.

## One-time setup

1. Kaggle → Settings → **Create New API Token** → downloads `kaggle.json`.
2. Put it where the CLI looks:
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`
3. `pip install kaggle`

## The kernel (thin runner)

The notebook/kernel is just 3 cells:

```python
# cell 1 — get the code
!git clone https://github.com/Vinyas1/FL-fedpest-LoRA.git
%cd FL-fedpest-LoRA
!pip install -q -r requirements.txt

# cell 2 — Phase 0 smoke test (run this FIRST)
!python scripts/phase0_smoke_test.py

# cell 3 — later: a real experiment
# !python scripts/run.py --config configs/base.yaml
```

Set **Accelerator → GPU T4** in the kernel settings (right panel).

## Driving it from the Kaggle CLI (hands-free loop)

```bash
kaggle kernels push    -p kaggle/          # send the kernel
kaggle kernels status  <user>/<kernel>     # poll
kaggle kernels output  <user>/<kernel> -p outputs/   # pull results
```

A `kernel-metadata.json` goes here in `kaggle/` once the GitHub repo URL and your
Kaggle username are known (Phase 0 wrap-up).

## Attaching data (Phase 1)

In the kernel UI: **Add Input → Datasets →** search and attach:
`nih-chest-xrays/data`, a CheXpert mirror, and a VinDr-CXR mirror. They mount
read-only under `/kaggle/input/...` — no downloading.
