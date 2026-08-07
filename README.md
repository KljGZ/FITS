# FITS for AIGC image forensics

This repository is reserved for studying FITS-inspired AIGC image detection and
source attribution. At this stage it contains infrastructure configuration only;
no detector implementation or experiment code has been added.

## Baseline environment

The baseline targets the remote Ubuntu host with NVIDIA RTX 4090 GPUs and a
CUDA-12.4-capable driver. Python 3.10 is used because it remains compatible with
representative AIGC-image-detection baselines while supporting a modern PyTorch
2.x stack.

Create the environment:

```bash
conda env create -f environment.yml
conda activate fits-aigc
```

On the configured workstation, the remote host is available through the SSH
alias `fits-remote`:

```bash
ssh fits-remote
cd /home/jkl/FITS
conda activate fits-aigc
```

Jupyter also exposes the environment as the `Python (fits-aigc)` kernel.

Verify the GPU runtime:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Remote storage layout

Large artifacts belong on the remote data volume rather than the system disk:

```text
/mnt/data/jkl/FITS/
├── datasets/
├── checkpoints/
├── outputs/
└── cache/
```

The installed remote environment defines `FITS_DATA_ROOT`,
`FITS_CHECKPOINT_ROOT`, `FITS_OUTPUT_ROOT`, `HF_HOME`, `TORCH_HOME`, and
`XDG_CACHE_HOME` so supported tools use this data volume automatically.

The environment intentionally covers three future experiment families:

- classical PRNU, correlation, and PCE-style analysis;
- residual, frequency-domain, and image-forensics features;
- PyTorch-based semantic and hybrid AIGC detectors.

Datasets, model weights, logs, and experiment outputs are ignored by Git. They
must not be committed to this repository.
