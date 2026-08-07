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

Verify the GPU runtime:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The environment intentionally covers three future experiment families:

- classical PRNU, correlation, and PCE-style analysis;
- residual, frequency-domain, and image-forensics features;
- PyTorch-based semantic and hybrid AIGC detectors.

Datasets, model weights, logs, and experiment outputs are ignored by Git. They
must not be committed to this repository.

