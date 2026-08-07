# G-FITS

G-FITS (Software-Noise-Calibrated Generator Fingerprint Matching for Robust
AIGC Attribution) is a reproducible digital-image-forensics research project.
The intended research question is whether a candidate generator fingerprint can
be calibrated against correlation shared by downstream software pipelines.

Phase 0 is the only completed phase. It provides repository structure, quality
gates, immutable third-party revision records, literature/code audits, and a
byte-level dataset manifest. It does **not** contain a detector or claim an
experimental result.

## Environment

The reference environment is Python 3.10 with PyTorch 2.5.1/CUDA 12.4. On the
configured remote workstation it is available as the `fits-aigc` Conda
environment and the `Python (fits-aigc)` Jupyter kernel.

```bash
conda env create -f environment.yml
conda activate fits-aigc
python -m pip install -e .
```

For development and validation:

```bash
python -m pip install -e ".[dev]"
pre-commit install
ruff check .
ruff format --check .
pytest
```

## Phase 0 CLI

The manifest records the raw bytes of supported image files. Building or
verifying a manifest never decodes, resizes, crops, recompresses, or changes an
image.

```bash
python -m gfits.cli build-manifest \
  --root /path/to/images \
  --output /path/to/manifest.json

python -m gfits.cli verify-manifest \
  --manifest /path/to/manifest.json \
  --root /path/to/images \
  --strict
```

Use repeated `--include` arguments to override the default image globs, or
`--all-files` when every regular file belongs in the evidence set. `--strict`
also fails on files that are present under the root but absent from the
manifest.

## Repository layout

```text
configs/              Versioned phase configuration and schemas
src/gfits/             Installable Python package
tests/                 Unit and CLI tests
third_party/LOCK.json  Audited upstream revisions; no vendored source
reports/phase0/        Literature, repository, and gate reports
artifacts/phase0/      Machine-generated Phase 0 validation evidence
```

The Phase 0 source audit is in
[`reports/phase0/LITERATURE_MATRIX.md`](reports/phase0/LITERATURE_MATRIX.md) and
[`reports/phase0/CODE_REPOSITORY_AUDIT.md`](reports/phase0/CODE_REPOSITORY_AUDIT.md).
The exact external revisions are in [`third_party/LOCK.json`](third_party/LOCK.json).

## Remote storage

Large datasets, weights, caches, and experiment outputs belong outside Git:

```text
/mnt/data/jkl/FITS/
|-- datasets/
|-- checkpoints/
|-- outputs/
`-- cache/
```

The configured remote environment defines `FITS_DATA_ROOT`,
`FITS_CHECKPOINT_ROOT`, `FITS_OUTPUT_ROOT`, `HF_HOME`, `TORCH_HOME`, and
`XDG_CACHE_HOME` for this volume. Credentials and private keys must never be
stored in this repository.
