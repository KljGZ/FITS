# G-FITS

G-FITS (Software-Noise-Calibrated Generator Fingerprint Matching for Robust
AIGC Attribution) is a reproducible digital-image-forensics research project.
The intended research question is whether a candidate generator fingerprint can
be calibrated against correlation shared by downstream software pipelines.

Phase 0 provides repository structure, quality gates, immutable third-party
revision records, literature/code audits, and a byte-level dataset manifest.
Phase E00 adds numerical matching primitives and a pre-registered synthetic
mechanism gate. E00 is residual-level evidence only: it is **not** a detector,
a real-image experiment, or an AIGC performance claim.

Phase E01 performs the first real-image mechanism replication on 400 official
VISION files. It tests whether same-pipeline, different-device controls align
the non-match distribution across native, Facebook High, Facebook Low, and
WhatsApp pipelines. E01 passes its registered alignment hypothesis, while
showing no attribution-performance improvement over the raw score. It is
therefore still **not** an AIGC detector result.

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

## Phase 0, E00, and E01 CLI

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

E00 implements zero-mean CC/NCC, signed PCE, the single-control FITS division,
and a project-defined multi-control median extension named FITS+. The locked
`prnu-python` checkout is used only for numerical compatibility checks; its
source is never modified.

```bash
python -m gfits.cli validate-prnu \
  --upstream-root third_party/prnu-python \
  --output artifacts/e00/prnu-cross-validation.json

python -m gfits.cli validate-synthetic-fits \
  --config configs/e00.yaml \
  --output-dir artifacts/e00 \
  --profile gate
```

The exact E00 gate seed and thresholds were frozen in `configs/e00.yaml` before
the gate run. A failed model-level check blocks E01.

E01 downloads a byte-identical, SHA-256-addressed VISION subset and evaluates
full-frame PRNU fingerprints without resizing, cropping, recompression, EXIF
transpose, or color-space conversion by G-FITS. The upstream PRNU checkout must
match the commit and canonical source hash locked by the project.

```bash
python -m gfits.cli download-vision-e01 \
  --config configs/e01.yaml \
  --data-root /mnt/data/jkl/FITS/datasets/vision-e01 \
  --manifest /mnt/data/jkl/FITS/datasets/vision-e01-manifest.json

python -m gfits.cli run-e01 \
  --config configs/e01.yaml \
  --manifest /mnt/data/jkl/FITS/datasets/vision-e01-manifest.json \
  --data-root /mnt/data/jkl/FITS/datasets/vision-e01 \
  --upstream-root third_party/prnu-python \
  --output-dir /mnt/data/jkl/FITS/outputs/e01/gate \
  --cache-root /mnt/data/jkl/FITS/cache/e01
```

The registered E01 threshold is fitted only on calibration H0 rows. Template,
calibration, and test source indices are disjoint. See the stage report for the
claim boundary and the negative performance finding.

## Repository layout

```text
configs/              Versioned phase configuration and schemas
src/gfits/             Installable Python package
tests/                 Unit and CLI tests
third_party/LOCK.json  Audited upstream revisions; no vendored source
reports/phase0/        Literature, repository, and gate reports
artifacts/phase0/      Machine-generated Phase 0 validation evidence
reports/e00/           E00 definitions, results, and limitations
artifacts/e00/         Raw E00 CSV, JSON, and diagnostic figure
reports/e01/           E01 protocol, results, evidence, and limitations
artifacts/e01/         E01 manifest, raw scores, metrics, figures, and QA
```

The Phase 0 source audit is in
[`reports/phase0/LITERATURE_MATRIX.md`](reports/phase0/LITERATURE_MATRIX.md) and
[`reports/phase0/CODE_REPOSITORY_AUDIT.md`](reports/phase0/CODE_REPOSITORY_AUDIT.md).
The exact external revisions are in [`third_party/LOCK.json`](third_party/LOCK.json).
The E00 result and interpretation are in [`reports/e00/E00_REPORT.md`](reports/e00/E00_REPORT.md).
The E01 result and interpretation are in [`reports/e01/E01_REPORT.md`](reports/e01/E01_REPORT.md).

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
