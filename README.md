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

Phase E02 runs a pre-registered same-source generator-signal screen on native
DMimageDetection and GenImage subsets. It compares four residual extractors
(wavelet, SRM, two-low-bit, and Noiseprint) with four fingerprint aggregators
(mean, PRNU MLE, median, and trimmed mean). The public releases do not publish
per-image random seeds, so E02 results are explicitly exploratory and the full
confirmatory metadata gate cannot pass, irrespective of apparent signal.

## Environment

The reference environment is Python 3.10 with PyTorch 2.5.1/CUDA 12.4. On the
configured remote workstation it is available as the `fits-aigc` Conda
environment and the `Python (fits-aigc)` Jupyter kernel.

```bash
conda env create -f environment.yml
conda activate fits-aigc
python -m pip install -e .
```

The controlled E02b generation stage uses its separate `fits-e02b`
environment. Its PyTorch 2.6.0/CUDA 12.4 wheel is intentionally newer than the
base analysis environment because Transformers blocks unsafe `.bin` loading on
older PyTorch releases after CVE-2025-32434.

```bash
conda env create -f environment-e02b.yml
conda activate fits-e02b
```

For development and validation:

```bash
python -m pip install -e ".[dev]"
pre-commit install
ruff check .
ruff format --check .
pytest
```

Noiseprint is licensed for informational/nonprofit use and its official code
requires TensorFlow 1.2.1. E02 runs the locked upstream graph through a
TensorFlow 2.15 compatibility adapter in an isolated environment; the upstream
checkout is not modified.

```bash
conda env create -f environment-noiseprint.yml
conda activate fits-noiseprint
```

## Phase 0 through E02 CLI

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

E02 range-extracts only the frozen archive members, preserving and hashing the
original PNG bytes. GenImage is kept in independent 256x256 and 512x512 suites;
no geometry is normalized.

```bash
python -m gfits.cli prepare-e02-data \
  --config configs/e02.yaml \
  --data-root /mnt/data/jkl/FITS/datasets/e02/selected \
  --manifest /mnt/data/jkl/FITS/datasets/e02/source-manifest.json

python -m gfits.cli extract-residuals \
  --config configs/e02.yaml \
  --manifest /mnt/data/jkl/FITS/datasets/e02/source-manifest.json \
  --data-root /mnt/data/jkl/FITS/datasets/e02/selected \
  --upstream-root /mnt/data/jkl/FITS/checkpoints/prnu-python \
  --cache-root /mnt/data/jkl/FITS/cache/e02/residuals \
  --residual-manifest /mnt/data/jkl/FITS/cache/e02/residual-manifest.json \
  --extractor wavelet --extractor srm --extractor low_bit

# Run this command from the isolated fits-noiseprint environment.
python -m gfits.cli extract-residuals \
  --config configs/e02.yaml \
  --manifest /mnt/data/jkl/FITS/datasets/e02/source-manifest.json \
  --data-root /mnt/data/jkl/FITS/datasets/e02/selected \
  --upstream-root /mnt/data/jkl/FITS/checkpoints/prnu-python \
  --noiseprint-root /mnt/data/jkl/FITS/checkpoints/noiseprint-src \
  --cache-root /mnt/data/jkl/FITS/cache/e02/residuals \
  --residual-manifest /mnt/data/jkl/FITS/cache/e02/residual-manifest.json \
  --extractor noiseprint
```

```bash
python -m gfits.cli build-fingerprint-bank \
  --config configs/e02.yaml \
  --manifest /mnt/data/jkl/FITS/datasets/e02/source-manifest.json \
  --residual-manifest /mnt/data/jkl/FITS/cache/e02/residual-manifest.json \
  --bank-root /mnt/data/jkl/FITS/cache/e02/fingerprints \
  --bank-manifest /mnt/data/jkl/FITS/cache/e02/fingerprint-bank.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m gfits.cli score-pairs \
  --config configs/e02.yaml \
  --manifest /mnt/data/jkl/FITS/datasets/e02/source-manifest.json \
  --residual-manifest /mnt/data/jkl/FITS/cache/e02/residual-manifest.json \
  --bank-manifest /mnt/data/jkl/FITS/cache/e02/fingerprint-bank.json \
  --scores /mnt/data/jkl/FITS/artifacts/e02/scores.csv

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m gfits.cli evaluate-attribution \
  --config configs/e02.yaml \
  --manifest /mnt/data/jkl/FITS/datasets/e02/source-manifest.json \
  --data-root /mnt/data/jkl/FITS/datasets/e02/selected \
  --residual-manifest /mnt/data/jkl/FITS/cache/e02/residual-manifest.json \
  --scores /mnt/data/jkl/FITS/artifacts/e02/scores.csv \
  --output-dir /mnt/data/jkl/FITS/artifacts/e02/evaluation

python -m gfits.cli generate-report \
  --config configs/e02.yaml \
  --manifest /mnt/data/jkl/FITS/datasets/e02/source-manifest.json \
  --summary /mnt/data/jkl/FITS/artifacts/e02/evaluation/summary.json \
  --evaluation /mnt/data/jkl/FITS/artifacts/e02/evaluation/condition-evaluation.csv \
  --report reports/e02/E02_REPORT.md
```

Test rows are used only by the frozen evaluation; calibration rows never select
an extractor, aggregator, or test criterion. The completed exploratory screen
found 25/48 passing conditions: 10/16 on DM 256x256, 15/16 on GenImage 256x256,
and 0/16 on GenImage 512x512. The cross-suite statistical gate therefore failed.
Both public releases also omit per-image seed identity, so the confirmatory gate
failed and the registered decision is
`stop_confirmatory_claim_and_require_controlled_seed_provenance`.

## E02b controlled confirmation

E02b is the confirmatory replacement for the exploratory E02 screen. It is a
fully controlled Generator x Prompt x Seed x Resolution experiment with exact
model revisions, shared prompts, split-disjoint seeds, native geometry, and one
tensor-to-PNG writer. The full registered design contains 22,400 images across
cross-family and near-family suites. Configuration, code, and a clean Git
commit must be frozen before generation starts.

```bash
conda env create -f environment-e02b.yml
conda activate fits-e02b
pip install -e .

HF_ENDPOINT=https://hf-mirror.com python -m gfits.cli generate-e02b-shard \
  --config configs/e02b.yaml \
  --repository-root . \
  --data-root /mnt/data/jkl/FITS/datasets/e02b \
  --cache-root /mnt/data/jkl/FITS/checkpoints/huggingface \
  --fragment /mnt/data/jkl/FITS/manifests/e02b/fragments/sd14-512.json \
  --model-id sd14 --resolution 512 --device cuda:0 --batch-size 1

python -m gfits.cli merge-e02b-generation \
  --config configs/e02b.yaml --repository-root . \
  --data-root /mnt/data/jkl/FITS/datasets/e02b \
  --fragment-root /mnt/data/jkl/FITS/manifests/e02b/fragments \
  --output /mnt/data/jkl/FITS/manifests/e02b/generation.json

python -m gfits.cli apply-e02b-counterfactuals \
  --config configs/e02b.yaml --repository-root . \
  --generation-manifest /mnt/data/jkl/FITS/manifests/e02b/generation.json \
  --data-root /mnt/data/jkl/FITS/datasets/e02b \
  --derivative-root /mnt/data/jkl/FITS/datasets/e02b-counterfactuals \
  --output /mnt/data/jkl/FITS/manifests/e02b/counterfactuals.json
```

Representation extraction is sharded by extractor and condition, then merged
before scoring. Use `python -m gfits.cli <command> --help` for the arguments to
`extract-e02b-representations`, `merge-e02b-representations`, `score-e02b`,
`select-e02b-condition`, `evaluate-e02b`, and `generate-e02b-report`.

The score API is deliberately explicit: `paper_fits_ratio` and
`paper_fits_plus_c` mean the paper statistics; `median_control_ratio` is a
project extension; `gallery_complement_ratio` is only a relative closed-gallery
score and is never presented as nuisance calibration. Candidate-independent
software controls are represented by `NuisanceControlBank` and require at least
three donors disjoint from candidates and queries. `robust_zscore` and the
fit/apply nuisance-subspace API likewise accept only explicit independent
control inputs.

The E02b Gate requires both source suites, a two-way-bootstrap lower confidence
bound above chance, significant Rank-1 attribution, source breadth, stable
improvement with template count, and survival after the unified low-bit export
control. E03, E04, and E05 remain claim-gated until their registered
prerequisites pass; a failed E02b Gate stops or restricts the fixed-generator
fingerprint mainline instead of being bypassed.

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
reports/e02/           E02 signal-screen result and metadata boundary
artifacts/e02/         E02 manifests, scores, statistics, figures, and QA
reports/e02b/          E02b controlled report and 73-item audit traceability
artifacts/e02b/        E02b manifests, raw tables, statistics, and QA
```

The Phase 0 source audit is in
[`reports/phase0/LITERATURE_MATRIX.md`](reports/phase0/LITERATURE_MATRIX.md) and
[`reports/phase0/CODE_REPOSITORY_AUDIT.md`](reports/phase0/CODE_REPOSITORY_AUDIT.md).
The exact external revisions are in [`third_party/LOCK.json`](third_party/LOCK.json).
The E00 result and interpretation are in [`reports/e00/E00_REPORT.md`](reports/e00/E00_REPORT.md).
The E01 result and interpretation are in [`reports/e01/E01_REPORT.md`](reports/e01/E01_REPORT.md).
The E02 result and interpretation are in [`reports/e02/E02_REPORT.md`](reports/e02/E02_REPORT.md).

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
