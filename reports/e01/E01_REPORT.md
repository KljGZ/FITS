# Phase E01: real downstream-pipeline mechanism replication

Completion gate: **PASS**. Registered mechanism hypothesis: **PASS**. E02 is
unblocked, but E01 does not contain generator images and is not an AIGC detector
or generator-attribution result.

## Claim boundary and pre-registration

E01 asks one narrow question: when a query and several source templates have
passed through the same downstream pipeline, can different-source template
scores estimate pipeline-dependent background correlation well enough to make
the H0 distribution and its operating threshold more comparable across
pipelines?

The protocol was frozen in `configs/e01.yaml` at `2026-08-07T06:55:02Z`, before
the dataset download finished and before any real score was produced. The two
registered mechanism checks were:

1. FITS+ threshold coefficient of variation across pipelines is lower than raw.
2. FITS+ mean pairwise H0 Kolmogorov-Smirnov statistic is lower than raw.

These checks concern calibration alignment, not classification accuracy. The
completion gate separately checks data integrity, geometry, split isolation,
minimum evaluation counts, and the absence of G-FITS image transformations.

## Source and implementation audit

The experiment uses the official [VISION dataset](https://lesc.dinfo.unifi.it/VISION/)
and its native, Facebook High, Facebook Low, and WhatsApp image versions. The
dataset paper is Shullani et al., *VISION: a video and image dataset for source
identification*, EURASIP JIS 2017,
[DOI 10.1186/s13635-017-0067-2](https://doi.org/10.1186/s13635-017-0067-2).
The files are distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The repository
stores the byte-level manifest and derived scores, not the 329 MB image set.

Residual extraction and maximum-likelihood PRNU estimation use the unmodified
[`polimi-ispl/prnu-python`](https://github.com/polimi-ispl/prnu-python) source at
commit `91e1585a39287e26f9e770b71cf9124c35d9248d`. The canonical
`prnu/functions.py` SHA-256 is
`9c26c0c854283145218950c7a63f04ef894b7c1b735bb18249ac42d74408b45f`.
E00 had already cross-validated the matching primitives against this revision.

The division interpretation follows the CCS 2023 FITS formula supplement at
[`fits-matching.github.io`](https://fits-matching.github.io/). FITS+ is this
project's median multi-control extension; it is not a formula claimed by the
FITS authors. The VISION social-media experiment is also not a replication of
the periodic Adobe 128 x 128 export feature reported by Butora and Bas,
[*The Adobe Hidden Feature and its Impact on Sensor Attribution*](https://janbutora.github.io/assets/pdfs/adobe.pdf).
It is only a generic downstream-pipeline mechanism replication.

## Dataset and leakage controls

Five devices were fixed in advance: `D02_Apple_iPhone4s`, `D04_LG_D290`,
`D05_Apple_iPhone5c`, `D15_Apple_iPhone6`, and `D19_Apple_iPhone6Plus`.

| Pipeline | Official directory | Native decoded geometry | Files |
|---|---|---:|---:|
| Native | `nat` | 3264 x 2448 | 100 |
| Facebook High | `natFBH` | 2048 x 1536 | 100 |
| Facebook Low | `natFBL` | 1224 x 918 | 100 |
| WhatsApp | `natWA` | 1280 x 960 | 100 |

For each device and pipeline, indices 1--5 build the template, 21--25 form the
calibration set, and 41--50 form the test set. The same original and any of its
pipeline derivatives therefore remain in exactly one split. The final manifest
contains exactly 400 unique files; every byte SHA-256, RGB mode, and registered
dimension was rechecked immediately before scoring.

Pillow decodes each JPEG in its stored RGB geometry. G-FITS performs no resize,
crop, recompression, EXIF transpose, color conversion, or other image rewrite.
Wavelet extraction uses 4 levels and sigma 5.0 over the full frame. Each source
template averages five images using the locked upstream PRNU MLE path.

## Scores and evaluation

For query residual `q`, candidate template `g`, and eligible controls `h`, E01
uses signed aligned NCC internally and defines the non-negative raw score as

```text
A_g(q) = NCC(q, g)^2
B_single(q) = first score after sorting eligible control source IDs
B_median(q) = median_h A_h(q)
subtraction = A_g(q) - B_single(q)
FITS        = A_g(q) / B_single(q)
FITS+       = A_g(q) / B_median(q)
log-ratio   = log(A_g(q)) - log(B_median(q))
```

An eligible control is from the same pipeline but a different device, excluding
both the candidate and the true query source. The registered stabilizer is zero.
Thresholds are the higher empirical 95th percentile of calibration H0 only,
corresponding to target FAR 0.05. No threshold or extractor choice uses the test
set. Per pipeline and method there are 100 calibration H0, 200 test H0, and 50
test H1 rows; `scores.csv` contains all 1,500 candidate-query rows.

## Registered mechanism result

| Method | Threshold CV | Mean pairwise H0 KS |
|---|---:|---:|
| Raw | 0.602925 | 0.311667 |
| Subtraction | 0.617422 | 0.212500 |
| FITS, single control | 0.400342 | 0.071667 |
| FITS+, median controls | **0.311223** | **0.060000** |

FITS+ reduces threshold CV by 48.4% and mean pairwise H0 KS by 80.7% relative
to raw. Both registered inequalities pass. The H0 ECDFs support the intended
interpretation: raw score scale changes with the social-media pipeline, while
the median control ratio concentrates the bulk of all four H0 distributions
near one.

## Independent test performance

The alignment result does not imply improved attribution. Test operating points
and ranking metrics are:

| Method | Pipeline | FAR | TPR | AUROC | AP |
|---|---|---:|---:|---:|---:|
| Raw | Native | 0.030 | 0.980 | 0.9809 | 0.9841 |
| Raw | Facebook High | 0.035 | 0.800 | 0.8826 | 0.8682 |
| Raw | Facebook Low | 0.045 | 0.460 | 0.7326 | 0.6208 |
| Raw | WhatsApp | 0.065 | 0.840 | 0.9333 | 0.9008 |
| Subtraction | Native | 0.035 | 0.980 | 0.9906 | 0.9869 |
| Subtraction | Facebook High | 0.035 | 0.800 | 0.9050 | 0.8713 |
| Subtraction | Facebook Low | 0.045 | 0.460 | 0.7396 | 0.6134 |
| Subtraction | WhatsApp | 0.060 | 0.820 | 0.9258 | 0.8923 |
| FITS | Native | 0.070 | 0.960 | 0.9807 | 0.9152 |
| FITS | Facebook High | 0.090 | 0.500 | 0.8241 | 0.6074 |
| FITS | Facebook Low | 0.070 | 0.280 | 0.6662 | 0.3680 |
| FITS | WhatsApp | 0.035 | 0.540 | 0.8756 | 0.6703 |
| FITS+ | Native | 0.040 | 0.960 | 0.9783 | 0.9760 |
| FITS+ | Facebook High | 0.025 | 0.540 | 0.8596 | 0.7467 |
| FITS+ | Facebook Low | 0.025 | 0.180 | 0.6760 | 0.4661 |
| FITS+ | WhatsApp | 0.090 | 0.780 | 0.9071 | 0.7266 |

Across pipelines, mean FAR/TPR/AUROC are 0.04375/0.770/0.88235 for raw and
0.04500/0.615/0.85525 for FITS+. FITS+ therefore improves neither test TPR nor
ranking in this experiment. Facebook Low is the hardest condition: its raw TPR
is 0.46 and FITS+ TPR is 0.18. The single-control ratio is also less reliable at
the registered operating point, with mean FAR 0.06625 versus 0.045 for FITS+.

## Evidence for and against the mechanism

Supporting evidence:

- Both pre-registered calibration-alignment inequalities pass on held-out H0.
- FITS and FITS+ sharply reduce cross-pipeline H0 KS relative to raw and
  subtraction, so the effect is specific to ratio normalization.
- FITS+ outperforms a deterministic single control on threshold CV, H0 KS, and
  mean observed FAR, consistent with robust aggregation reducing denominator
  variability.
- All 400 inputs, 20 fingerprints, 1,500 scores, and 16 threshold evaluations
  retain provenance and hashes.

Limiting or opposing evidence:

- FITS+ loses 0.155 mean TPR and 0.0271 mean AUROC relative to raw. Aligning the
  null does not preserve a weak H1 signal under aggressive processing.
- The ratio distributions are heavy-tailed because squared NCC control scores
  can approach zero. A median denominator reduces but does not eliminate this
  behavior; no stabilizer was fitted in E01.
- Observed FAR still ranges from 0.025 to 0.090 for FITS+. With only 200 test H0
  rows per pipeline, E01 does not claim exact FAR control or confidence bounds.
- VISION contains five real cameras and historical social-media derivatives,
  not generators, modern AIGC services, Adobe exports, screenshots, or learned
  post-processing pipelines.
- Candidate and control scores share a query, so their dependence is useful for
  nuisance calibration but violates any interpretation as independent samples.
- Squared NCC deliberately removes sign. E01 does not establish that this is the
  optimal statistic; alternate residuals and aggregators belong to E02/E05.

The strongest defensible E01 conclusion is therefore: same-pipeline median
controls can normalize a real pipeline-dependent PRNU score background, but
this normalization alone can reduce source-discrimination power. G-FITS must
demonstrate a stable generator H1 signal in E02 before downstream calibration
can be claimed useful for AIGC attribution.

## Reproducibility evidence

- `artifacts/e01/vision-manifest.json`: 400 URLs, byte sizes, SHA-256 values,
  decoded modes/geometries, split IDs, and original-group IDs.
- `artifacts/e01/fingerprints.csv`: all 20 template provenance rows and cached
  fingerprint hashes.
- `artifacts/e01/scores.csv`: all 1,500 calibration/test candidate rows and raw,
  control, subtraction, FITS, FITS+, and log-ratio values.
- `artifacts/e01/threshold-evaluation.csv`: all 16 calibration thresholds and
  independent test metrics.
- `artifacts/e01/summary.json`: configuration/source/artifact hashes, completion
  checks, aggregate metrics, and registered decisions.
- `artifacts/e01/h0-h1-histograms.png`, `h0-ecdf.png`, and
  `calibration-diagnostics.png`: distributions and alignment diagnostics.
- `artifacts/e01/remote-validation.json` and `pytest-junit.xml`: remote runtime
  versions and the final 23-test result.

Local Windows and remote Linux quality gates both pass `ruff check`,
`ruff format --check`, and all 23 tests. The remote environment is Python
3.10.20 with NumPy 1.26.4, Pillow 11.0.0, PyWavelets 1.7.0, SciPy 1.14.1,
scikit-image 0.24.0, and scikit-learn 1.5.2. The host exposes eight RTX 4090
GPUs, but E01's locked PRNU implementation is CPU-based and makes no GPU speed
or determinism claim.
