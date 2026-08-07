# Phase E00: numerical mechanism validation

Gate decision: **PASS**. E01 is unblocked, but no E01 experiment is included in
this phase or commit.

## Scope and claim boundary

E00 tests only the numerical mechanism required before real-image work. It
implements the matching statistics, cross-validates the PRNU-compatible
primitives, and asks whether a controlled nuisance ratio behaves as expected on
synthetic residuals. It does not establish that modern generators have stable
fingerprints, that real software pipelines create shared correlation, or that
G-FITS detects AIGC images.

The configuration, development seed, independent gate seed, and thresholds were
frozen in `configs/e00.yaml` at `2026-08-07T06:23:00Z`. The development profile
ran at `06:26:14Z`; the gate profile first ran at `06:26:23Z`. No gate threshold
was selected or changed from gate output. The gate artifact was regenerated at
`06:39:29Z` only to add the project-required provenance columns to both raw CSV
schemas; the numerical implementation, seed, configuration, metrics, and gate
decision were unchanged.

## Definitions and source audit

The CCS 2023 FITS residual model is

```text
K_hat = alpha K + beta S + xi,
```

where `K` is the source fingerprint, `S` is a shared software component, and
`xi` is residual noise. The authors' formula supplement defines division-form
FITS for CC, NCC, and PCE as the candidate statistic divided by a same-software,
different-device control statistic (equations 45, 50, and 55). For example,

```text
FITS_NCC(Q, R, Z) = NCC(Q, R) / NCC(Q, Z).
```

E00 implements:

```text
CC(x,y)  = sum((x - mean(x)) * (y - mean(y)))
NCC(x,y) = CC(x,y) / (||x - mean(x)||_2 ||y - mean(y)||_2)
PCE      = sign(peak) peak^2 / mean(floor_excluding_neighborhood^2)
FITS     = candidate_score / single_control_score
FITS+    = candidate_score / median(multiple_control_scores)
log-ratio = log(candidate_score) - log(median_control_score)
```

`FITS+` is project terminology for the multi-control robust extension. It is not
attributed to the CCS paper. The paper authors did not publish an executable
implementation located by this audit; their CC0 project repository is locked as
a formula supplement at commit
`80ecd4fa177efd1da2296149b302b8ea9f2912fc`.

The primary G-FITS PCE uses an inclusive circular exclusion and divides by the
true number of floor samples, `mn - |N|`. A separate compatibility function
exactly preserves locked `prnu-python` behavior: an upstream half-open slice and
a zero-filled full-map energy mean. Keeping the paths separate avoids silently
turning an implementation convention into the project's mathematical
definition.

The ACM full-text endpoint presented an automated security verification page in
this environment. Equations 38--56 were therefore verified against the authors'
official formula supplement; CC/PCE implementation conventions were checked
against the locked public `prnu-python` source.

## Cross-validation against locked PRNU code

Reference: `polimi-ispl/prnu-python` commit
`91e1585a39287e26f9e770b71cf9124c35d9248d`. The checkout was read-only and was
not vendored or modified in the project.

| Check | Cases | Maximum error | Result |
|---|---:|---:|---|
| FFT cross-correlation map | 4 | `3.814697265625e-06` absolute | PASS |
| Compatibility signed PCE value | 4 | `0.0` | PASS |
| Compatibility PCE floor energy | 4 | `0.0` | PASS |

## Synthetic experiment

All patterns are deterministic zero-mean, unit-RMS arrays of shape `32 x 32`.
There are 64 independent source patterns and one nuisance pattern per residual
model. A reference averages 8 samples. Each model has 300 H0 and 300 H1 trials,
9 controls per hypothesis trial, and 300 fixed-candidate variance replications
at `nZ = 1, 3, 9, 27`.

```text
additive:       R = gamma G + beta S + epsilon
multiplicative: R = (1 + gamma G)(1 + beta S) - 1 + epsilon
nonadditive:    R = tanh(gamma G + beta S) + epsilon
```

The frozen values are `gamma=0.45`, `beta=0.80`, and `noise_std=0.85`. H0 uses a
different query source from the candidate source; H1 reuses the candidate
source. Controls are same-pipeline, different-source references. No images,
resizing, cropping, compression, color conversion, learned model, or real data
are involved.

Pre-registered gates were: H0 median FITS+ in `[0.90, 1.10]`, H1 median FITS+
at least `1.15`, final/initial log-ratio variance at most `0.55`, and log-log
variance slope at most `-0.15`.

| Residual model | H0 median FITS+ | H1 median FITS+ | Variance ratio `nZ=27/1` | Log slope | Result |
|---|---:|---:|---:|---:|---|
| Additive | 1.000381 | 1.316515 | 0.058299 | -0.864132 | PASS |
| Multiplicative | 0.997287 | 1.511004 | 0.082531 | -0.763636 | PASS |
| Nonadditive | 1.001797 | 1.355215 | 0.056758 | -0.867993 | PASS |

All model-level checks passed. The independent Linux/RTX 4090 reproduction had
the same 1,800 discrete rows; the maximum numeric row delta was `1.11e-15`, and
the maximum delta across 30 summary metrics was `1.33e-15`.

## Evidence for and against the mechanism

Supporting evidence:

- Same-nuisance H0 ratios center tightly around one in all three registered
  models.
- Same-source H1 medians remain above both H0 and the frozen 1.15 threshold.
- Increasing the number of controls reduces log-ratio estimator variance by
  more than 91% in every model.
- An independent OS/environment reproduces every decision and metric to
  approximately machine precision.

Limiting or opposing evidence:

- Shared nuisance correlation is deliberately built into these simulations;
  E00 cannot show that any real platform or editor produces it.
- The multiplicative and `tanh` models only falsify the narrow claim that the
  code works solely for a linear sum. They do not span real decoder, VAE,
  compression, resize, screenshot, or content interactions.
- Positive NCC is guaranteed by the registered signal regime. Near-zero or
  signed control statistics can destabilize ratios and require later robust
  statistics, abstention, or a training-only stabilizer.
- No conclusion about generator fingerprint existence, resolution stability,
  known-source attribution accuracy, open-world detection, or real-image false
  alarms is permitted from E00.

## Reproducibility evidence

- `artifacts/e00/synthetic_scores.csv`: all 1,800 H0/H1 raw score rows.
- `artifacts/e00/variance_by_nz.csv`: all 3,600 control-variance rows.
- `artifacts/e00/summary.json`: gate definitions, hashes, and decisions.
- `artifacts/e00/diagnostics.png`: distributions, medians, and variance curves.
- `artifacts/e00/prnu-cross-validation.json`: locked upstream comparison.
- `artifacts/e00/remote-validation.json`: Windows/Linux numerical comparison and
  remote environment.
- `artifacts/e00/pytest-junit.xml`: remote test results (`18 passed`).

The final quality gate is `ruff check`, `ruff format --check`, and `pytest` on
both the local development environment and the remote `fits-aigc` Conda
environment. The configured remote exposes eight NVIDIA GeForce RTX 4090 GPUs;
E00 itself is CPU numerical validation and does not claim GPU acceleration.
