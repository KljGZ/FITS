# Phase 0 stage report

Status: **PASS**
Phase configuration: `configs/phase0.yaml`
Third-party revision record: `third_party/LOCK.json`

## Delivered scope

- installable `src/` package and Python 3.10 project metadata;
- `ruff`, `pytest`, `pre-commit`, and GitHub Actions quality gates;
- byte-preserving image manifest construction and SHA-256 verification CLI;
- path-traversal, symlink, tamper, unexpected-file, and lock-format tests;
- immutable revisions for all 19 audited priority repositories;
- literature matrix and code/license/reproduction audit.

No FITS score, generator fingerprint, dataset measurement, plot, or performance
metric is produced in Phase 0. The later command interfaces are intentionally not
stubbed because a successful no-op would create misleading evidence.

## Gate definition

Phase 0 passes only when all of the following are true on the reference remote
environment:

1. `ruff check .` passes;
2. `ruff format --check .` passes;
3. `pytest --junitxml=artifacts/phase0/pytest-junit.xml` passes;
4. the package installs in the `fits-aigc` environment;
5. a manifest build/strict verification smoke test succeeds;
6. Git worktree changes are limited to Phase 0 plus the pre-existing untracked
   user file `main.py`.

## Evidence for the research hypothesis (literature only)

- FITS and the Adobe hidden-pattern study show that software pipelines can
  create shared residual correlation and false source matches in camera work.
- GAN/diffusion attribution studies motivate a controlled search for
  source-dependent low-level traces.
- Bias-controlled detection work independently supports paired content and
  preprocessing controls.

These items justify experiments; they do not validate G-FITS.

## Evidence against or limiting the research hypothesis

- Recent camera-identification/AIGC analysis reports resolution dependence,
  non-additive behavior, and standard-PCE false positives.
- Generator, VAE/decoder, resolution, content, and software pipeline may be
  inseparable in observational datasets.
- A useful binary detector need not provide a stable source fingerprint, and a
  closed-set attribution score does not establish unknown-generator detection.
- If E04 finds no cross-source, same-pipeline shared correlation, FITS-style
  calibration has no demonstrated nuisance to remove.

## Raw artifacts

The reference JUnit XML is `artifacts/phase0/pytest-junit.xml` (SHA-256
`46614eba7352b6bc4c1676ba127b0897f5ab96ca9d61eafe51839145da61ac37`).
The machine-readable environment and gate summary is
`artifacts/phase0/remote-validation.json`. CSV/Parquet result tables and figures
are not applicable because Phase 0 contains no numerical experiment.

## Validation results

The staged source snapshot with Git tree
`f8281374aaaee7cfcb198813d3e6d1ae2a5d0436` was copied to a fresh remote
temporary directory and validated at 2026-08-07T05:04:39Z. Evidence/report files
were then updated; no source or configuration logic changed after that run.

- Remote: Linux 5.15, Python 3.10.20, PyTorch 2.5.1+cu124, CUDA 12.4,
  `torch.cuda.is_available() == True`, NVIDIA GeForce RTX 4090.
- `ruff check .`: pass.
- `ruff format --check .`: pass.
- `pytest`: 9 passed in 0.19 s.
- Installed-package manifest smoke test: build 1 record, strict verify 1 record,
  zero issues and zero unexpected paths.
- Local supplemental gate: Python 3.12.13, 9 tests passed; all pre-commit hooks
  passed.
- Online lock audit: all 19 locked revisions matched the audited default-branch
  heads; zero mismatches.
- The pre-existing untracked `main.py` was neither edited nor staged.

## Gate decision

**PASS.** Phase 0 infrastructure is reproducible on the reference remote and the
recorded evidence satisfies the gate above. This permits work on Phase E00 only;
it provides no evidence that the G-FITS scientific hypothesis is true.
