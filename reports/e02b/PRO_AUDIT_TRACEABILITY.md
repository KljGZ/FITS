# Pro Audit: 73-Item Implementation Traceability

This is the one-to-one implementation register for the post-E02 Pro audit.
`READY` means code and protocol exist but formal execution must wait for the
clean committed E02b baseline. `GATED` is an intentional stage dependency, not
an omitted recommendation. E02 remains frozen exploratory evidence.

| ID | State | Requirement and concrete evidence |
|---:|---|---|
| 001 | IMPLEMENTED | Formal commands call `repository_state` and reject an uncommitted or dirty tree; source hashes and commit are embedded per sample. |
| 002 | IMPLEMENTED | `configs/e02b.yaml` marks frozen E02 test scores audit-only; E02b selection reads only controlled calibration rows. |
| 003 | IMPLEMENTED | README and the frozen E02 report retain the exploratory label and failed confirmatory claim boundary. |
| 004 | IMPLEMENTED | The registered order is E02b then E03, E04, and E05; later claims are Gate-controlled. |
| 005 | IMPLEMENTED | Score columns and APIs explicitly distinguish signed NCC, signed energy, raw PCE, gallery contrast, FITS ratios, robust z-score, and nuisance-subspace residual energy. |
| 006 | IMPLEMENTED | `paper_fits_ratio` is the unstabilized candidate/control statistic and fails on a zero denominator. |
| 007 | IMPLEMENTED | `paper_fits_plus_c` implements `(A+C)/(B+C)` with an explicit non-negative calibration constant. |
| 008 | IMPLEMENTED | `gallery_complement_ratio` is documented and reported only as a closed-gallery relative score, never as a software control. |
| 009 | IMPLEMENTED | Parametric tests verify gallery-complement Top-1 monotonicity as candidate count changes. |
| 010 | IMPLEMENTED | Median/robust nuisance estimators require at least three independent controls; `NuisanceControlBank` exposes mean and median for comparison. |
| 011 | IMPLEMENTED; E05 GATED | `NuisanceControlBank` has an independent pipeline/geometry/source contract for future software controls. |
| 012 | IMPLEMENTED; E05 GATED | Control donors must be disjoint from candidate sources, query sources, and query image hashes. |
| 013 | IMPLEMENTED; E05 GATED | One nuisance denominator is shared across every candidate and asserted by `assert_candidate_independent_denominator`. |
| 014 | IMPLEMENTED | The former SRM shorthand is renamed `fixed_three_kernel_high_pass_residual_bank`; outputs never claim to be full SRM. |
| 015 | IMPLEMENTED | `prnu_mle` is rejected for every non-wavelet residual. |
| 016 | IMPLEMENTED | Non-wavelet intensity weighting is separately named `intensity_weighted`. |
| 017 | IMPLEMENTED | Query-intensity modulation is enabled only by calibration-only binned variance, regression, and Breusch-Pagan evidence. |
| 018 | IMPLEMENTED | Global NCC and channel-mean NCC are separate registered scorers. |
| 019 | IMPLEMENTED | Channel whitening is fit on the calibration split only, serialized, and rejected on other splits. |
| 020 | IMPLEMENTED | Noiseprint participates in common-template and source-delta decomposition like the other spatial residuals. |
| 021 | READY | `build_generation_plan` creates the complete Generator x Prompt x Seed x Resolution factorial, including registered auxiliary crossing cells. |
| 022 | READY | Independent cross-family and near-family four-source suites are registered. |
| 023 | READY | Every source uses the identical versioned 200-prompt factorial. |
| 024 | READY | Each sample records prompt, seed, resolution, revision inventories, VAE/MOVQ hash, scheduler, writer, tensor hash, output hash, code commit, and clean-tree state. |
| 025 | READY | All pipelines return float tensors to one `clip -> rint -> uint8 -> Pillow PNG` writer; writer version mismatch fails closed. |
| 026 | READY | All seven unique registered models generate at 256, 512, 768, and 1024 pixels. |
| 027 | READY | No resize or cross-resolution normalization is performed during generation or primary matching; native geometry is verified. |
| 028 | READY | Template counts 1, 3, 5, 10, 20, and 50 are registered and rebuilt for the curve. |
| 029 | READY | The fixed spatial branch covers wavelet, the renamed three-kernel bank, low-bit planes, and Noiseprint. |
| 030 | READY | Eight separate signatures are implemented: 2-D power, phase, radial power, autocorrelation, cepstrum, SRM-style co-occurrence, patch Gram/covariance, and low/high partition. |
| 031 | READY | Mean/median spatial templates are evaluated separately from covariance/Gram signatures; the names are not collapsed into a single fingerprint claim. |
| 032 | READY | A secondary calibration-then-test native-resolution profile covers all four resolutions and cannot affect the E02b Gate. |
| 033 | READY | Canonical uniform re-quantization/export is deterministic and records byte/pixel identity. |
| 034 | READY | Deterministic random +/-1 LSB perturbation is registered per sample. |
| 035 | READY | Six-bit quantization is registered as an independent counterfactual. |
| 036 | READY | PNG level-9 re-encoding records unchanged pixels separately from changed bytes. |
| 037 | READY | Per-source AUROC and per-source Rank-1 expose single-source dominance. |
| 038 | READY | Candidate H0 mean/std and prediction frequency expose universally favored templates. |
| 039 | READY | Pooled pairwise AUROC is reported but is not the only inferential unit. |
| 040 | READY | Macro and per-source AUROC are primary source-level summaries. |
| 041 | READY | Rank-1, Rank-5, mAP, and MRR are computed from full query galleries. |
| 042 | READY | Confusion, template bias, and template-number curves are emitted as raw CSV evidence. |
| 043 | READY | Templates are rebuilt after within-source image resampling in every template-bootstrap draw. |
| 044 | READY | Query bootstrap resamples shared `prompt_id` blocks across all sources. |
| 045 | READY | The confirmatory confidence interval is the joint two-way template-image and prompt-group bootstrap. |
| 046 | READY | Prompt-block Rank-1/margin, whole-template-label, and hierarchical family/prompt permutations are confirmatory; sign-flip is auxiliary only; Holm controls FWER. |
| 047 | IMPLEMENTED | Reports explicitly prohibit treating the exploratory 25/48 E02 conditions as independent replications. |
| 048 | E03 GATED | Common, family, model, and configuration hierarchy analysis can execute only after E02b passes. |
| 049 | E03 GATED | Hierarchical attribution and variance decomposition are reserved for E03. |
| 050 | E03 GATED | Calibration-H0 z-scoring for hierarchy decisions is reserved for E03. |
| 051 | E03 GATED | E03 must test whether a generator term `G_g` is stable. |
| 052 | E03 GATED | E03 must separately test configuration-dependent `G_{g,rho}` rather than assuming a fixed model fingerprint. |
| 053 | E04 GATED | Downstream software/export/compression pipelines are not introduced before source evidence and hierarchy prerequisites pass. |
| 054 | E04 GATED | E04 will use paired before/after score deltas on identical source images. |
| 055 | E04 GATED | Same-pipeline correlations are an E04 registered output. |
| 056 | E04 GATED | E04 will estimate pipeline-specific H0 location and scale shifts. |
| 057 | E04 GATED | Pipeline pollution is separated from source-signal attenuation. |
| 058 | E04 GATED | Matched and mismatched pipeline controls are required before a software-noise claim. |
| 059 | API IMPLEMENTED; E05 GATED | Candidate statistic `A` has explicit scorer semantics. |
| 060 | API IMPLEMENTED; E05 GATED | Control statistic `B` must come from an independent, source-disjoint nuisance bank. |
| 061 | API IMPLEMENTED; E05 GATED | Paper FITS+ uses exactly `(A+C)/(B+C)`. |
| 062 | API IMPLEMENTED; E05 GATED | Tests assert one denominator across candidates, preserving within-query candidate rank when the numerator transform is monotone. |
| 063 | E05 GATED | NullShift comparison awaits E04 pipeline H0 estimates. |
| 064 | E05 GATED | Thresholds and `C` must be chosen by nested/calibration cross-validation without test access. |
| 065 | E05 GATED | KS and Wasserstein H0 alignment are registered future calibration diagnostics. |
| 066 | E05 GATED | H1 retention must be reported beside H0 alignment to prevent apparent gains from signal destruction. |
| 067 | READY | The E02b Gate requires all six registered checks: two suites, two-way CI, Rank-1 permutation, source breadth, stable N trend, and unified-export low-bit survival. |
| 068 | E03 GATED | The E03 Gate cannot be evaluated until E02b passes and E03 hierarchy outputs exist. |
| 069 | E04 GATED | The E04 Gate cannot be evaluated until E03 identifies the admissible source level. |
| 070 | E05 GATED | The E05 Gate cannot be evaluated until E04 validates a nuisance term and matched controls. |
| 071 | IMPLEMENTED | Machine-readable and Markdown reports lead with source-level results and raw evidence hashes. |
| 072 | IMPLEMENTED | Reports enumerate forbidden universal-fingerprint, generator-independence, software-noise-removal, and universal-detector claims. |
| 073 | IMPLEMENTED | A failed registered Gate stops or restricts the mainline; later stages cannot silently bypass it. |
