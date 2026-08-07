# Phase 0 code repository audit

Audit date: 2026-08-06. Default-branch heads were resolved through the GitHub
API and copied to `third_party/LOCK.json`. “Official” below means the repository
README or GitHub description claims association with the named paper; it is not
an independent provenance guarantee. No external repository is cloned or
executed in Phase 0.

| Repository | Locked revision | Claimed relationship / install signal | License signal | Integration decision and risk |
|---|---:|---|---|---|
| [polimi-ispl/prnu-python](https://github.com/polimi-ispl/prnu-python) | `91e1585a3928` | Python port of PRNU extractor; requirements present | MIT (`LICENSE.md`) | Preferred E00 cross-check through an adapter. Confirm numerical conventions for signed PCE. |
| [janbutora/prnu-python](https://github.com/janbutora/prnu-python) | `645b418154d3` | Modified PRNU port for Adobe-pattern removal | MIT (`LICENSE.md`) | E01/E04 comparator only; do not replace the untouched reference silently. |
| [janbutora/adobe-detector](https://github.com/janbutora/adobe-detector) | `4474031b1481` | Author-linked Adobe pattern detector; minimal usage example | MIT (`LICENSE.md`) | Candidate E01 mechanism tool; architecture/content assumptions must be logged. |
| [grip-unina/noiseprint](https://github.com/grip-unina/noiseprint) | `c06034eedc92` | Paper authors' project; CPU/GPU requirement files | GitHub API `NOASSERTION`; `LICENSE.txt` present | Manual license-text review before reuse; isolate legacy dependencies. |
| [ningyu1991/GANFingerprints](https://github.com/ningyu1991/GANFingerprints) | `5a280e0e6c28` | README/description says official ICCV 2019 TensorFlow implementation; requirements/training code | API `NOASSERTION`; `LICENSE` present | Adapter/container likely needed for legacy TensorFlow. Do not treat learned classification features as pure residual evidence. |
| [grip-unina/DMimageDetection](https://github.com/grip-unina/DMimageDetection) | `745ad9e1eee8` | README says official; `environment.yml` present | Apache-2.0 | Preferred E02 data/method source after dataset checksum and split reconstruction. |
| [hongsong-wang/LIDA](https://github.com/hongsong-wang/LIDA) | `fefbcaecb4de` | README links CVPR 2026 paper and evaluation flow | Apache-2.0 | E06 attribution baseline; external datasets/weights need their own manifests and terms. |
| [jumpycat/GenSign](https://github.com/jumpycat/GenSign) | `0d2953d68982` | CVPR 2026 README; training/evaluation/checkpoints marked available; DAIR is separate | MIT | E02/E06 learned-signature comparator. Pin/audit DAIR separately if activated. |
| [GenImage-Dataset/GenImage](https://github.com/GenImage-Dataset/GenImage) | `746781bfa446` | Benchmark README; contains multiple imported detector/generator trees | API `NOASSERTION`; `License` present | Repository is a fork and has vendoring/license-boundary risk. Use dataset manifest and method-specific adapters, not wholesale import. |
| [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) | `76a0e3e60a8a` | CVPR 2023 project; train/eval scripts | MIT | Detection baseline only. Freeze preprocessing and CLIP weight identity. |
| [ZhendongWang6/DIRE](https://github.com/ZhendongWang6/DIRE) | `1f89fff15f0e` | Description says official ICCV 2023 implementation; requirements/train/test present | API `NOASSERTION`; no license file found | Archived repository. Reuse/redistribution blocked pending license clarification; environment risk is high. |
| [chuangchuangtan/NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) | `781ced3f7ca2` | CVPR 2024 README; requirements/train/test present | API `NOASSERTION`; no license file found | Execution for comparison may be separately reviewed; no code incorporation now. |
| [jonasricker/aeroblade](https://github.com/jonasricker/aeroblade) | `50fc2b22c929` | CVPR 2024 README; requirements/setup present | API `NOASSERTION`; no license file found | License-blocked for incorporation. Model downloads and autoencoder versions must be pinned. |
| [beibuwandeluori/DRCT](https://github.com/beibuwandeluori/DRCT) | `01aa7d0b6de9` | Description says official ICML 2024 code; requirements/train present | API `NOASSERTION`; no license file found | License-blocked for incorporation; large dataset and reconstruction model need independent checksums. |
| [shilinyan99/AIDE](https://github.com/shilinyan99/AIDE) | `6725b710d5c4` | ICLR 2025 README; requirements/checkpoints/data instructions | MIT for code; README gives Chameleon separate academic-only/non-commercial terms | Code and data permissions must remain distinct. Useful sanity-check baseline, not attribution evidence. |
| [grip-unina/B-Free](https://github.com/grip-unina/B-Free) | `c6a9f898782f` | README says official CVPR 2025; code/requirements/data links | API `NOASSERTION`; `LICENSE.txt` present | Manual license review. High-priority paired-content control and calibration comparator. |
| [ductai199x/Forensic-Self-Descriptions-CVPR25](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25) | `50f2eae06efd` | README says official CVPR 2025; `pyproject.toml` present | API `NOASSERTION`; `LICENSE` present | Candidate E07 adapter after license text and per-image optimization cost review. |
| [Ekko-zn/SDAIE](https://github.com/Ekko-zn/SDAIE) | `532f0fe3773e` | README says official TPAMI implementation; `pyproject.toml` and weight/data links | API `NOASSERTION`; no license file found | License-blocked for incorporation. EXIF and dataset leakage controls are mandatory for E07. |
| [Ekko-zn/AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark) | `795ba629d645` | Living benchmark aggregating multiple methods; requirements/train/test paths | API `NOASSERTION`; no license file found | Do not import wholesale. Audit each included method, preprocessing path, and weight license separately. |

## Non-GitHub or unresolved resources

- **FITS project:** the CCS 2023 paper and project page were identified, but no
  public code repository was located. E00 must therefore be an explicitly
  independent implementation, with equations/tests traceable to the paper.
- **Binghamton camera fingerprint package:** the project page advertises MATLAB
  and Python code and currently states CC BY-NC 4.0 research terms. Access and
  redistribution must follow that page; it is not represented as a Git lock.
- **Model weights and datasets:** no repository code license is assumed to cover
  them. Every downloaded archive must receive a separate manifest, SHA-256, URL,
  date, and terms record before experiments.

## Reproducibility risks that affect later gates

1. Several repositories have no SPDX-resolvable license; pinning them does not
   authorize copying or modification.
2. Historical projects span TensorFlow and incompatible PyTorch/CUDA stacks.
   The Phase 0 Conda environment is the native G-FITS environment, not a promise
   that all baselines run inside it unchanged.
3. Many baselines bundle preprocessing. Any resize, crop, JPEG conversion, color
   conversion, or residual normalization must be surfaced in the adapter config.
4. Upstream default branches can move after this audit. Only the full 40-byte
   revisions in `LOCK.json` are admissible for a reported experiment.
