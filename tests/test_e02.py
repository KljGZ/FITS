from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gfits.e02 import (
    _aggregate_fingerprint,
    _holm_adjust,
    _low_bit_residual,
    _paired_margins,
    _signed_ncc,
    _srm_residual,
)
from gfits.e02_data import _select_suite, load_e02_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "e02.yaml"
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "e02"


@dataclass(frozen=True)
class _Member:
    filename: str

    def is_dir(self) -> bool:
        return False


def test_e02_protocol_is_frozen_and_complete() -> None:
    config = load_e02_config(CONFIG_PATH)

    assert config["selection"]["splits"] == {
        "template": 12,
        "calibration": 12,
        "test": 20,
    }
    assert config["completion_gate"]["expected_selected_file_count"] == 440
    assert config["completion_gate"]["expected_condition_count"] == 48
    assert set(config["extractors"]) == {"wavelet", "srm", "low_bit", "noiseprint"}
    assert config["aggregators"]["names"] == [
        "mean",
        "prnu_mle",
        "median",
        "trimmed_mean",
    ]
    assert not any(
        config["selection"]["transformations"][name]
        for name in ("resize", "crop", "recompress", "exif_transpose", "color_space_conversion")
    )
    checkpoint_hashes = config["extractors"]["noiseprint"]["checkpoint_sha256"]
    assert set(checkpoint_hashes) == {
        "model.data-00000-of-00001",
        "model.index",
        "model.meta",
    }
    assert all(len(value) == 64 for value in checkpoint_hashes.values())


def test_shared_prompt_selection_is_deterministic_and_split_disjoint() -> None:
    suite = {
        "prompt_policy": "shared_across_sources",
        "sources": {
            "source_a": {"member_prefix": "a/", "prompt_regex": r"^(?P<prompt>\d+)_a\.png$"},
            "source_b": {"member_prefix": "b/", "prompt_regex": r"^(?P<prompt>\d+)_b\.png$"},
            "source_c": {"member_prefix": "c/", "prompt_regex": r"^(?P<prompt>\d+)_c\.png$"},
        },
    }
    members = [
        _Member(f"{prefix}/{prompt}_{prefix}.png")
        for prefix in ("a", "b", "c")
        for prompt in range(10)
    ]
    splits = {"template": 2, "calibration": 2, "test": 3}

    first = _select_suite("dataset", "suite", suite, members, splits, 17)
    second = _select_suite("dataset", "suite", suite, members, splits, 17)

    assert first == second
    assert len(first) == 21
    prompt_sets = {
        split: {row["prompt_id"] for row in first if row["split"] == split} for split in splits
    }
    assert not (prompt_sets["template"] & prompt_sets["calibration"])
    assert not (prompt_sets["template"] & prompt_sets["test"])
    assert not (prompt_sets["calibration"] & prompt_sets["test"])
    assert all(
        len({row["source_id"] for row in first if row["prompt_id"] == prompt}) == 3
        for prompt in set.union(*prompt_sets.values())
    )


def test_fixed_residual_extractors_preserve_native_geometry() -> None:
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    original = image.copy()

    srm = _srm_residual(image)
    low_bit = _low_bit_residual(image, 2)

    assert srm.shape == (8, 8, 3)
    assert low_bit.shape == image.shape
    assert srm.dtype == np.float32
    assert low_bit.dtype == np.float32
    np.testing.assert_array_equal(image, original)
    assert low_bit.min() >= -0.5
    assert low_bit.max() <= 0.5


def test_fingerprint_aggregators_are_exact_on_small_arrays() -> None:
    residuals = np.asarray([[[1.0]], [[2.0]], [[100.0]], [[3.0]], [[4.0]]], dtype=np.float32)
    intensities = np.asarray([[[1.0]], [[1.0]], [[2.0]], [[1.0]], [[1.0]]], dtype=np.float32)
    config = {
        "aggregators": {
            "trimmed_mean_fraction": 0.2,
            "prnu_mle_denominator_epsilon": 1.0e-8,
        }
    }

    assert _aggregate_fingerprint(residuals, intensities, "mean", config).item() == 22.0
    assert _aggregate_fingerprint(residuals, intensities, "median", config).item() == 3.0
    assert _aggregate_fingerprint(residuals, intensities, "trimmed_mean", config).item() == 3.0
    expected_mle = (1.0 + 2.0 + 200.0 + 3.0 + 4.0) / (1.0 + 1.0 + 4.0 + 1.0 + 1.0)
    assert np.isclose(
        _aggregate_fingerprint(residuals, intensities, "prnu_mle", config).item(),
        expected_mle,
    )


def test_paired_margin_and_holm_adjustment() -> None:
    rows = [
        {"query_id": "q1", "raw_score": "0.8", "label_match": True},
        {"query_id": "q1", "raw_score": "0.2", "label_match": False},
        {"query_id": "q1", "raw_score": "0.4", "label_match": False},
        {"query_id": "q2", "raw_score": "0.5", "label_match": True},
        {"query_id": "q2", "raw_score": "0.1", "label_match": False},
        {"query_id": "q2", "raw_score": "0.3", "label_match": False},
    ]

    np.testing.assert_allclose(_paired_margins(rows), [0.5, 0.3])
    np.testing.assert_allclose(_holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_signed_ncc_preserves_all_three_channel_elements() -> None:
    left = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    right = np.flip(left, axis=1).copy()

    assert _signed_ncc(left, right) == _signed_ncc(left.reshape(2, -1), right.reshape(2, -1))


def test_e02_frozen_artifacts_and_registered_decision() -> None:
    summary = json.loads(
        (ARTIFACT_ROOT / "evaluation" / "summary.json").read_text(encoding="utf-8")
    )
    validation = json.loads((ARTIFACT_ROOT / "remote-validation.json").read_text(encoding="utf-8"))

    assert summary["completion_gate"]["passed"] is False
    assert summary["statistical_signal_gate"] == {
        "passed": False,
        "passing_conditions_by_suite": {
            "dm_256": 10,
            "genimage_256": 15,
            "genimage_512": 0,
        },
    }
    assert summary["mainline_decision"] == (
        "stop_confirmatory_claim_and_require_controlled_seed_provenance"
    )
    for filename, expected in summary["artifacts"].items():
        observed = hashlib.sha256(
            (ARTIFACT_ROOT / "evaluation" / filename).read_bytes()
        ).hexdigest()
        assert observed == expected

    frozen = {
        "source_manifest_sha256": ARTIFACT_ROOT / "source-manifest.json",
        "residual_manifest_sha256": ARTIFACT_ROOT / "residual-manifest.json",
        "fingerprint_bank_sha256": ARTIFACT_ROOT / "fingerprint-bank.json",
        "scores_sha256": ARTIFACT_ROOT / "scores.csv",
        "condition_evaluation_sha256": (ARTIFACT_ROOT / "evaluation" / "condition-evaluation.csv"),
        "summary_sha256": ARTIFACT_ROOT / "evaluation" / "summary.json",
        "report_sha256": REPOSITORY_ROOT / "reports" / "e02" / "E02_REPORT.md",
    }
    for key, path in frozen.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == validation[key]
    assert (
        hashlib.sha256((ARTIFACT_ROOT / "pytest-junit.xml").read_bytes()).hexdigest()
        == validation["quality_gate"]["pytest"]["junit_sha256"]
    )
