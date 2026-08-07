from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from gfits.e01 import _vision_records, load_e01_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "e01.yaml"
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "e01"


def test_e01_selection_is_pre_registered_and_complete() -> None:
    configuration = load_e01_config(CONFIG_PATH)
    records = _vision_records(configuration)

    assert len(records) == 400
    assert len({record.relative_path for record in records}) == 400
    assert {record.split for record in records} == {"template", "calibration", "test"}
    assert not any(
        configuration["extractor"][operation]
        for operation in ("resize", "crop", "recompress", "color_conversion")
    )


def test_e01_url_and_registered_geometry_are_explicit() -> None:
    configuration = load_e01_config(CONFIG_PATH)
    records = _vision_records(configuration)
    first = records[0]

    assert first.source_url.endswith("/D02_Apple_iPhone4s/images/nat/D02_I_nat_0001.jpg")
    assert (first.expected_width, first.expected_height) == (3264, 2448)
    assert first.relative_path == ("D02_Apple_iPhone4s/native/D02_I_nat_0001.jpg")


def test_e01_rejects_source_index_leakage(tmp_path: Path) -> None:
    configuration = deepcopy(load_e01_config(CONFIG_PATH))
    configuration["dataset"]["splits"]["calibration"] = [5, 9]
    path = tmp_path / "leaking.yaml"
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")

    with pytest.raises(ValueError, match="leakage"):
        load_e01_config(path)


def test_e01_frozen_artifact_hashes_match_summary() -> None:
    summary = json.loads((ARTIFACT_ROOT / "summary.json").read_text(encoding="utf-8"))

    assert summary["completion_gate"]["passed"] is True
    assert summary["aggregate"]["mechanism_hypothesis"]["passed"] is True
    for filename, expected in summary["artifacts"].items():
        observed = hashlib.sha256((ARTIFACT_ROOT / filename).read_bytes()).hexdigest()
        assert observed == expected
    for relative_path, expected in summary["source_sha256"].items():
        # Git may materialize source text with CRLF on Windows. The experiment
        # ran from canonical LF text, so normalize only source line endings.
        source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        observed = hashlib.sha256(source.encode("utf-8")).hexdigest()
        assert observed == expected


def test_e01_frozen_tables_have_registered_counts() -> None:
    with (ARTIFACT_ROOT / "scores.csv").open(newline="", encoding="utf-8") as stream:
        scores = list(csv.DictReader(stream))
    with (ARTIFACT_ROOT / "fingerprints.csv").open(newline="", encoding="utf-8") as stream:
        fingerprints = list(csv.DictReader(stream))
    with (ARTIFACT_ROOT / "threshold-evaluation.csv").open(newline="", encoding="utf-8") as stream:
        evaluations = list(csv.DictReader(stream))

    assert len(scores) == 1500
    assert len(fingerprints) == 20
    assert len(evaluations) == 16
    assert all(row["calibration_h0_count"] == "100" for row in evaluations)
    assert all(row["test_h0_count"] == "200" for row in evaluations)
    assert all(row["test_h1_count"] == "50" for row in evaluations)
