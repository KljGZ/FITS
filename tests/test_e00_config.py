from __future__ import annotations

from pathlib import Path

from gfits.synthetic import SYNTHETIC_MODELS, load_e00_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_e00_gate_is_pre_registered_and_complete() -> None:
    configuration = load_e00_config(REPOSITORY_ROOT / "configs" / "e00.yaml")

    assert configuration["status"] == "pre_registered"
    assert tuple(configuration["synthetic"]["models"]) == SYNTHETIC_MODELS
    assert configuration["synthetic"]["development_seed"] != configuration["synthetic"]["gate_seed"]
    assert configuration["stop_rule"].startswith("Any failed model-level check")


def test_fits_plus_is_identified_as_project_extension() -> None:
    configuration = load_e00_config(REPOSITORY_ROOT / "configs" / "e00.yaml")

    note = configuration["definitions"]["terminology_note"]
    assert "project extension" in note
    assert "not a name attributed" in note
