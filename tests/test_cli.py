from __future__ import annotations

import json
from pathlib import Path

from gfits.cli import _parser, main


def test_e02b_command_contract_is_exposed() -> None:
    parser = _parser()
    subparsers = next(
        action
        for action in parser._actions
        if action.dest == "command"  # noqa: SLF001
    )
    assert {
        "generate-e02b-shard",
        "merge-e02b-generation",
        "apply-e02b-counterfactuals",
        "extract-e02b-representations",
        "merge-e02b-representations",
        "score-e02b",
        "select-e02b-condition",
        "evaluate-e02b",
        "generate-e02b-report",
    }.issubset(subparsers.choices)


def test_cli_build_and_verify(tmp_path: Path, capsys: object) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    (data_root / "query.png").write_bytes(b"query")
    manifest = tmp_path / "manifest.json"

    assert main(["build-manifest", "--root", str(data_root), "--output", str(manifest)]) == 0
    build_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert build_output["record_count"] == 1

    assert (
        main(
            [
                "verify-manifest",
                "--manifest",
                str(manifest),
                "--root",
                str(data_root),
                "--strict",
            ]
        )
        == 0
    )
    verify_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert verify_output == {
        "ok": True,
        "checked": 1,
        "issues": [],
        "unexpected_paths": [],
    }


def test_cli_returns_nonzero_for_tamper(tmp_path: Path, capsys: object) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    image = data_root / "query.png"
    image.write_bytes(b"query")
    manifest = tmp_path / "manifest.json"
    assert main(["build-manifest", "--root", str(data_root), "--output", str(manifest)]) == 0
    capsys.readouterr()  # type: ignore[attr-defined]
    image.write_bytes(b"tampered")

    assert main(["verify-manifest", "--manifest", str(manifest)]) == 1
