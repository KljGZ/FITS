from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gfits.manifest import MANIFEST_SCHEMA, ManifestError, build_manifest, verify_manifest


def test_build_and_verify_manifest_without_decoding(tmp_path: Path) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    payload = b"not decoded; these bytes are intentionally not a valid PNG"
    image = data_root / "sample.PNG"
    image.write_bytes(payload)
    (data_root / "notes.txt").write_text("excluded by default", encoding="utf-8")
    output = tmp_path / "manifest.json"

    manifest = build_manifest(data_root, output)

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["record_count"] == 1
    assert manifest["records"] == [
        {
            "sample_id": "sample.PNG",
            "relative_path": "sample.PNG",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    result = verify_manifest(output, data_root, strict=True)
    assert result.ok
    assert result.checked == 1


def test_verification_detects_changed_bytes(tmp_path: Path) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    image = data_root / "sample.jpg"
    image.write_bytes(b"before")
    output = tmp_path / "manifest.json"
    build_manifest(data_root, output)

    image.write_bytes(b"after!")

    result = verify_manifest(output, data_root)
    assert not result.ok
    assert result.issues[0].reason == "sha256_mismatch"


def test_strict_verification_detects_untracked_image(tmp_path: Path) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    (data_root / "registered.png").write_bytes(b"registered")
    output = tmp_path / "manifest.json"
    build_manifest(data_root, output)
    (data_root / "untracked.webp").write_bytes(b"untracked")

    result = verify_manifest(output, data_root, strict=True)

    assert not result.ok
    assert result.unexpected_paths == ("untracked.webp",)


def test_verification_rejects_path_traversal(tmp_path: Path) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    output = tmp_path / "manifest.json"
    output.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "hash_algorithm": "sha256",
                "source_root": str(data_root),
                "include_patterns": ["*.png"],
                "record_count": 1,
                "records": [
                    {
                        "sample_id": "escape",
                        "relative_path": "../escape.png",
                        "size_bytes": 0,
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="unsafe relative_path"):
        verify_manifest(output, data_root)


def test_verification_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    output = tmp_path / "manifest.json"
    records = [
        {
            "sample_id": "duplicate",
            "relative_path": name,
            "size_bytes": 0,
            "sha256": "0" * 64,
        }
        for name in ("first.png", "second.png")
    ]
    output.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "hash_algorithm": "sha256",
                "source_root": str(data_root),
                "include_patterns": ["*.png"],
                "record_count": len(records),
                "records": records,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="duplicate sample_id"):
        verify_manifest(output, data_root)


def test_all_files_excludes_manifest_output_under_root(tmp_path: Path) -> None:
    data_root = tmp_path / "evidence"
    data_root.mkdir()
    (data_root / "payload.bin").write_bytes(b"payload")
    output = data_root / "manifest.json"

    first = build_manifest(data_root, output, ("**/*",))
    second = build_manifest(data_root, output, ("**/*",))

    assert first["record_count"] == second["record_count"] == 1
    assert second["records"][0]["relative_path"] == "payload.bin"
