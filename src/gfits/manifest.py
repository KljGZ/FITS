"""Byte-preserving dataset manifests for forensic evidence.

This module hashes files as raw byte streams. It deliberately does not import an
image decoder so that manifest construction cannot silently transform evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_SCHEMA = "gfits.dataset-manifest/v1"
DEFAULT_IMAGE_GLOBS = (
    "*.bmp",
    "*.dng",
    "*.gif",
    "*.heic",
    "*.heif",
    "*.jpeg",
    "*.jpg",
    "*.png",
    "*.tif",
    "*.tiff",
    "*.webp",
)


class ManifestError(ValueError):
    """Raised when a manifest or manifest request is invalid."""


@dataclass(frozen=True)
class VerificationIssue:
    """One failed integrity check."""

    path: str
    reason: str
    expected: str | int | None = None
    actual: str | int | None = None


@dataclass(frozen=True)
class VerificationResult:
    """Aggregate result returned by :func:`verify_manifest`."""

    ok: bool
    checked: int
    issues: tuple[VerificationIssue, ...]
    unexpected_paths: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "ok": self.ok,
            "checked": self.checked,
            "issues": [asdict(issue) for issue in self.issues],
            "unexpected_paths": list(self.unexpected_paths),
        }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without decoding its contents."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(relative_path: str, patterns: Sequence[str]) -> bool:
    candidate = PurePosixPath(relative_path)
    lowered = PurePosixPath(relative_path.lower())
    return any(
        pattern == "**/*" or candidate.match(pattern) or lowered.match(pattern.lower())
        for pattern in patterns
    )


def _relative_files(
    root: Path,
    patterns: Sequence[str],
    excluded: Iterable[Path] = (),
) -> list[tuple[str, Path]]:
    excluded_resolved = {path.resolve() for path in excluded}
    selected: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not _matches(relative, patterns):
            continue
        if path.is_symlink():
            raise ManifestError(f"symbolic links are not valid evidence files: {relative}")
        if path.resolve() in excluded_resolved:
            continue
        selected.append((relative, path))
    return sorted(selected, key=lambda item: item[0])


def _patterns(include_patterns: Sequence[str] | None) -> tuple[str, ...]:
    patterns = tuple(include_patterns or DEFAULT_IMAGE_GLOBS)
    if not patterns or any(not pattern.strip() for pattern in patterns):
        raise ManifestError("at least one non-empty include pattern is required")
    return patterns


def build_manifest(
    root: Path,
    output: Path,
    include_patterns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build and atomically write a dataset manifest.

    Paths stored in the manifest are POSIX-style and relative to *root*. The
    output itself is excluded when it is located below the evidence root.
    """

    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"dataset root is not a directory: {root}")

    patterns = _patterns(include_patterns)
    files = _relative_files(root, patterns, excluded=(output,))
    records = [
        {
            "sample_id": relative,
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for relative, path in files
    ]
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "hash_algorithm": "sha256",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_root": str(root),
        "include_patterns": list(patterns),
        "record_count": len(records),
        "records": records,
    }
    _write_json_atomic(output, manifest)
    return manifest


def _write_json_atomic(output: Path, payload: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate a manifest document."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be a JSON object")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"unsupported manifest schema: {payload.get('schema')!r}")
    if payload.get("hash_algorithm") != "sha256":
        raise ManifestError("only SHA-256 manifests are supported")
    if not isinstance(payload.get("records"), list):
        raise ManifestError("manifest records must be a JSON array")
    if payload.get("record_count") != len(payload["records"]):
        raise ManifestError("record_count does not match records length")
    return payload


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError("every record requires a non-empty relative_path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ManifestError(f"unsafe relative_path: {value!r}")
    if "\\" in value:
        raise ManifestError(f"relative_path must use POSIX separators: {value!r}")
    return value


def verify_manifest(
    manifest_path: Path,
    root: Path | None = None,
    *,
    strict: bool = False,
) -> VerificationResult:
    """Verify file presence, size, and SHA-256 against a manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    payload = load_manifest(manifest_path)
    recorded_root = payload.get("source_root")
    if root is None and (not isinstance(recorded_root, str) or not recorded_root):
        raise ManifestError("manifest source_root must be a non-empty string")
    source_root = root if root is not None else Path(recorded_root)
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise ManifestError(f"dataset root is not a directory: {source_root}")

    issues: list[VerificationIssue] = []
    expected_paths: set[str] = set()
    expected_ids: set[str] = set()
    for index, raw_record in enumerate(payload["records"]):
        if not isinstance(raw_record, dict):
            raise ManifestError(f"record {index} must be a JSON object")
        sample_id = raw_record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ManifestError(f"record {index} requires a non-empty sample_id")
        if sample_id in expected_ids:
            raise ManifestError(f"duplicate sample_id: {sample_id}")
        expected_ids.add(sample_id)
        relative = _safe_relative_path(raw_record.get("relative_path"))
        if relative in expected_paths:
            raise ManifestError(f"duplicate relative_path: {relative}")
        expected_paths.add(relative)
        candidate = source_root / Path(*PurePosixPath(relative).parts)
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(source_root)
        except ValueError as error:
            raise ManifestError(f"path escapes dataset root: {relative}") from error

        if not candidate.is_file() or candidate.is_symlink():
            issues.append(VerificationIssue(relative, "missing_or_not_regular_file"))
            continue
        expected_size = raw_record.get("size_bytes")
        actual_size = candidate.stat().st_size
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ManifestError(f"invalid size_bytes for {relative}")
        if actual_size != expected_size:
            issues.append(VerificationIssue(relative, "size_mismatch", expected_size, actual_size))
            continue
        expected_hash = raw_record.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ManifestError(f"invalid sha256 for {relative}")
        actual_hash = sha256_file(candidate)
        if actual_hash != expected_hash:
            issues.append(
                VerificationIssue(relative, "sha256_mismatch", expected_hash, actual_hash)
            )

    unexpected: tuple[str, ...] = ()
    if strict:
        raw_patterns = payload.get("include_patterns")
        if not isinstance(raw_patterns, list) or not all(
            isinstance(item, str) for item in raw_patterns
        ):
            raise ManifestError("include_patterns must be a JSON string array")
        observed = {
            relative
            for relative, _ in _relative_files(
                source_root,
                _patterns(raw_patterns),
                excluded=(manifest_path,),
            )
        }
        unexpected = tuple(sorted(observed - expected_paths))

    return VerificationResult(
        ok=not issues and not unexpected,
        checked=len(expected_paths),
        issues=tuple(issues),
        unexpected_paths=unexpected,
    )
