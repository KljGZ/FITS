"""Immutable member-level data preparation for the E02 generator-signal gate."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import requests
import yaml
from gdown.download import get_url_from_gdrive_confirmation
from PIL import Image
from remotezip import RemoteZip

from gfits.manifest import sha256_file

E02_MANIFEST_SCHEMA = "gfits.e02-source-manifest/v1"


def _required(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise ValueError(f"missing configuration key: {name}")
    return mapping[name]


def load_e02_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen E02 protocol without inspecting scores."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("phase") != "E02":
        raise ValueError("configuration must be a Phase E02 mapping")
    if payload.get("status") != "pre_registered":
        raise ValueError("E02 configuration must be pre_registered")
    selection = _required(payload, "selection")
    splits = selection.get("splits")
    if not isinstance(splits, dict) or list(splits) != ["template", "calibration", "test"]:
        raise ValueError("selection.splits must define template, calibration, and test in order")
    if any(int(value) <= 0 for value in splits.values()):
        raise ValueError("every E02 split count must be positive")
    transformations = selection.get("transformations", {})
    forbidden = ("resize", "crop", "recompress", "exif_transpose", "color_space_conversion")
    if any(bool(transformations.get(operation)) for operation in forbidden):
        raise ValueError("E02 forbids image geometry, encoding, orientation, and color transforms")
    datasets = _required(payload, "datasets")
    if set(datasets) != {"dmimage_detection", "genimage"}:
        raise ValueError("E02 requires the registered DMimageDetection and GenImage collections")
    source_count = 0
    for dataset_id, dataset in datasets.items():
        archive = _required(dataset, "archive")
        if archive.get("kind") not in {"google_drive_zip", "http_zip_mirror"}:
            raise ValueError(f"unsupported archive kind for {dataset_id}")
        for suite_id, suite in _required(dataset, "suites").items():
            width, height = suite.get("expected_resolution", [0, 0])
            if int(width) <= 0 or int(height) <= 0:
                raise ValueError(f"invalid native geometry for {suite_id}")
            sources = _required(suite, "sources")
            if len(sources) < 3:
                raise ValueError(f"suite {suite_id} needs at least three sources")
            source_count += len(sources)
            for source_id, source in sources.items():
                try:
                    re.compile(str(source["prompt_regex"]))
                except (KeyError, re.error) as error:
                    raise ValueError(f"invalid prompt regex for {source_id}: {error}") from error
    expected = source_count * sum(int(value) for value in splits.values())
    registered = int(payload["completion_gate"]["expected_selected_file_count"])
    if expected != registered:
        raise ValueError(f"registered E02 file count is {registered}, selection implies {expected}")
    extractors = set(_required(payload, "extractors"))
    if extractors != {"wavelet", "srm", "low_bit", "noiseprint"}:
        raise ValueError("E02 extractor matrix is incomplete")
    aggregators = payload.get("aggregators", {}).get("names")
    if aggregators != ["mean", "prnu_mle", "median", "trimmed_mean"]:
        raise ValueError("E02 aggregator matrix is incomplete or reordered")
    payload["_config_path"] = str(path.resolve())
    return payload


def _rank(seed: int, *parts: str) -> str:
    material = "|".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _safe_member_path(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe ZIP member path: {name}")
    return Path(*pure.parts)


@contextmanager
def _open_archive(archive: Mapping[str, Any]) -> Iterator[tuple[RemoteZip, dict[str, Any]]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "G-FITS-E02-research/1.0"})
    if archive["kind"] == "google_drive_zip":
        origin = f"https://drive.google.com/uc?id={archive['file_id']}"
        landing = session.get(origin, timeout=120)
        landing.raise_for_status()
        url = get_url_from_gdrive_confirmation(landing.text)
    else:
        url = str(archive["url"])
    response = session.head(url, allow_redirects=True, timeout=120)
    response.raise_for_status()
    size = int(response.headers.get("Content-Length", -1))
    if size != int(archive["size_bytes"]):
        raise ValueError(
            f"archive size mismatch: observed {size}, expected {archive['size_bytes']}"
        )
    observed_etag = response.headers.get("ETag", "").strip('"')
    expected_etag = str(archive.get("etag", ""))
    if expected_etag and observed_etag != expected_etag:
        raise ValueError(f"archive ETag mismatch: {observed_etag}")
    metadata = {
        "kind": archive["kind"],
        "filename": archive["filename"],
        "size_bytes": size,
        "etag": observed_etag or None,
        "last_modified": response.headers.get("Last-Modified"),
        "archive_sha256": archive["archive_sha256"],
        "access": "HTTP byte ranges; selected members are independently SHA-256 hashed",
    }
    with RemoteZip(url, session=session, initial_buffer_size=4 << 20) as remote:
        yield remote, metadata


def _member_groups(
    members: Sequence[Any],
    source: Mapping[str, Any],
) -> dict[str, list[str]]:
    prefix = str(source["member_prefix"])
    expression = re.compile(str(source["prompt_regex"]))
    groups: dict[str, list[str]] = defaultdict(list)
    for member in members:
        if member.is_dir() or not member.filename.startswith(prefix):
            continue
        filename = member.filename.rsplit("/", 1)[-1]
        match = expression.fullmatch(filename)
        if match:
            groups[match.group("prompt")].append(member.filename)
    if not groups:
        raise ValueError(f"no archive members matched {prefix} and {expression.pattern}")
    return dict(groups)


def _split_prompt_ids(
    prompt_ids: Sequence[str],
    split_counts: Mapping[str, Any],
    *,
    seed: int,
    rank_parts: Sequence[str],
) -> dict[str, list[str]]:
    ordered = sorted(prompt_ids, key=lambda value: _rank(seed, *rank_parts, value))
    required = sum(int(value) for value in split_counts.values())
    if len(ordered) < required:
        raise ValueError(f"only {len(ordered)} prompt groups available; {required} required")
    result: dict[str, list[str]] = {}
    offset = 0
    for split, raw_count in split_counts.items():
        count = int(raw_count)
        result[split] = ordered[offset : offset + count]
        offset += count
    return result


def _choose_member(
    candidates: Sequence[str],
    *,
    seed: int,
    dataset_id: str,
    suite_id: str,
    source_id: str,
    prompt_id: str,
) -> str:
    return min(
        candidates,
        key=lambda name: _rank(seed, dataset_id, suite_id, source_id, prompt_id, name),
    )


def _select_suite(
    dataset_id: str,
    suite_id: str,
    suite: Mapping[str, Any],
    members: Sequence[Any],
    split_counts: Mapping[str, Any],
    seed: int,
) -> list[dict[str, str]]:
    groups = {
        source_id: _member_groups(members, source) for source_id, source in suite["sources"].items()
    }
    selections: list[dict[str, str]] = []
    if suite["prompt_policy"] == "shared_across_sources":
        common = set.intersection(*(set(value) for value in groups.values()))
        shared = _split_prompt_ids(
            list(common),
            split_counts,
            seed=seed,
            rank_parts=(dataset_id, suite_id, "shared_prompt"),
        )
        split_map = {source_id: shared for source_id in groups}
    elif suite["prompt_policy"] == "per_source_disjoint":
        split_map = {
            source_id: _split_prompt_ids(
                list(source_groups),
                split_counts,
                seed=seed,
                rank_parts=(dataset_id, suite_id, source_id, "prompt"),
            )
            for source_id, source_groups in groups.items()
        }
    else:
        raise ValueError(f"unsupported prompt policy in {suite_id}")
    for source_id, prompt_splits in split_map.items():
        for split, prompt_ids in prompt_splits.items():
            for prompt_id in prompt_ids:
                member = _choose_member(
                    groups[source_id][prompt_id],
                    seed=seed,
                    dataset_id=dataset_id,
                    suite_id=suite_id,
                    source_id=source_id,
                    prompt_id=prompt_id,
                )
                selections.append(
                    {
                        "dataset_id": dataset_id,
                        "suite_id": suite_id,
                        "source_id": source_id,
                        "split": split,
                        "prompt_id": prompt_id,
                        "member": member,
                    }
                )
    return selections


def _inspect_image(payload: bytes, suite: Mapping[str, Any], member: str) -> dict[str, Any]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        image_format = image.format
        opaque_alpha = None
        if mode == "RGBA":
            opaque_alpha = image.getchannel("A").getextrema() == (255, 255)
    expected_width, expected_height = (int(value) for value in suite["expected_resolution"])
    if (width, height) != (expected_width, expected_height):
        raise ValueError(f"unexpected geometry for {member}: {width}x{height}")
    expected_mode = suite["expected_mode"]
    if expected_mode == "RGB" and mode != "RGB":
        raise ValueError(f"unexpected mode for {member}: {mode}")
    if expected_mode == "RGB_or_fully_opaque_RGBA":
        if mode not in {"RGB", "RGBA"} or (mode == "RGBA" and not opaque_alpha):
            raise ValueError(f"unsupported or non-opaque mode for {member}: {mode}")
    if image_format != "PNG":
        raise ValueError(f"unexpected image encoding for {member}: {image_format}")
    return {
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
        "opaque_alpha_verified": opaque_alpha,
        "analysis_channels": "RGB" if mode == "RGB" else "RGB with verified opaque alpha ignored",
    }


def _write_member(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
    temporary.write_bytes(payload)
    temporary.replace(target)


def _leakage_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_disjoint = True
    original_disjoint = True
    for source_id in {str(record["source_id"]) for record in records}:
        source_records = [record for record in records if record["source_id"] == source_id]
        prompts = {
            split: {
                str(record["prompt_id"]) for record in source_records if record["split"] == split
            }
            for split in ("template", "calibration", "test")
        }
        originals = {
            split: {
                str(record["archive_member"])
                for record in source_records
                if record["split"] == split
            }
            for split in ("template", "calibration", "test")
        }
        for left in prompts:
            for right in prompts:
                if left < right:
                    prompt_disjoint &= not bool(prompts[left] & prompts[right])
                    original_disjoint &= not bool(originals[left] & originals[right])
    return {
        "prompt_disjoint_splits": prompt_disjoint,
        "original_disjoint_splits": original_disjoint,
        "seed_metadata_complete": False,
        "seed_disjoint_splits": None,
        "seed_limitation": "Neither public release publishes a per-image random-seed mapping.",
    }


def prepare_e02_data(
    config_path: Path,
    data_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Range-extract only pre-registered members and hash their original bytes."""

    config = load_e02_config(config_path.resolve())
    data_root = data_root.resolve()
    manifest_path = manifest_path.resolve()
    seed = int(config["selection"]["deterministic_seed"])
    split_counts = config["selection"]["splits"]
    records: list[dict[str, Any]] = []
    archive_audits: dict[str, Any] = {}
    for dataset_id, dataset in config["datasets"].items():
        with _open_archive(dataset["archive"]) as (archive, archive_metadata):
            members = archive.infolist()
            if len(members) != int(dataset["archive"]["member_count"]):
                raise ValueError(f"archive member-count mismatch for {dataset_id}")
            archive_audits[dataset_id] = archive_metadata
            selections: list[dict[str, str]] = []
            for suite_id, suite in dataset["suites"].items():
                selections.extend(
                    _select_suite(
                        dataset_id,
                        suite_id,
                        suite,
                        members,
                        split_counts,
                        seed,
                    )
                )
            for index, selected in enumerate(selections, start=1):
                payload = archive.read(selected["member"])
                suite = dataset["suites"][selected["suite_id"]]
                inspection = _inspect_image(payload, suite, selected["member"])
                relative_path = Path(dataset_id) / _safe_member_path(selected["member"])
                target = data_root / relative_path
                if not target.is_file() or target.read_bytes() != payload:
                    _write_member(target, payload)
                digest = hashlib.sha256(payload).hexdigest()
                if sha256_file(target) != digest:
                    raise RuntimeError(f"written member hash mismatch: {selected['member']}")
                records.append(
                    {
                        "sample_id": (
                            f"{dataset_id}:{selected['suite_id']}:"
                            f"{selected['source_id']}:{selected['member']}"
                        ),
                        "dataset_id": dataset_id,
                        "suite_id": selected["suite_id"],
                        "source_id": selected["source_id"],
                        "split": selected["split"],
                        "prompt_id": selected["prompt_id"],
                        "seed_id": None,
                        "seed_metadata": dataset["seed_metadata"],
                        "original_id": selected["member"],
                        "archive_member": selected["member"],
                        "relative_path": relative_path.as_posix(),
                        "size_bytes": len(payload),
                        "sha256": digest,
                        **inspection,
                        "transformations_by_gfits": [],
                    }
                )
                if index % 25 == 0 or index == len(selections):
                    print(
                        f"range_extracted={dataset_id}:{index}/{len(selections)}",
                        flush=True,
                    )
    records.sort(key=lambda value: value["sample_id"])
    leakage = _leakage_audit(records)
    payload = {
        "schema": E02_MANIFEST_SCHEMA,
        "phase": "E02",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "configuration_sha256": sha256_file(config_path.resolve()),
        "source_root": str(data_root),
        "record_count": len(records),
        "archive_audits": archive_audits,
        "leakage_audit": leakage,
        "records": records,
    }
    expected = int(config["completion_gate"]["expected_selected_file_count"])
    if len(records) != expected:
        raise RuntimeError(f"selected {len(records)} files, expected {expected}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "ok": leakage["prompt_disjoint_splits"] and leakage["original_disjoint_splits"],
        "manifest": str(manifest_path),
        "record_count": len(records),
        "manifest_sha256": sha256_file(manifest_path),
        "seed_metadata_complete": leakage["seed_metadata_complete"],
    }


def load_e02_manifest(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Load an E02 manifest and bind it to the exact pre-registration file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != E02_MANIFEST_SCHEMA:
        raise ValueError("unsupported E02 manifest schema")
    if payload.get("record_count") != len(payload.get("records", [])):
        raise ValueError("E02 manifest record count is inconsistent")
    if payload.get("configuration_sha256") != sha256_file(Path(config["_config_path"])):
        raise ValueError("E02 manifest was built from a different configuration")
    return payload


def verify_e02_manifest(payload: Mapping[str, Any], data_root: Path) -> dict[str, Any]:
    """Verify every selected source byte without decoding or transforming it."""

    issues: list[str] = []
    for record in payload["records"]:
        path = data_root / Path(record["relative_path"])
        if not path.is_file():
            issues.append(f"missing:{record['relative_path']}")
        elif path.stat().st_size != int(record["size_bytes"]):
            issues.append(f"size:{record['relative_path']}")
        elif sha256_file(path) != record["sha256"]:
            issues.append(f"sha256:{record['relative_path']}")
    return {"passed": not issues, "checked": len(payload["records"]), "issues": issues}
