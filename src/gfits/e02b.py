"""Controlled E02b execution: counterfactuals, representations, matching, and gates."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from gfits.e02 import _load_prnu, _NoiseprintSession, _wavelet_residual
from gfits.e02b_data import (
    E02B_DERIVATIVE_SCHEMA,
    E02B_GENERATION_SCHEMA,
    decode_manifest_rgb,
    load_e02b_config,
    repository_state,
    repository_state_dict,
    validate_generation_manifest,
    write_counterfactual,
    write_json_atomic,
)
from gfits.e02b_features import (
    ChannelWhitening,
    aggregate_template,
    channel_mean_ncc,
    fit_channel_whitening,
    fixed_three_kernel_high_pass_residual_bank,
    global_ncc,
    intensity_multiplicativity_diagnostics,
    low_bit_residual,
    luminance,
    multiplicative_modulation_supported,
    statistical_signature,
    whitened_global_ncc,
)
from gfits.e02b_statistics import (
    attribution_metrics,
    auxiliary_sign_flip_test,
    e02b_gate,
    hierarchical_family_prompt_permutation,
    holm_adjust,
    prompt_block_source_label_permutation,
    template_source_label_permutation,
    two_way_bootstrap,
)
from gfits.explicit_scoring import gallery_complement_ratio
from gfits.manifest import sha256_file
from gfits.matching import (
    aligned_peak_position,
    cross_correlation_map,
    signed_pce,
)

E02B_REPRESENTATION_FRAGMENT_SCHEMA = "gfits.e02b-representation-fragment/v1"
E02B_REPRESENTATION_SCHEMA = "gfits.e02b-representation-manifest/v1"
E02B_SCORE_SCHEMA = "gfits.e02b-controlled-scores/v1"
E02B_RESULT_SCHEMA = "gfits.e02b-controlled-confirmation/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_generation(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != E02B_GENERATION_SCHEMA:
        raise ValueError("unsupported E02b generation manifest")
    if payload.get("configuration_sha256") != sha256_file(Path(config["_config_path"])):
        raise ValueError("generation manifest uses a different E02b configuration")
    return payload


def apply_e02b_counterfactuals(
    config_path: Path,
    repository_root: Path,
    generation_manifest_path: Path,
    data_root: Path,
    derivative_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Apply all exporter/low-bit counterfactuals to formal non-auxiliary samples."""

    config = load_e02b_config(config_path.resolve())
    state = repository_state(repository_root.resolve(), config)
    validation = validate_generation_manifest(
        generation_manifest_path.resolve(), data_root.resolve(), config, verify_files=True
    )
    if not validation["passed"]:
        raise ValueError("generation manifest did not pass before counterfactual creation")
    generation = _load_generation(generation_manifest_path.resolve(), config)
    records = []
    writer = config["design"]["writer"]
    for parent in generation["records"]:
        if parent["split"] == "auxiliary":
            continue
        source = data_root.resolve() / str(parent["relative_path"])
        for counterfactual_id, settings in config["counterfactuals"].items():
            relative = Path(counterfactual_id) / str(parent["relative_path"])
            target = derivative_root.resolve() / relative
            record = write_counterfactual(
                source,
                target,
                parent,
                str(counterfactual_id),
                settings,
                writer,
            )
            record["relative_path"] = relative.as_posix()
            record.update(
                {
                    key: parent[key]
                    for key in (
                        "suite_ids",
                        "model_id",
                        "prompt_id",
                        "prompt",
                        "negative_prompt",
                        "seed",
                        "split",
                        "native_resolution",
                    )
                }
            )
            records.append(record)
    payload = {
        "schema": E02B_DERIVATIVE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "configuration_sha256": sha256_file(config_path.resolve()),
        "generation_manifest_sha256": sha256_file(generation_manifest_path.resolve()),
        "repository_state": repository_state_dict(state),
        "derivative_root": str(derivative_root.resolve()),
        "records": sorted(records, key=lambda row: row["sample_id"]),
    }
    write_json_atomic(output_path.resolve(), payload)
    pixel_checks: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in records:
        key = "identical" if row["pixel_identical"] else "changed"
        pixel_checks[str(row["counterfactual_id"])][key] += 1
    return {
        "ok": True,
        "manifest": str(output_path.resolve()),
        "manifest_sha256": sha256_file(output_path.resolve()),
        "record_count": len(records),
        "pixel_checks": {key: dict(value) for key, value in pixel_checks.items()},
        "repository_commit": state.commit,
    }


def _save_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part.npy")
    np.save(temporary, np.asarray(value, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)
    return {
        "relative_path": path.as_posix(),
        "sha256": sha256_file(path),
        "shape": list(value.shape),
        "dtype": "float32",
        "size_bytes": path.stat().st_size,
    }


def _load_derivatives(path: Path | None, config: Mapping[str, Any]) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != E02B_DERIVATIVE_SCHEMA:
        raise ValueError("unsupported E02b derivative manifest")
    if payload.get("configuration_sha256") != sha256_file(Path(config["_config_path"])):
        raise ValueError("derivative manifest uses a different E02b configuration")
    return payload


def _decode_derivative(path: Path, record: Mapping[str, Any]) -> np.ndarray:
    if sha256_file(path) != record["output_sha256"]:
        raise ValueError(f"derivative SHA-256 mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.format != "PNG":
            raise ValueError(f"non-canonical derivative: {path}")
        if image.size != (int(record["native_resolution"]), int(record["native_resolution"])):
            raise ValueError(f"derivative geometry mismatch: {path}")
        return np.ascontiguousarray(np.asarray(image))


def extract_e02b_representations(
    config_path: Path,
    repository_root: Path,
    generation_manifest_path: Path,
    data_root: Path,
    cache_root: Path,
    fragment_path: Path,
    upstream_root: Path,
    *,
    extractor: str,
    condition: str = "native",
    derivative_manifest_path: Path | None = None,
    derivative_root: Path | None = None,
    noiseprint_root: Path | None = None,
) -> dict[str, Any]:
    """Extract one registered residual and its registered statistical signatures."""

    config = load_e02b_config(config_path.resolve())
    state = repository_state(repository_root.resolve(), config)
    if extractor not in config["spatial_branch"]["extractors"]:
        raise ValueError(f"unregistered E02b extractor: {extractor}")
    generation = _load_generation(generation_manifest_path.resolve(), config)
    derivatives = _load_derivatives(
        derivative_manifest_path.resolve() if derivative_manifest_path else None, config
    )
    if condition == "native":
        source_records = [row for row in generation["records"] if row["split"] != "auxiliary"]
        root = data_root.resolve()
    else:
        if derivatives is None or derivative_root is None:
            raise ValueError("counterfactual extraction requires its manifest and root")
        source_records = [
            row for row in derivatives["records"] if row["counterfactual_id"] == condition
        ]
        root = derivative_root.resolve()
    prnu = None
    noiseprint = None
    if extractor == "wavelet":
        prnu = _load_prnu(upstream_root.resolve(), config["residuals"]["wavelet"])
    if extractor == "noiseprint":
        if noiseprint_root is None:
            raise ValueError("Noiseprint extraction requires --noiseprint-root")
        noiseprint = _NoiseprintSession(
            noiseprint_root.resolve(), config["residuals"]["noiseprint"]
        )
    existing: dict[str, dict[str, Any]] = {}
    if fragment_path.is_file():
        payload = json.loads(fragment_path.read_text(encoding="utf-8"))
        if payload.get("schema") != E02B_REPRESENTATION_FRAGMENT_SCHEMA:
            raise ValueError("unsupported existing E02b representation fragment")
        if payload.get("repository_state", {}).get("commit") != state.commit:
            raise ValueError("representation fragment belongs to another code commit")
        existing = {str(row["record_id"]): row for row in payload["records"]}
    records = dict(existing)
    try:
        for index, source_record in enumerate(source_records):
            logical_sample_id = str(
                source_record.get("parent_sample_id", source_record["sample_id"])
            )
            record_id = f"{condition}::{logical_sample_id}::{extractor}"
            old = records.get(record_id)
            if old:
                cached = cache_root.resolve() / str(old["relative_path"])
                if cached.is_file() and sha256_file(cached) == old["sha256"]:
                    continue
            path = root / str(source_record["relative_path"])
            if condition == "native":
                image = decode_manifest_rgb(path, source_record)
            else:
                image = _decode_derivative(path, source_record)
            if extractor == "wavelet":
                residual = _wavelet_residual(image, prnu, config["residuals"]["wavelet"])
            elif extractor == "fixed_three_kernel_high_pass_residual_bank":
                residual = fixed_three_kernel_high_pass_residual_bank(image)
            elif extractor == "low_bit":
                residual = low_bit_residual(image, int(config["residuals"]["low_bit"]["bit_count"]))
            else:
                residual = noiseprint.extract(image)
            relative = Path(condition) / extractor / f"{logical_sample_id}.npy"
            evidence = _save_array(cache_root.resolve() / relative, residual)
            evidence["relative_path"] = relative.as_posix()
            records[record_id] = {
                "record_id": record_id,
                "sample_id": logical_sample_id,
                "parent_sample_id": logical_sample_id,
                "condition": condition,
                "branch": "spatial_mean",
                "representation": extractor,
                "model_id": source_record["model_id"],
                "suite_ids": source_record["suite_ids"],
                "prompt_id": source_record["prompt_id"],
                "seed": source_record["seed"],
                "split": source_record["split"],
                "native_resolution": source_record["native_resolution"],
                **evidence,
            }
            if extractor == "fixed_three_kernel_high_pass_residual_bank":
                grid_size = int(
                    config["statistical_signature_branch"]["normalized_signature_grid_size"]
                )
                for signature_name in config["statistical_signature_branch"]["signatures"]:
                    signature = statistical_signature(
                        str(signature_name), residual, grid_size=grid_size
                    )
                    signature_id = f"{condition}::{logical_sample_id}::{signature_name}"
                    signature_relative = (
                        Path(condition)
                        / "statistical_signature"
                        / str(signature_name)
                        / f"{logical_sample_id}.npy"
                    )
                    signature_evidence = _save_array(
                        cache_root.resolve() / signature_relative, signature
                    )
                    signature_evidence["relative_path"] = signature_relative.as_posix()
                    records[signature_id] = {
                        "record_id": signature_id,
                        "sample_id": logical_sample_id,
                        "parent_sample_id": logical_sample_id,
                        "condition": condition,
                        "branch": "statistical_signature",
                        "representation": str(signature_name),
                        "model_id": source_record["model_id"],
                        "suite_ids": source_record["suite_ids"],
                        "prompt_id": source_record["prompt_id"],
                        "seed": source_record["seed"],
                        "split": source_record["split"],
                        "native_resolution": source_record["native_resolution"],
                        **signature_evidence,
                    }
            if index % 25 == 0 or index + 1 == len(source_records):
                write_json_atomic(
                    fragment_path.resolve(),
                    {
                        "schema": E02B_REPRESENTATION_FRAGMENT_SCHEMA,
                        "configuration_sha256": sha256_file(config_path.resolve()),
                        "generation_manifest_sha256": sha256_file(
                            generation_manifest_path.resolve()
                        ),
                        "derivative_manifest_sha256": (
                            sha256_file(derivative_manifest_path.resolve())
                            if derivative_manifest_path
                            else None
                        ),
                        "repository_state": repository_state_dict(state),
                        "condition": condition,
                        "extractor": extractor,
                        "records": [records[key] for key in sorted(records)],
                    },
                )
    finally:
        if noiseprint is not None:
            noiseprint.close()
    expected_per_sample = (
        1 + len(config["statistical_signature_branch"]["signatures"])
        if extractor == "fixed_three_kernel_high_pass_residual_bank"
        else 1
    )
    return {
        "ok": len(records) == len(source_records) * expected_per_sample,
        "fragment": str(fragment_path.resolve()),
        "condition": condition,
        "extractor": extractor,
        "record_count": len(records),
        "expected_record_count": len(source_records) * expected_per_sample,
        "repository_commit": state.commit,
    }


def merge_e02b_representation_fragments(
    config_path: Path,
    repository_root: Path,
    fragment_root: Path,
    cache_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Merge representation shards and verify every external cached array."""

    config = load_e02b_config(config_path.resolve())
    state = repository_state(repository_root.resolve(), config)
    records: dict[str, dict[str, Any]] = {}
    fragments = []
    for path in sorted(fragment_root.resolve().glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != E02B_REPRESENTATION_FRAGMENT_SCHEMA:
            raise ValueError(f"unsupported representation fragment: {path}")
        if payload.get("repository_state", {}).get("commit") != state.commit:
            raise ValueError(f"representation fragment commit mismatch: {path}")
        for row in payload["records"]:
            record_id = str(row["record_id"])
            if record_id in records:
                raise ValueError(f"duplicate representation record: {record_id}")
            cached = cache_root.resolve() / str(row["relative_path"])
            if not cached.is_file() or sha256_file(cached) != row["sha256"]:
                raise ValueError(f"representation cache verification failed: {record_id}")
            records[record_id] = row
        fragments.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "count": len(payload["records"]),
            }
        )
    payload = {
        "schema": E02B_REPRESENTATION_SCHEMA,
        "generated_at_utc": _utc_now(),
        "configuration_sha256": sha256_file(config_path.resolve()),
        "repository_state": repository_state_dict(state),
        "cache_root": str(cache_root.resolve()),
        "fragments": fragments,
        "records": [records[key] for key in sorted(records)],
    }
    write_json_atomic(output_path.resolve(), payload)
    return {
        "ok": bool(records),
        "manifest": str(output_path.resolve()),
        "manifest_sha256": sha256_file(output_path.resolve()),
        "record_count": len(records),
        "fragment_count": len(fragments),
    }


class _RepresentationStore:
    def __init__(self, manifest_path: Path, cache_root: Path, config: Mapping[str, Any]):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != E02B_REPRESENTATION_SCHEMA:
            raise ValueError("unsupported E02b representation manifest")
        if payload.get("configuration_sha256") != sha256_file(Path(config["_config_path"])):
            raise ValueError("representation manifest uses a different E02b configuration")
        self.cache_root = cache_root.resolve()
        self.rows = payload["records"]
        self.index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        for row in self.rows:
            key = (str(row["condition"]), str(row["sample_id"]), str(row["representation"]))
            if key in self.index:
                raise ValueError(f"duplicate representation key: {key}")
            self.index[key] = row

    def load(self, condition: str, sample_id: str, representation: str) -> np.ndarray:
        row = self.index[(condition, sample_id, representation)]
        path = self.cache_root / str(row["relative_path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"representation cache hash changed: {row['record_id']}")
        return np.load(path, allow_pickle=False, mmap_mode="r")


class _ImageResolver:
    def __init__(
        self,
        generation: Mapping[str, Any],
        data_root: Path,
        derivatives: Mapping[str, Any] | None,
        derivative_root: Path | None,
    ):
        self.native = {str(row["sample_id"]): row for row in generation["records"]}
        self.data_root = data_root.resolve()
        self.derivatives = (
            {
                (str(row["counterfactual_id"]), str(row["parent_sample_id"])): row
                for row in derivatives["records"]
            }
            if derivatives
            else {}
        )
        self.derivative_root = derivative_root.resolve() if derivative_root else None

    def record(self, condition: str, sample_id: str) -> Mapping[str, Any]:
        if condition == "native":
            return self.native[sample_id]
        return self.derivatives[(condition, sample_id)]

    def rgb(self, condition: str, sample_id: str) -> np.ndarray:
        record = self.record(condition, sample_id)
        if condition == "native":
            return decode_manifest_rgb(self.data_root / str(record["relative_path"]), record)
        if self.derivative_root is None:
            raise ValueError("derivative image root is unavailable")
        return _decode_derivative(self.derivative_root / str(record["relative_path"]), record)


def _rank_samples(
    records: Sequence[Mapping[str, Any]], seed: int, *parts: str
) -> list[Mapping[str, Any]]:
    def key(row: Mapping[str, Any]) -> str:
        material = "|".join((str(seed), *parts, str(row["sample_id"]))).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    return sorted(records, key=key)


def _signed_pce_arrays(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first)
    right = np.asarray(second)
    if left.ndim == 2:
        left = left[..., np.newaxis]
        right = right[..., np.newaxis]
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("PCE requires aligned HxW or HxWxC arrays")
    values = []
    for channel in range(left.shape[-1]):
        correlation = cross_correlation_map(left[..., channel], right[..., channel])
        values.append(
            signed_pce(
                correlation,
                peak_position=aligned_peak_position(correlation.shape),
                neighborhood_radius=2,
            ).value
        )
    return float(np.mean(values))


def _score_pair(
    query: np.ndarray,
    template: np.ndarray,
    scorer: str,
    whitening: ChannelWhitening | None,
) -> tuple[float, float, float | None]:
    if scorer == "raw_signed_ncc_global":
        ncc = global_ncc(query, template)
        score = ncc
        pce = None
    elif scorer == "raw_signed_ncc_channel_mean":
        ncc = channel_mean_ncc(query, template)
        score = ncc
        pce = None
    elif scorer == "raw_signed_ncc_whitened":
        if whitening is None:
            raise ValueError("whitened scorer requires calibration-only parameters")
        ncc = whitened_global_ncc(query, template, whitening)
        score = ncc
        pce = None
    elif scorer == "raw_signed_energy":
        ncc = global_ncc(query, template)
        score = float(np.sign(ncc) * ncc * ncc)
        pce = None
    elif scorer == "raw_pce":
        ncc = global_ncc(query, template)
        pce = _signed_pce_arrays(query, template)
        score = pce
    elif scorer == "signed_ncc":
        ncc = global_ncc(query, template)
        score = ncc
        pce = None
    else:
        raise ValueError(f"unsupported E02b scorer: {scorer}")
    return score, ncc, pce


def _calibration_whitening(
    records: Sequence[Mapping[str, Any]],
    store: _RepresentationStore,
    condition: str,
    representation: str,
) -> ChannelWhitening:
    arrays = [
        store.load(condition, str(row["sample_id"]), representation)
        for row in records
        if row["split"] == "calibration"
    ]
    return fit_channel_whitening(arrays, split="calibration")


def _multiplicative_diagnostics(
    records: Sequence[Mapping[str, Any]],
    store: _RepresentationStore,
    resolver: _ImageResolver,
    condition: str,
    representation: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    calibration = [row for row in records if row["split"] == "calibration"]
    residuals = [
        store.load(condition, str(row["sample_id"]), representation) for row in calibration
    ]
    intensities = [luminance(resolver.rgb(condition, str(row["sample_id"]))) for row in calibration]
    settings = config["multiplicative_diagnostics"]
    diagnostics = intensity_multiplicativity_diagnostics(
        residuals,
        intensities,
        bins=int(settings["intensity_bins"]),
    )
    diagnostics["split"] = "calibration"
    diagnostics["representation"] = representation
    diagnostics["modulation_supported"] = multiplicative_modulation_supported(diagnostics, settings)
    return diagnostics


def _templates(
    source_records: Mapping[str, Sequence[Mapping[str, Any]]],
    store: _RepresentationStore,
    resolver: _ImageResolver,
    condition: str,
    representation: str,
    aggregator: str,
    extractor: str,
    template_size: int,
    rank_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    templates = {}
    selected_ids = {}
    for source, candidates in source_records.items():
        selected = _rank_samples(candidates, rank_seed, condition, representation, source)[
            :template_size
        ]
        if len(selected) != template_size:
            raise ValueError(f"source {source} has fewer than {template_size} template images")
        arrays = np.asarray(
            [store.load(condition, str(row["sample_id"]), representation) for row in selected]
        )
        intensities = None
        if aggregator in {"prnu_mle", "intensity_weighted"}:
            intensities = np.asarray(
                [luminance(resolver.rgb(condition, str(row["sample_id"]))) for row in selected]
            )
        templates[source] = aggregate_template(
            arrays,
            intensities,
            aggregator,
            extractor=extractor,
        )
        selected_ids[source] = [str(row["sample_id"]) for row in selected]
    return templates, selected_ids


def _templates_from_ids(
    template_ids: Mapping[str, Sequence[str]],
    store: _RepresentationStore,
    resolver: _ImageResolver,
    condition: str,
    representation: str,
    aggregator: str,
    extractor: str,
) -> dict[str, np.ndarray]:
    templates = {}
    for source, identifiers in template_ids.items():
        arrays = np.asarray(
            [store.load(condition, str(sample_id), representation) for sample_id in identifiers]
        )
        intensities = None
        if aggregator in {"prnu_mle", "intensity_weighted"}:
            intensities = np.asarray(
                [luminance(resolver.rgb(condition, str(sample_id))) for sample_id in identifiers]
            )
        templates[source] = aggregate_template(
            arrays,
            intensities,
            aggregator,
            extractor=extractor,
        )
    return templates


def _component_templates(
    templates: Mapping[str, np.ndarray], component: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    sources = sorted(templates)
    common = np.mean(np.asarray([templates[source] for source in sources]), axis=0)
    if component == "full":
        return {source: np.asarray(templates[source]) for source in sources}, common
    if component == "source_delta":
        return {source: np.asarray(templates[source]) - common for source in sources}, common
    raise ValueError(f"unsupported attribution component: {component}")


def _score_condition(
    query_records: Sequence[Mapping[str, Any]],
    templates: Mapping[str, np.ndarray],
    common: np.ndarray,
    store: _RepresentationStore,
    resolver: _ImageResolver,
    condition: str,
    representation: str,
    aggregator: str,
    scorer: str,
    component: str,
    suite_id: str,
    template_size: int,
    whitening: ChannelWhitening | None,
    modulation_supported: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    common_rows = []
    for query_record in query_records:
        sample_id = str(query_record.get("representation_sample_id", query_record["sample_id"]))
        query_id = str(query_record.get("query_instance_id", query_record["sample_id"]))
        query = np.asarray(store.load(condition, sample_id, representation))
        if aggregator in {"prnu_mle", "intensity_weighted"} and modulation_supported:
            intensity = luminance(resolver.rgb(condition, sample_id))
            if query.ndim == 3:
                intensity = intensity[..., np.newaxis]
            candidate_templates = {source: intensity * value for source, value in templates.items()}
            common_template = intensity * common
        else:
            candidate_templates = templates
            common_template = common
        block = []
        for candidate, template in candidate_templates.items():
            score, ncc, pce = _score_pair(query, template, scorer, whitening)
            block.append((candidate, score, ncc, pce))
        energies = np.asarray([np.sign(ncc) * ncc * ncc for _, _, ncc, _ in block])
        for index, (candidate, score, ncc, pce) in enumerate(block):
            other = np.delete(energies, index)
            signed_energy = float(energies[index])
            rows.append(
                {
                    "suite_id": suite_id,
                    "condition": condition,
                    "resolution": int(query_record["native_resolution"]),
                    "split": query_record["split"],
                    "query_id": query_id,
                    "prompt_id": query_record["prompt_id"],
                    "source_id": query_record["model_id"],
                    "candidate_source_id": candidate,
                    "representation": representation,
                    "aggregator": aggregator,
                    "scorer": scorer,
                    "component": component,
                    "template_size": template_size,
                    "raw_signed_ncc": ncc,
                    "raw_signed_energy": signed_energy,
                    "raw_pce": pce,
                    "gallery_complement_ratio_mean": gallery_complement_ratio(
                        abs(signed_energy), np.abs(other), reducer="mean", constant=1.0e-12
                    ),
                    "gallery_complement_ratio_median": gallery_complement_ratio(
                        abs(signed_energy), np.abs(other), reducer="median", constant=1.0e-12
                    ),
                    "score": score,
                    "score_name": scorer,
                    "label_match": query_record["model_id"] == candidate,
                    "modulation_supported": modulation_supported,
                }
            )
        common_score, _, _ = _score_pair(query, common_template, scorer, whitening)
        common_rows.append(
            {
                "suite_id": suite_id,
                "condition": condition,
                "resolution": int(query_record["native_resolution"]),
                "split": query_record["split"],
                "query_id": query_id,
                "prompt_id": query_record["prompt_id"],
                "source_id": query_record["model_id"],
                "representation": representation,
                "aggregator": aggregator,
                "scorer": scorer,
                "template_size": template_size,
                "common_component_score": common_score,
            }
        )
    return rows, common_rows


def _condition_id(
    representation: str,
    aggregator: str,
    scorer: str,
    component: str,
    template_size: int,
) -> str:
    return "__".join((representation, aggregator, scorer, component, f"n{int(template_size)}"))


def score_e02b_matrix(
    config_path: Path,
    repository_root: Path,
    generation_manifest_path: Path,
    data_root: Path,
    representation_manifest_path: Path,
    cache_root: Path,
    scores_path: Path,
    common_scores_path: Path,
    calibration_path: Path,
    *,
    condition: str = "native",
    derivative_manifest_path: Path | None = None,
    derivative_root: Path | None = None,
    profile: str = "matrix",
) -> dict[str, Any]:
    """Score the complete registered N=50 matrix on calibration and untouched test rows."""

    if profile not in {"matrix", "counterfactual", "resolution"}:
        raise ValueError("E02b scoring profile must be matrix, counterfactual, or resolution")
    config = load_e02b_config(config_path.resolve())
    state = repository_state(repository_root.resolve(), config)
    generation = _load_generation(generation_manifest_path.resolve(), config)
    derivatives = _load_derivatives(
        derivative_manifest_path.resolve() if derivative_manifest_path else None, config
    )
    resolver = _ImageResolver(generation, data_root, derivatives, derivative_root)
    store = _RepresentationStore(representation_manifest_path.resolve(), cache_root, config)
    primary_resolution = int(config["statistics"]["confirmatory_selection"]["primary_resolution"])
    resolutions = (
        [int(value) for value in config["design"]["resolutions"]]
        if profile == "resolution"
        else [primary_resolution]
    )
    template_size = int(config["statistics"]["confirmatory_selection"]["template_size"])
    rank_seed = int(config["design"]["split_rank_seed"])
    all_scores: list[dict[str, Any]] = []
    all_common: list[dict[str, Any]] = []
    calibration_evidence: dict[str, Any] = {
        "schema": "gfits.e02b-calibration/v1",
        "configuration_sha256": sha256_file(config_path.resolve()),
        "repository_state": repository_state_dict(state),
        "condition": condition,
        "resolutions": resolutions,
        "whitening": {},
        "multiplicative_diagnostics": {},
    }
    spatial = list(config["spatial_branch"]["extractors"])
    signatures = list(config["statistical_signature_branch"]["signatures"])
    for resolution in resolutions:
        base_records = [
            row
            for row in generation["records"]
            if int(row["native_resolution"]) == resolution and row["split"] != "auxiliary"
        ]
        for suite_id, suite in config["model_groups"].items():
            suite_sources = list(suite["models"])
            records = [row for row in base_records if row["model_id"] in suite_sources]
            source_templates = {
                source: [
                    row
                    for row in records
                    if row["model_id"] == source and row["split"] == "template"
                ]
                for source in suite_sources
            }
            queries = [row for row in records if row["split"] in {"calibration", "test"}]
            for representation in spatial + signatures:
                branch = "spatial_mean" if representation in spatial else "statistical_signature"
                extractor = (
                    representation
                    if branch == "spatial_mean"
                    else str(config["statistical_signature_branch"]["base_residual"])
                )
                whitening = (
                    _calibration_whitening(records, store, condition, representation)
                    if branch == "spatial_mean"
                    else None
                )
                evidence_key = f"r{resolution}::{suite_id}::{representation}"
                calibration_evidence["whitening"][evidence_key] = (
                    whitening.as_dict() if whitening is not None else None
                )
                modulation_supported = False
                if branch == "spatial_mean":
                    diagnostics = _multiplicative_diagnostics(
                        records, store, resolver, condition, representation, config
                    )
                    calibration_evidence["multiplicative_diagnostics"][evidence_key] = diagnostics
                    modulation_supported = bool(diagnostics["modulation_supported"])
                    aggregators = list(config["spatial_branch"]["aggregators"]["additive"])
                    if representation == "wavelet":
                        aggregators += list(
                            config["spatial_branch"]["aggregators"]["multiplicative_wavelet_only"]
                        )
                    else:
                        aggregators.append(
                            str(
                                config["spatial_branch"]["aggregators"]["non_wavelet_weighted_name"]
                            )
                        )
                    scorers = list(config["spatial_branch"]["scorers"]) + list(
                        config["spatial_branch"]["secondary_scorers"]
                    )
                else:
                    aggregators = list(config["statistical_signature_branch"]["aggregators"])
                    scorers = [str(config["statistical_signature_branch"]["scorer"])]
                if profile in {"counterfactual", "resolution"}:
                    aggregators = ["median"]
                    scorers = (
                        ["raw_signed_ncc_global", "raw_signed_ncc_channel_mean"]
                        if branch == "spatial_mean"
                        else ["signed_ncc"]
                    )
                for aggregator in aggregators:
                    raw_templates, selected_ids = _templates(
                        source_templates,
                        store,
                        resolver,
                        condition,
                        representation,
                        str(aggregator),
                        extractor,
                        template_size,
                        rank_seed,
                    )
                    for component in config["statistics"]["confirmatory_selection"][
                        "eligible_components"
                    ]:
                        component_templates, common = _component_templates(
                            raw_templates, str(component)
                        )
                        for scorer in scorers:
                            rows, common_rows = _score_condition(
                                queries,
                                component_templates,
                                common,
                                store,
                                resolver,
                                condition,
                                representation,
                                str(aggregator),
                                str(scorer),
                                str(component),
                                str(suite_id),
                                template_size,
                                whitening,
                                modulation_supported,
                            )
                            condition_id = _condition_id(
                                representation,
                                str(aggregator),
                                str(scorer),
                                str(component),
                                template_size,
                            )
                            for row in rows:
                                row["condition_id"] = condition_id
                                row["template_sample_ids"] = json.dumps(
                                    selected_ids[row["candidate_source_id"]],
                                    separators=(",", ":"),
                                )
                            for row in common_rows:
                                row["condition_id"] = condition_id
                            all_scores.extend(rows)
                            all_common.extend(common_rows)
    _write_csv(scores_path.resolve(), all_scores)
    _write_csv(common_scores_path.resolve(), all_common)
    write_json_atomic(calibration_path.resolve(), calibration_evidence)
    summary = {
        "schema": E02B_SCORE_SCHEMA,
        "configuration_sha256": sha256_file(config_path.resolve()),
        "repository_state": repository_state_dict(state),
        "condition": condition,
        "profile": profile,
        "resolutions": resolutions,
        "scores": str(scores_path.resolve()),
        "scores_sha256": sha256_file(scores_path.resolve()),
        "common_scores": str(common_scores_path.resolve()),
        "common_scores_sha256": sha256_file(common_scores_path.resolve()),
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": sha256_file(calibration_path.resolve()),
        "score_row_count": len(all_scores),
        "common_row_count": len(all_common),
        "condition_count": len(
            {(row["resolution"], row["suite_id"], row["condition_id"]) for row in all_scores}
        ),
    }
    summary_path = scores_path.resolve().with_suffix(".summary.json")
    write_json_atomic(summary_path, summary)
    return {"ok": True, **summary, "summary": str(summary_path)}


def _read_score_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["score"] = float(row["score"])
    return rows


def select_e02b_conditions(
    config_path: Path,
    scores_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Select one condition per suite using calibration rows and no test scores."""

    config = load_e02b_config(config_path.resolve())
    primary_resolution = int(config["statistics"]["confirmatory_selection"]["primary_resolution"])
    rows = []
    with scores_path.resolve().open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "calibration" or int(row["resolution"]) != primary_resolution:
                continue
            row["score"] = float(row["score"])
            rows.append(row)
    selected = {}
    calibration_table = []
    for suite_id in config["model_groups"]:
        candidates = sorted({row["condition_id"] for row in rows if row["suite_id"] == suite_id})
        suite_metrics = []
        for condition_id in candidates:
            subset = [
                row
                for row in rows
                if row["suite_id"] == suite_id and row["condition_id"] == condition_id
            ]
            metrics = attribution_metrics(subset)
            suite_metrics.append((float(metrics["macro_source_auroc"]), condition_id, metrics))
            calibration_table.append(
                {
                    "suite_id": suite_id,
                    "condition_id": condition_id,
                    "macro_source_auroc": metrics["macro_source_auroc"],
                    "pooled_pairwise_auroc": metrics["pooled_pairwise_auroc"],
                    "rank1": metrics["rank1"],
                }
            )
        if not suite_metrics:
            raise ValueError(f"no calibration conditions for suite {suite_id}")
        best_value = max(value for value, _, _ in suite_metrics)
        tied = sorted(
            (condition_id, metrics)
            for value, condition_id, metrics in suite_metrics
            if value == best_value
        )
        selected[suite_id] = {
            "condition_id": tied[0][0],
            "calibration_metrics": tied[0][1],
            "selection_split": "calibration",
            "criterion": "maximum_macro_source_auroc",
            "tie_break": "lexicographic_condition_id",
        }
    table_path = output_path.resolve().with_suffix(".csv")
    _write_csv(table_path, calibration_table)
    payload = {
        "schema": "gfits.e02b-calibration-selection/v1",
        "configuration_sha256": sha256_file(config_path.resolve()),
        "scores_sha256": sha256_file(scores_path.resolve()),
        "test_rows_inspected": False,
        "primary_resolution": primary_resolution,
        "selected": selected,
        "calibration_table": str(table_path),
        "calibration_table_sha256": sha256_file(table_path),
    }
    write_json_atomic(output_path.resolve(), payload)
    return {"ok": True, "selection": str(output_path.resolve()), **payload}


def _parse_condition_id(condition_id: str) -> dict[str, Any]:
    parts = condition_id.split("__")
    if len(parts) != 5 or not parts[-1].startswith("n"):
        raise ValueError(f"invalid E02b condition ID: {condition_id}")
    return {
        "representation": parts[0],
        "aggregator": parts[1],
        "scorer": parts[2],
        "component": parts[3],
        "template_size": int(parts[4][1:]),
    }


def _definition_scores(
    config: Mapping[str, Any],
    generation: Mapping[str, Any],
    store: _RepresentationStore,
    resolver: _ImageResolver,
    *,
    suite_id: str,
    condition: str,
    definition: Mapping[str, Any],
    query_split: str,
    template_ids_override: Mapping[str, Sequence[str]] | None = None,
    query_prompt_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], float]:
    resolution = int(config["statistics"]["confirmatory_selection"]["primary_resolution"])
    sources = list(config["model_groups"][suite_id]["models"])
    records = [
        row
        for row in generation["records"]
        if int(row["native_resolution"]) == resolution
        and row["model_id"] in sources
        and row["split"] != "auxiliary"
    ]
    source_templates = {
        source: [row for row in records if row["model_id"] == source and row["split"] == "template"]
        for source in sources
    }
    representation = str(definition["representation"])
    aggregator = str(definition["aggregator"])
    scorer = str(definition["scorer"])
    component = str(definition["component"])
    extractor = (
        representation
        if representation in config["spatial_branch"]["extractors"]
        else str(config["statistical_signature_branch"]["base_residual"])
    )
    if template_ids_override is None:
        raw_templates, selected_ids = _templates(
            source_templates,
            store,
            resolver,
            condition,
            representation,
            aggregator,
            extractor,
            int(definition["template_size"]),
            int(config["design"]["split_rank_seed"]),
        )
    else:
        selected_ids = {key: list(value) for key, value in template_ids_override.items()}
        raw_templates = _templates_from_ids(
            selected_ids,
            store,
            resolver,
            condition,
            representation,
            aggregator,
            extractor,
        )
    whitening = (
        _calibration_whitening(records, store, condition, representation)
        if representation in config["spatial_branch"]["extractors"]
        else None
    )
    diagnostics = None
    modulation_supported = False
    if representation in config["spatial_branch"]["extractors"]:
        diagnostics = _multiplicative_diagnostics(
            records, store, resolver, condition, representation, config
        )
        modulation_supported = bool(diagnostics["modulation_supported"])
    templates, common = _component_templates(raw_templates, component)
    queries = [row for row in records if row["split"] == query_split]
    if query_prompt_ids is not None:
        prompt_map: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in queries:
            prompt_map[str(row["prompt_id"])].append(row)
        resampled = []
        for occurrence, prompt_id in enumerate(query_prompt_ids):
            for row in prompt_map[str(prompt_id)]:
                copy = dict(row)
                copy["representation_sample_id"] = row["sample_id"]
                copy["query_instance_id"] = f"{row['sample_id']}__boot{occurrence}"
                resampled.append(copy)
        queries = resampled
    rows, _ = _score_condition(
        queries,
        templates,
        common,
        store,
        resolver,
        condition,
        representation,
        aggregator,
        scorer,
        component,
        suite_id,
        int(definition["template_size"]),
        whitening,
        modulation_supported,
    )
    variance_values = []
    for identifiers in selected_ids.values():
        arrays = np.asarray(
            [store.load(condition, sample_id, representation) for sample_id in identifiers]
        )
        variance_values.append(float(np.mean(np.var(arrays, axis=0))))
    return rows, selected_ids, float(np.mean(variance_values))


def _prompt_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]], *, draws: int, confidence: float, seed: int
) -> tuple[float, float]:
    prompt_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt_rows[str(row["prompt_id"])].append(row)
    prompts = sorted(prompt_rows)
    rng = np.random.default_rng(seed)
    values = []
    for draw in range(draws):
        sampled = rng.choice(prompts, size=len(prompts), replace=True)
        block = []
        for occurrence, prompt in enumerate(sampled):
            for row in prompt_rows[str(prompt)]:
                copy = dict(row)
                copy["query_id"] = f"{row['query_id']}__promptboot{draw}_{occurrence}"
                block.append(copy)
        values.append(float(attribution_metrics(block)["macro_source_auroc"]))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def _evaluate_resolution_profile(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select within calibration at each resolution and summarize untouched test rows."""

    rows = _read_score_rows(path.resolve())
    results = []
    for suite_id in config["model_groups"]:
        suite_rows = [row for row in rows if row["suite_id"] == suite_id]
        for resolution in config["design"]["resolutions"]:
            resolution_rows = [
                row for row in suite_rows if int(row["resolution"]) == int(resolution)
            ]
            candidates = sorted({row["condition_id"] for row in resolution_rows})
            ranked = []
            for condition_id in candidates:
                calibration = [
                    row
                    for row in resolution_rows
                    if row["condition_id"] == condition_id and row["split"] == "calibration"
                ]
                if calibration:
                    ranked.append(
                        (
                            float(attribution_metrics(calibration)["macro_source_auroc"]),
                            condition_id,
                        )
                    )
            if not ranked:
                raise ValueError(f"resolution profile lacks {suite_id}/{resolution}")
            best_value = max(value for value, _ in ranked)
            selected = min(condition for value, condition in ranked if value == best_value)
            test = [
                row
                for row in resolution_rows
                if row["condition_id"] == selected and row["split"] == "test"
            ]
            metrics = attribution_metrics(test)
            results.append(
                {
                    "suite_id": suite_id,
                    "resolution": int(resolution),
                    "condition_id": selected,
                    "calibration_macro_source_auroc": best_value,
                    "test_macro_source_auroc": metrics["macro_source_auroc"],
                    "test_rank1": metrics["rank1"],
                    "test_mrr": metrics["mean_reciprocal_rank"],
                }
            )
    return results


def evaluate_e02b(
    config_path: Path,
    repository_root: Path,
    generation_manifest_path: Path,
    data_root: Path,
    representation_manifest_path: Path,
    cache_root: Path,
    scores_path: Path,
    selection_path: Path,
    output_dir: Path,
    *,
    condition: str = "native",
    derivative_manifest_path: Path | None = None,
    derivative_root: Path | None = None,
    resolution_score_path: Path | None = None,
    counterfactual_score_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Run untouched-test source inference, two-way bootstrap, counterfactuals, and Gate."""

    config = load_e02b_config(config_path.resolve())
    state = repository_state(repository_root.resolve(), config)
    generation = _load_generation(generation_manifest_path.resolve(), config)
    derivatives = _load_derivatives(
        derivative_manifest_path.resolve() if derivative_manifest_path else None, config
    )
    resolver = _ImageResolver(generation, data_root, derivatives, derivative_root)
    store = _RepresentationStore(representation_manifest_path.resolve(), cache_root, config)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("scores_sha256") != sha256_file(scores_path.resolve()):
        raise ValueError("calibration selection does not match the score table")
    all_rows = _read_score_rows(scores_path.resolve())
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = config["statistics"]
    bootstrap_settings = settings["bootstrap"]
    permutation_settings = settings["permutations"]
    suite_results: dict[str, dict[str, Any]] = {}
    source_metric_rows = []
    confusion_rows = []
    bias_rows = []
    template_curve_rows = []
    confirmatory_pvalues = []
    confirmatory_pvalue_targets = []
    for suite_index, suite_id in enumerate(config["model_groups"]):
        condition_id = selection["selected"][suite_id]["condition_id"]
        definition = _parse_condition_id(condition_id)
        test_rows = [
            row
            for row in all_rows
            if row["suite_id"] == suite_id
            and row["condition_id"] == condition_id
            and row["split"] == "test"
        ]
        metrics = attribution_metrics(test_rows)
        prompt_permutation = prompt_block_source_label_permutation(
            test_rows,
            draws=int(permutation_settings["prompt_block_source_label"]["draws"]),
            seed=int(permutation_settings["prompt_block_source_label"]["seed"]) + suite_index,
        )
        template_permutation = template_source_label_permutation(
            test_rows,
            draws=int(permutation_settings["template_source_label"]["draws"]),
            seed=int(permutation_settings["template_source_label"]["seed"]) + suite_index,
        )
        source_families = {
            source: str(config["models"][source]["family"])
            for source in config["model_groups"][suite_id]["models"]
        }
        hierarchical = hierarchical_family_prompt_permutation(
            test_rows,
            source_families,
            draws=int(permutation_settings["hierarchical_family_prompt"]["draws"]),
            seed=int(permutation_settings["hierarchical_family_prompt"]["seed"]) + suite_index,
        )
        sign_flip = auxiliary_sign_flip_test(
            test_rows,
            draws=int(permutation_settings["auxiliary_sign_flip"]["draws"]),
            seed=int(permutation_settings["auxiliary_sign_flip"]["seed"]) + suite_index,
        )
        _, template_ids, _ = _definition_scores(
            config,
            generation,
            store,
            resolver,
            suite_id=suite_id,
            condition=condition,
            definition=definition,
            query_split="test",
        )
        query_prompts = sorted(
            {
                str(row["prompt_id"])
                for row in generation["records"]
                if row["split"] == "test"
                and row["model_id"] in config["model_groups"][suite_id]["models"]
                and int(row["native_resolution"])
                == int(settings["confirmatory_selection"]["primary_resolution"])
            }
        )

        def score_builder(
            sampled_templates: Mapping[str, Sequence[str]],
            sampled_prompts: Sequence[str],
            *,
            _suite_id: str = suite_id,
            _definition: Mapping[str, Any] = definition,
        ) -> Sequence[Mapping[str, Any]]:
            rows, _, _ = _definition_scores(
                config,
                generation,
                store,
                resolver,
                suite_id=_suite_id,
                condition=condition,
                definition=_definition,
                query_split="test",
                template_ids_override=sampled_templates,
                query_prompt_ids=sampled_prompts,
            )
            return rows

        bootstrap = two_way_bootstrap(
            template_ids,
            query_prompts,
            score_builder,
            draws=int(bootstrap_settings["draws"]),
            confidence=float(bootstrap_settings["confidence"]),
            seed=int(bootstrap_settings["seed"]) + suite_index,
        )
        suite_results[suite_id] = {
            "condition_id": condition_id,
            "selection": selection["selected"][suite_id],
            "metrics": metrics,
            "bootstrap": bootstrap,
            "prompt_block_permutation": prompt_permutation,
            "template_source_permutation": template_permutation,
            "hierarchical_permutation": hierarchical,
            "auxiliary_sign_flip": sign_flip,
        }
        confirmatory_pvalues.append(float(prompt_permutation["rank1_pvalue"]))
        confirmatory_pvalue_targets.append((suite_id, "prompt_block_rank1_permutation"))
        for test in (template_permutation, hierarchical):
            confirmatory_pvalues.append(float(test["pvalue"]))
            confirmatory_pvalue_targets.append((suite_id, str(test["kind"])))
        for source, row in metrics["per_source"].items():
            source_metric_rows.append({"suite_id": suite_id, "source_id": source, **row})
        for actual, predictions in metrics["confusion_matrix"].items():
            for predicted, count in predictions.items():
                confusion_rows.append(
                    {
                        "suite_id": suite_id,
                        "actual_source_id": actual,
                        "predicted_source_id": predicted,
                        "count": count,
                    }
                )
        for candidate, row in metrics["candidate_template_bias"].items():
            bias_rows.append({"suite_id": suite_id, "candidate_source_id": candidate, **row})
        selected_curve_definition = dict(definition)
        for template_size in config["spatial_branch"]["template_sizes"]:
            selected_curve_definition["template_size"] = int(template_size)
            selected_curve_scores, _, selected_template_variance = _definition_scores(
                config,
                generation,
                store,
                resolver,
                suite_id=suite_id,
                condition=condition,
                definition=selected_curve_definition,
                query_split="test",
            )
            selected_curve_metrics = attribution_metrics(selected_curve_scores)
            template_curve_rows.append(
                {
                    "suite_id": suite_id,
                    "curve_kind": "selected_condition",
                    "representation": definition["representation"],
                    "n_template": int(template_size),
                    "macro_source_auroc": selected_curve_metrics["macro_source_auroc"],
                    "rank1": selected_curve_metrics["rank1"],
                    "template_variance": selected_template_variance,
                }
            )
        for representation in config["spatial_branch"]["extractors"]:
            curve_definition = {
                "representation": representation,
                "aggregator": "median",
                "scorer": "raw_signed_ncc_channel_mean",
                "component": "source_delta",
                "template_size": 50,
            }
            for template_size in config["spatial_branch"]["template_sizes"]:
                curve_definition["template_size"] = int(template_size)
                curve_scores, _, template_variance = _definition_scores(
                    config,
                    generation,
                    store,
                    resolver,
                    suite_id=suite_id,
                    condition=condition,
                    definition=curve_definition,
                    query_split="test",
                )
                curve_metrics = attribution_metrics(curve_scores)
                template_curve_rows.append(
                    {
                        "suite_id": suite_id,
                        "curve_kind": "major_extractor_default",
                        "representation": representation,
                        "n_template": int(template_size),
                        "macro_source_auroc": curve_metrics["macro_source_auroc"],
                        "rank1": curve_metrics["rank1"],
                        "template_variance": template_variance,
                    }
                )
    adjusted = holm_adjust(confirmatory_pvalues)
    for (suite_id, kind), adjusted_p in zip(confirmatory_pvalue_targets, adjusted, strict=True):
        suite_results[suite_id].setdefault("holm_confirmatory_pvalues", {})[kind] = adjusted_p
    counterfactual_rows = []
    low_bit_uniform_results = {}
    score_paths = list(counterfactual_score_paths or ())
    for path_index, path in enumerate(score_paths):
        rows = _read_score_rows(path.resolve())
        if not rows:
            continue
        counterfactual_id = str(rows[0]["condition"])
        for suite_id in config["model_groups"]:
            for representation in config["spatial_branch"]["extractors"]:
                subset = [
                    row
                    for row in rows
                    if row["suite_id"] == suite_id
                    and row["representation"] == representation
                    and row["aggregator"] == "median"
                    and row["scorer"] == "raw_signed_ncc_channel_mean"
                    and row["component"] == "source_delta"
                    and row["split"] == "test"
                ]
                if not subset:
                    continue
                metrics = attribution_metrics(subset)
                ci_low, ci_high = _prompt_bootstrap_ci(
                    subset,
                    draws=int(bootstrap_settings["draws"]),
                    confidence=float(bootstrap_settings["confidence"]),
                    seed=int(bootstrap_settings["seed"]) + 100 + path_index,
                )
                counterfactual_rows.append(
                    {
                        "counterfactual_id": counterfactual_id,
                        "suite_id": suite_id,
                        "representation": representation,
                        "macro_source_auroc": metrics["macro_source_auroc"],
                        "rank1": metrics["rank1"],
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                    }
                )
                if counterfactual_id == "canonical_uniform_export" and representation == "low_bit":
                    low_bit_uniform_results[suite_id] = counterfactual_rows[-1]
    resolution_rows = (
        _evaluate_resolution_profile(resolution_score_path, config)
        if resolution_score_path is not None
        else []
    )
    selected_curve = {
        suite_id: [
            row
            for row in template_curve_rows
            if row["suite_id"] == suite_id and row["curve_kind"] == "selected_condition"
        ]
        for suite_id in suite_results
    }
    gate = e02b_gate(
        suite_results,
        selected_curve,
        low_bit_uniform_results,
        alpha=float(settings["alpha"]),
    )
    tables = {
        "source-level-metrics.csv": source_metric_rows,
        "confusion-matrix.csv": confusion_rows,
        "candidate-template-bias.csv": bias_rows,
        "template-number-curves.csv": template_curve_rows,
        "counterfactual-results.csv": counterfactual_rows,
        "resolution-profile.csv": resolution_rows,
    }
    table_hashes = {}
    for filename, rows in tables.items():
        if rows:
            path = output_dir / filename
            _write_csv(path, rows)
            table_hashes[filename] = sha256_file(path)
    summary = {
        "schema": E02B_RESULT_SCHEMA,
        "phase": "E02b",
        "generated_at_utc": _utc_now(),
        "claim_boundary": (
            "controlled generator-source confirmation only; " "no downstream software calibration"
        ),
        "configuration_sha256": sha256_file(config_path.resolve()),
        "generation_manifest_sha256": sha256_file(generation_manifest_path.resolve()),
        "representation_manifest_sha256": sha256_file(representation_manifest_path.resolve()),
        "scores_sha256": sha256_file(scores_path.resolve()),
        "selection_sha256": sha256_file(selection_path.resolve()),
        "resolution_scores_sha256": (
            sha256_file(resolution_score_path.resolve()) if resolution_score_path else None
        ),
        "repository_state": repository_state_dict(state),
        "suite_results": suite_results,
        "confirmatory_holm_family": {
            f"{suite}/{kind}": value
            for (suite, kind), value in zip(confirmatory_pvalue_targets, adjusted, strict=True)
        },
        "e02b_gate": gate,
        "artifacts": table_hashes,
    }
    summary_path = output_dir / "summary.json"
    write_json_atomic(summary_path, summary)
    return {
        "ok": True,
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "e02b_gate": gate,
    }


def generate_e02b_report(
    config_path: Path,
    summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Render the controlled-stage report and the 73-item audit traceability matrix."""

    load_e02b_config(config_path.resolve())
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != E02B_RESULT_SCHEMA:
        raise ValueError("unsupported E02b result summary")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = summary["e02b_gate"]
    lines = [
        "# E02b Controlled Generator-Fingerprint Confirmation",
        "",
        "## Outcome",
        "",
        f"- Gate status: **{gate['status']}**.",
        f"- Registered decision: `{gate['decision']}`.",
        (
            f"- Runtime code commit: `{summary['repository_state']['commit']}` "
            f"(clean: `{summary['repository_state']['clean']}`)."
        ),
        (
            "- Claim boundary: controlled generator-source evidence only; this phase does "
            "not apply a downstream software pipeline and cannot validate G-FITS."
        ),
        "",
        "## Six registered E02b checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, value in gate["checks"].items():
        lines.append(f"| `{name}` | **{'PASS' if value else 'FAIL'}** |")
    lines.extend(
        [
            "",
            "## Confirmatory suite results",
            "",
            (
                "The condition for each suite was chosen using calibration macro "
                "source-level AUROC. Test rows were not read by the selection command."
            ),
            "",
            (
                "| Suite | Selected condition | Pooled AUROC | Macro source AUROC | "
                "Rank-1 | Chance | Two-way 95% CI |"
            ),
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for suite_id, result in summary["suite_results"].items():
        metrics = result["metrics"]
        interval = result["bootstrap"]["intervals"]["two_way"]
        lines.append(
            (
                "| {suite} | `{condition}` | {pooled:.4f} | {macro:.4f} | "
                "{rank1:.4f} | {chance:.4f} | [{low:.4f}, {high:.4f}] |"
            ).format(
                suite=suite_id,
                condition=result["condition_id"],
                pooled=float(metrics["pooled_pairwise_auroc"]),
                macro=float(metrics["macro_source_auroc"]),
                rank1=float(metrics["rank1"]),
                chance=float(metrics["chance_rank1"]),
                low=float(interval["low"]),
                high=float(interval["high"]),
            )
        )
    lines.extend(
        [
            "",
            "## Source-level results",
            "",
            "| Suite | Source | AUROC | Rank-1 | H1 n | H0 n |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for suite_id, result in summary["suite_results"].items():
        metrics = result["metrics"]
        for source, values in metrics["per_source"].items():
            lines.append(
                f"| {suite_id} | {source} | {float(values['auroc']):.4f} | "
                f"{float(metrics['per_source_rank1'][source]):.4f} | "
                f"{int(values['h1_count'])} | {int(values['h0_count'])} |"
            )
    lines.extend(
        [
            "",
            "## Statistical controls",
            "",
            (
                "- The confirmatory interval is a two-way bootstrap that rebuilds "
                "templates and resamples `prompt_id` blocks across every source."
            ),
            (
                "- Prompt-block query-label, whole-template label, and hierarchical "
                "family/prompt permutations are confirmatory; sign-flip is retained "
                "only as an auxiliary symmetric-null diagnostic."
            ),
            (
                "- Holm correction is applied to the registered confirmatory permutation "
                "family. Multiple extractors and aggregators share data and are "
                "method-sensitivity conditions, not independent replications."
            ),
            "",
            "## Counterfactual and shortcut audits",
            "",
            (
                "The machine-readable tables record uniform re-quantization/export, "
                "random +/-1 LSB perturbation, six-bit conversion, and pixel-checked PNG "
                "re-encoding. Results are split by source; candidate H0 mean/std and "
                "prediction frequency are saved so a universally high-scoring template "
                "cannot hide behind pooled AUROC."
            ),
            "",
            "## Supported conclusions",
            "",
        ]
    )
    if gate["passed"]:
        lines.append(
            "- Under the registered controlled writer, prompt, seed, resolution, and "
            "model revisions, both cross-family and near-family suites contain repeatable "
            "source-related evidence meeting the E02b engineering gate."
        )
    else:
        lines.append(
            "- The controlled experiment is a valid falsification/restriction result. "
            "The registered evidence does not support a general fixed model-level "
            "fingerprint claim."
        )
    lines.extend(
        [
            "",
            "## Unsupported conclusions",
            "",
            "- Different generator fingerprints are independent.",
            "- A downstream AIGC software-noise term exists.",
            "- FITS/G-FITS aligns pipeline-specific null distributions.",
            "- Known-template attribution is a universal unknown-generator detector.",
            "",
            "## Next-stage enforcement",
            "",
            (
                "E03 configuration/hierarchy analysis is permitted by the registered "
                "Gate. E04 and E05 remain blocked until their own prerequisites pass."
                if gate["passed"]
                else "E03, E04, and E05 formal claims remain blocked. The project must "
                "stop or restrict the fixed-generator-fingerprint mainline rather than "
                "bypass this Gate."
            ),
            "",
            "## Reproducibility evidence",
            "",
            f"- Configuration SHA-256: `{summary['configuration_sha256']}`.",
            f"- Generation manifest SHA-256: `{summary['generation_manifest_sha256']}`.",
            f"- Representation manifest SHA-256: `{summary['representation_manifest_sha256']}`.",
            f"- Score table SHA-256: `{summary['scores_sha256']}`.",
            f"- Calibration selection SHA-256: `{summary['selection_sha256']}`.",
            "",
        ]
    )
    report_path = output_dir / "E02B_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    source_matrix = (
        config_path.resolve().parent.parent / "reports" / "e02b" / "PRO_AUDIT_TRACEABILITY.md"
    )
    if not source_matrix.is_file():
        raise ValueError(f"73-item traceability source is missing: {source_matrix}")
    matrix_path = output_dir / "AUDIT_IMPLEMENTATION_MATRIX.md"
    matrix_path.write_text(source_matrix.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "ok": True,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "matrix": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "gate": gate,
    }
