"""Residual, fingerprint, scoring, and statistical gates for Phase E02."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from scipy.stats import trim_mean
from sklearn.metrics import average_precision_score, roc_auc_score

from gfits.e02_data import (
    load_e02_config,
    load_e02_manifest,
    verify_e02_manifest,
)
from gfits.manifest import sha256_file
from gfits.matching import fits, fits_plus, log_ratio, normalized_cross_correlation

E02_RESIDUAL_SCHEMA = "gfits.e02-residual-manifest/v1"
E02_BANK_SCHEMA = "gfits.e02-fingerprint-bank/v1"
E02_RESULT_SCHEMA = "gfits.e02-generator-signal/v1"

_LUMINANCE = np.asarray([0.29893602, 0.58704307, 0.11402090], dtype=np.float32)
_NOISEPRINT_LUMINANCE = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
_SRM_KERNELS = np.asarray(
    [
        [[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]],
        [[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]],
    ],
    dtype=np.float32,
)
_SRM_KERNELS[2] /= 4.0


def _canonical_git_file_hash(root: Path, relative_path: str) -> str:
    content = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def _load_prnu(root: Path, extractor: Mapping[str, Any]) -> ModuleType:
    root = root.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != extractor["commit"]:
        raise ValueError(f"unexpected prnu-python commit: {commit}")
    relative = "prnu/functions.py"
    if _canonical_git_file_hash(root, relative) != extractor["functions_sha256"]:
        raise ValueError("locked prnu-python functions.py SHA-256 mismatch")
    source = root / relative
    specification = importlib.util.spec_from_file_location("gfits_e02_prnu", source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load upstream module: {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _decode_rgb(path: Path, record: Mapping[str, Any]) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.size != (int(record["width"]), int(record["height"])):
            raise ValueError(f"geometry changed after manifest creation: {path}")
        if image.mode == "RGB":
            array = np.asarray(image)
        elif image.mode == "RGBA":
            if image.getchannel("A").getextrema() != (255, 255):
                raise ValueError(f"non-opaque alpha is not admissible: {path}")
            array = np.asarray(image)[..., :3]
        else:
            raise ValueError(f"unsupported native image mode: {path} ({image.mode})")
    if array.dtype != np.uint8 or array.shape != (record["height"], record["width"], 3):
        raise ValueError(f"unsupported decoded representation: {path}")
    return np.ascontiguousarray(array)


def _luminance(image: np.ndarray) -> np.ndarray:
    return np.tensordot(image.astype(np.float32), _LUMINANCE, axes=([-1], [0])) / 255.0


def _wavelet_residual(
    image: np.ndarray,
    upstream: ModuleType,
    settings: Mapping[str, Any],
) -> np.ndarray:
    return upstream.extract_single(
        image,
        levels=int(settings["levels"]),
        sigma=float(settings["sigma"]),
    ).astype(np.float32)


def _srm_residual(image: np.ndarray) -> np.ndarray:
    gray = _luminance(image)
    return np.stack(
        [convolve(gray, kernel, mode="reflect") for kernel in _SRM_KERNELS],
        axis=-1,
    ).astype(np.float32)


def _low_bit_residual(image: np.ndarray, bit_count: int) -> np.ndarray:
    mask = (1 << bit_count) - 1
    scale = float(mask)
    return ((np.bitwise_and(image, mask).astype(np.float32) / scale) - 0.5).astype(np.float32)


class _NoiseprintSession:
    """TensorFlow-compatibility adapter around an unmodified locked checkout."""

    def __init__(self, root: Path, settings: Mapping[str, Any]):
        root = root.resolve()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != settings["commit"]:
            raise ValueError(f"unexpected Noiseprint commit: {commit}")
        try:
            import tensorflow.compat.v1 as tf
        except ImportError as error:
            raise ImportError(
                "Noiseprint requires the isolated fits-noiseprint environment "
                "with tensorflow-cpu==2.15.1"
            ) from error
        tf.disable_v2_behavior()
        self.tf = tf
        self.root = root
        sys.path.insert(0, str(root))
        sys.modules["tensorflow"] = tf
        try:
            module = importlib.import_module("noiseprint.noiseprint")
        finally:
            sys.path.remove(str(root))
        model_root = root / "noiseprint" / "nets" / str(settings["model"])
        checkpoint = model_root / "model"
        for filename, expected_hash in settings["checkpoint_sha256"].items():
            model_file = model_root / filename
            if not model_file.is_file():
                raise ValueError(f"Noiseprint checkpoint file is missing: {model_file}")
            if sha256_file(model_file) != expected_hash:
                raise ValueError(f"Noiseprint checkpoint SHA-256 mismatch: {model_file}")
        self.module = module
        self.session = tf.Session(config=module.configSess)
        module.saver.restore(self.session, str(checkpoint))

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = (
            np.tensordot(
                image.astype(np.float32),
                _NOISEPRINT_LUMINANCE,
                axes=([-1], [0]),
            )
            / 256.0
        )
        output = self.session.run(
            self.module.net.output,
            feed_dict={self.module.x_data: gray[np.newaxis, :, :, np.newaxis]},
        )
        return np.squeeze(output).astype(np.float32)

    def close(self) -> None:
        self.session.close()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_residual_manifest(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != E02_RESIDUAL_SCHEMA:
        raise ValueError("unsupported E02 residual manifest schema")
    if payload.get("configuration_sha256") != sha256_file(Path(config["_config_path"])):
        raise ValueError("residuals were extracted under a different configuration")
    return payload


def extract_e02_residuals(
    config_path: Path,
    source_manifest_path: Path,
    data_root: Path,
    upstream_root: Path,
    cache_root: Path,
    residual_manifest_path: Path,
    *,
    extractor_names: Sequence[str] | None = None,
    noiseprint_root: Path | None = None,
) -> dict[str, Any]:
    """Extract registered residuals into an external, hash-addressed cache."""

    config = load_e02_config(config_path.resolve())
    source_manifest = load_e02_manifest(source_manifest_path.resolve(), config)
    verification = verify_e02_manifest(source_manifest, data_root.resolve())
    if not verification["passed"]:
        raise ValueError(f"source manifest verification failed: {verification['issues'][:5]}")
    requested = list(extractor_names or config["extractors"])
    unknown = set(requested) - set(config["extractors"])
    if unknown:
        raise ValueError(f"unknown E02 extractors: {sorted(unknown)}")
    upstream = (
        _load_prnu(upstream_root, config["extractors"]["wavelet"])
        if "wavelet" in requested
        else None
    )
    noiseprint = None
    if "noiseprint" in requested:
        if noiseprint_root is None:
            raise ValueError("--noiseprint-root is required for the Noiseprint extractor")
        noiseprint = _NoiseprintSession(noiseprint_root, config["extractors"]["noiseprint"])
    residual_manifest_path = residual_manifest_path.resolve()
    if residual_manifest_path.is_file():
        existing = _load_residual_manifest(residual_manifest_path, config)
        rows = {(row["sample_id"], row["extractor"]): row for row in existing.get("records", [])}
    else:
        rows = {}
    try:
        for sample_index, record in enumerate(source_manifest["records"], start=1):
            image = _decode_rgb(data_root / Path(record["relative_path"]), record)
            intensity = _luminance(image).astype(np.float32)
            for extractor in requested:
                if extractor == "wavelet":
                    assert upstream is not None
                    residual = _wavelet_residual(image, upstream, config["extractors"][extractor])
                elif extractor == "srm":
                    residual = _srm_residual(image)
                elif extractor == "low_bit":
                    residual = _low_bit_residual(
                        image,
                        int(config["extractors"][extractor]["bit_count"]),
                    )
                else:
                    assert noiseprint is not None
                    residual = noiseprint.extract(image)
                relative_cache = Path(extractor) / record["dataset_id"] / record["suite_id"]
                cache_path = cache_root.resolve() / relative_cache / f"{record['sha256']}.npz"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, residual=residual, intensity=intensity)
                rows[(record["sample_id"], extractor)] = {
                    "sample_id": record["sample_id"],
                    "dataset_id": record["dataset_id"],
                    "suite_id": record["suite_id"],
                    "source_id": record["source_id"],
                    "split": record["split"],
                    "extractor": extractor,
                    "source_sha256": record["sha256"],
                    "cache_path": str(cache_path),
                    "cache_sha256": sha256_file(cache_path),
                    "residual_shape": list(residual.shape),
                    "residual_dtype": str(residual.dtype),
                    "intensity_shape": list(intensity.shape),
                }
            if sample_index % 20 == 0 or sample_index == len(source_manifest["records"]):
                print(
                    f"residual_samples={sample_index}/{len(source_manifest['records'])}", flush=True
                )
    finally:
        if noiseprint is not None:
            noiseprint.close()
    ordered = sorted(rows.values(), key=lambda row: (row["sample_id"], row["extractor"]))
    payload = {
        "schema": E02_RESIDUAL_SCHEMA,
        "phase": "E02",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "configuration_sha256": sha256_file(config_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path.resolve()),
        "record_count": len(ordered),
        "extractors_present": sorted({row["extractor"] for row in ordered}),
        "records": ordered,
    }
    residual_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    residual_manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "residual_manifest": str(residual_manifest_path),
        "record_count": len(ordered),
        "extractors_present": payload["extractors_present"],
        "manifest_sha256": sha256_file(residual_manifest_path),
    }


def _load_cached(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return payload["residual"], payload["intensity"]


def _signed_ncc(left: np.ndarray, right: np.ndarray) -> float:
    """Compute NCC over every residual element without changing its values."""

    if left.shape != right.shape:
        raise ValueError(f"NCC shape mismatch: {left.shape} != {right.shape}")
    if left.ndim == 2:
        return normalized_cross_correlation(left, right)
    if left.ndim == 3:
        return normalized_cross_correlation(
            left.reshape(left.shape[0], -1),
            right.reshape(right.shape[0], -1),
        )
    raise ValueError(f"unsupported residual rank for NCC: {left.ndim}")


def _aggregate_fingerprint(
    residuals: np.ndarray,
    intensities: np.ndarray,
    aggregator: str,
    config: Mapping[str, Any],
) -> np.ndarray:
    if aggregator == "mean":
        result = np.mean(residuals, axis=0)
    elif aggregator == "median":
        result = np.median(residuals, axis=0)
    elif aggregator == "trimmed_mean":
        result = trim_mean(
            residuals,
            proportiontocut=float(config["aggregators"]["trimmed_mean_fraction"]),
            axis=0,
        )
    elif aggregator == "prnu_mle":
        weights = intensities
        while weights.ndim < residuals.ndim:
            weights = weights[..., np.newaxis]
        numerator = np.sum(residuals * weights, axis=0)
        denominator = np.sum(np.square(weights), axis=0)
        result = numerator / (
            denominator + float(config["aggregators"]["prnu_mle_denominator_epsilon"])
        )
    else:
        raise ValueError(f"unsupported fingerprint aggregator: {aggregator}")
    return np.asarray(result, dtype=np.float32)


def build_e02_fingerprint_bank(
    config_path: Path,
    source_manifest_path: Path,
    residual_manifest_path: Path,
    bank_root: Path,
    bank_manifest_path: Path,
) -> dict[str, Any]:
    """Build every pre-registered source fingerprint from template records only."""

    config = load_e02_config(config_path.resolve())
    source_manifest = load_e02_manifest(source_manifest_path.resolve(), config)
    residual_manifest = _load_residual_manifest(residual_manifest_path.resolve(), config)
    source_by_id = {row["sample_id"]: row for row in source_manifest["records"]}
    residual_rows = [
        row
        for row in residual_manifest["records"]
        if source_by_id[row["sample_id"]]["split"] == "template"
    ]
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in residual_rows:
        grouped[(row["suite_id"], row["source_id"], row["extractor"])].append(row)
    rows: list[dict[str, Any]] = []
    template_count = int(config["selection"]["splits"]["template"])
    for (suite_id, source_id, extractor), group in sorted(grouped.items()):
        if len(group) != template_count:
            raise ValueError(
                f"template count for {suite_id}/{source_id}/{extractor} is {len(group)}"
            )
        arrays = [_load_cached(Path(row["cache_path"])) for row in group]
        residuals = np.stack([value[0] for value in arrays])
        intensities = np.stack([value[1] for value in arrays])
        for aggregator in config["aggregators"]["names"]:
            fingerprint = _aggregate_fingerprint(residuals, intensities, aggregator, config)
            path = bank_root.resolve() / suite_id / extractor / aggregator / f"{source_id}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, fingerprint, allow_pickle=False)
            rows.append(
                {
                    "suite_id": suite_id,
                    "source_id": source_id,
                    "extractor": extractor,
                    "aggregator": aggregator,
                    "template_count": len(group),
                    "template_sample_ids": "|".join(sorted(row["sample_id"] for row in group)),
                    "fingerprint_path": str(path),
                    "fingerprint_sha256": sha256_file(path),
                    "fingerprint_shape": list(fingerprint.shape),
                    "fingerprint_dtype": str(fingerprint.dtype),
                }
            )
    payload = {
        "schema": E02_BANK_SCHEMA,
        "phase": "E02",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "configuration_sha256": sha256_file(config_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path.resolve()),
        "residual_manifest_sha256": sha256_file(residual_manifest_path.resolve()),
        "record_count": len(rows),
        "records": rows,
    }
    bank_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    bank_manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "bank_manifest": str(bank_manifest_path),
        "fingerprint_count": len(rows),
        "manifest_sha256": sha256_file(bank_manifest_path),
    }


def _load_bank_manifest(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != E02_BANK_SCHEMA:
        raise ValueError("unsupported E02 fingerprint-bank schema")
    if payload.get("configuration_sha256") != sha256_file(Path(config["_config_path"])):
        raise ValueError("fingerprint bank was built under a different configuration")
    return payload


def score_e02_pairs(
    config_path: Path,
    source_manifest_path: Path,
    residual_manifest_path: Path,
    bank_manifest_path: Path,
    scores_path: Path,
) -> dict[str, Any]:
    """Score every calibration/test query against every same-geometry source."""

    config = load_e02_config(config_path.resolve())
    source_manifest = load_e02_manifest(source_manifest_path.resolve(), config)
    residual_manifest = _load_residual_manifest(residual_manifest_path.resolve(), config)
    bank_manifest = _load_bank_manifest(bank_manifest_path.resolve(), config)
    source_by_id = {row["sample_id"]: row for row in source_manifest["records"]}
    fingerprints = {
        (row["suite_id"], row["source_id"], row["extractor"], row["aggregator"]): np.load(
            row["fingerprint_path"], allow_pickle=False
        )
        for row in bank_manifest["records"]
    }
    source_ids_by_suite = {
        suite_id: [
            source_id
            for dataset in config["datasets"].values()
            for sid, suite in dataset["suites"].items()
            if sid == suite_id
            for source_id in suite["sources"]
        ]
        for dataset in config["datasets"].values()
        for suite_id in dataset["suites"]
    }
    stabilizer = float(config["scoring"]["stabilizer"])
    rows: list[dict[str, Any]] = []
    query_rows = [
        row
        for row in residual_manifest["records"]
        if source_by_id[row["sample_id"]]["split"] in {"calibration", "test"}
    ]
    for query_index, residual_row in enumerate(query_rows, start=1):
        source_record = source_by_id[residual_row["sample_id"]]
        residual, intensity = _load_cached(Path(residual_row["cache_path"]))
        suite_id = residual_row["suite_id"]
        extractor = residual_row["extractor"]
        candidates = source_ids_by_suite[suite_id]
        for aggregator in config["aggregators"]["names"]:
            signed_scores: dict[str, float] = {}
            for candidate in candidates:
                fingerprint = fingerprints[(suite_id, candidate, extractor, aggregator)]
                predicted = fingerprint
                if aggregator == "prnu_mle":
                    weights = intensity
                    while weights.ndim < fingerprint.ndim:
                        weights = weights[..., np.newaxis]
                    predicted = fingerprint * weights
                signed_scores[candidate] = _signed_ncc(residual, predicted)
            energy_scores = {key: value * value for key, value in signed_scores.items()}
            for candidate in candidates:
                controls = np.asarray(
                    [energy_scores[other] for other in candidates if other != candidate],
                    dtype=np.float64,
                )
                control_mean = float(np.mean(controls))
                control_median = float(np.median(controls))
                control_mad = float(np.median(np.abs(controls - control_median)))
                candidate_energy = energy_scores[candidate]
                rows.append(
                    {
                        "query_id": residual_row["sample_id"],
                        "query_source_id": residual_row["source_id"],
                        "candidate_source_id": candidate,
                        "control_type": "other_generator_templates_same_dataset_native_geometry",
                        "pipeline_query": "native_release_bytes",
                        "pipeline_reference": "native_release_bytes",
                        "resolution_query": f"{source_record['width']}x{source_record['height']}",
                        "resolution_reference": (
                            f"{source_record['width']}x{source_record['height']}"
                        ),
                        "dataset_id": residual_row["dataset_id"],
                        "suite_id": suite_id,
                        "split": source_record["split"],
                        "prompt_id": source_record["prompt_id"],
                        "seed_id": "",
                        "extractor": extractor,
                        "aggregator": aggregator,
                        "similarity": "signed_zero_mean_ncc",
                        "raw_score": signed_scores[candidate],
                        "raw_energy": candidate_energy,
                        "control_mean": control_mean,
                        "control_median": control_median,
                        "control_mad": control_mad,
                        "fits_score": fits(
                            candidate_energy,
                            control_mean,
                            stabilizer=stabilizer,
                        ),
                        "fits_plus_score": fits_plus(
                            candidate_energy,
                            controls,
                            stabilizer=stabilizer,
                        ),
                        "log_ratio": log_ratio(
                            candidate_energy,
                            control_median,
                            stabilizer=stabilizer,
                        ),
                        "label_match": candidate == residual_row["source_id"],
                    }
                )
        if query_index % 40 == 0 or query_index == len(query_rows):
            print(f"scored_residual_queries={query_index}/{len(query_rows)}", flush=True)
    _write_csv(scores_path.resolve(), rows)
    return {
        "ok": True,
        "scores": str(scores_path.resolve()),
        "score_count": len(rows),
        "scores_sha256": sha256_file(scores_path.resolve()),
    }


def _query_blocks(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    return [grouped[key] for key in sorted(grouped)]


def _paired_margins(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    margins: list[float] = []
    for block in _query_blocks(rows):
        matched = [float(row["raw_score"]) for row in block if row["label_match"]]
        nonmatched = [float(row["raw_score"]) for row in block if not row["label_match"]]
        if len(matched) != 1 or not nonmatched:
            raise ValueError("each query must have exactly one match and at least one non-match")
        margins.append(matched[0] - float(np.mean(nonmatched)))
    return np.asarray(margins, dtype=np.float64)


def _permutation_pvalue(margins: np.ndarray, *, draws: int, seed: int) -> float:
    observed = float(np.mean(margins))
    generator = np.random.default_rng(seed)
    signs = generator.choice(np.asarray([-1.0, 1.0]), size=(draws, margins.size))
    permuted = np.mean(signs * margins[np.newaxis, :], axis=1)
    return float((1 + np.count_nonzero(permuted >= observed)) / (draws + 1))


def _cluster_bootstrap_auroc(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    blocks = _query_blocks(rows)
    generator = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = generator.integers(0, len(blocks), size=len(blocks))
        sampled = [row for index in selected for row in blocks[int(index)]]
        labels = np.asarray([bool(row["label_match"]) for row in sampled])
        values = np.asarray([float(row["raw_score"]) for row in sampled])
        estimates[draw] = roc_auc_score(labels, values)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def _holm_adjust(pvalues: Sequence[float]) -> list[float]:
    count = len(pvalues)
    order = np.argsort(np.asarray(pvalues))
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(pvalues[int(index)]))
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted.tolist()


def _read_scores(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["label_match"] = row["label_match"] == "True"
    return rows


def _completion_gate(
    config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    verification: Mapping[str, Any],
    residual_manifest: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    registered = config["completion_gate"]
    expected_extractors = set(config["extractors"])
    expected_aggregators = set(config["aggregators"]["names"])
    observed_pairs = {(row["extractor"], row["aggregator"]) for row in evaluations}
    leakage = source_manifest["leakage_audit"]
    checks = {
        "exact_selected_file_count": source_manifest["record_count"]
        == int(registered["expected_selected_file_count"]),
        "all_selected_sha256_verified": bool(verification["passed"]),
        "exact_native_geometry": all(
            int(row["width"]) == int(row["height"]) and int(row["width"]) in {256, 512}
            for row in source_manifest["records"]
        ),
        "no_gfits_image_transformations": all(
            not row["transformations_by_gfits"] for row in source_manifest["records"]
        ),
        "prompt_disjoint_splits": bool(leakage["prompt_disjoint_splits"]),
        "original_disjoint_splits": bool(leakage["original_disjoint_splits"]),
        "seed_metadata_and_disjoint_splits": bool(leakage["seed_metadata_complete"])
        and bool(leakage["seed_disjoint_splits"]),
        "all_extractors_present": set(residual_manifest["extractors_present"])
        == expected_extractors,
        "all_aggregators_present": all(
            (extractor, aggregator) in observed_pairs
            for extractor in expected_extractors
            for aggregator in expected_aggregators
        ),
        "exact_condition_count": len(evaluations) == int(registered["expected_condition_count"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _write_e02_figures(
    output_dir: Path,
    evaluations: Sequence[Mapping[str, Any]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suites = sorted({str(row["suite_id"]) for row in evaluations})
    extractors = ["wavelet", "srm", "low_bit", "noiseprint"]
    aggregators = ["mean", "prnu_mle", "median", "trimmed_mean"]
    figure, axes = plt.subplots(1, len(suites), figsize=(15, 4.8), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    image = None
    for axis, suite in zip(axes_array, suites, strict=True):
        matrix = np.full((len(extractors), len(aggregators)), np.nan)
        for row in evaluations:
            if row["suite_id"] == suite:
                matrix[
                    extractors.index(str(row["extractor"])),
                    aggregators.index(str(row["aggregator"])),
                ] = float(row["auroc"])
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(range(len(aggregators)), aggregators, rotation=35, ha="right")
        axis.set_yticks(range(len(extractors)), extractors)
        axis.set_title(suite)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    assert image is not None
    figure.colorbar(image, ax=axes_array.tolist(), label="test AUROC")
    figure.suptitle("E02 same-source generator signal (native geometry only)")
    matrix_path = output_dir / "signal-auroc-matrix.png"
    figure.savefig(matrix_path, dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    positions = np.arange(len(evaluations))
    values = np.asarray([float(row["paired_margin_mean"]) for row in evaluations])
    detected = np.asarray([bool(row["signal_detected"]) for row in evaluations])
    axis.bar(positions, values, color=np.where(detected, "#2c7fb8", "#bdbdbd"))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        xlabel="registered condition (suite / extractor / aggregator)",
        ylabel="mean paired signed-NCC margin",
        title="E02 paired match-minus-nonmatch margins",
    )
    axis.set_xticks([])
    margin_path = output_dir / "paired-margin-summary.png"
    figure.savefig(margin_path, dpi=170)
    plt.close(figure)
    return [matrix_path, margin_path]


def evaluate_e02(
    config_path: Path,
    source_manifest_path: Path,
    data_root: Path,
    residual_manifest_path: Path,
    scores_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the frozen paired signal tests and write the E02 stop/go decision."""

    config = load_e02_config(config_path.resolve())
    source_manifest = load_e02_manifest(source_manifest_path.resolve(), config)
    residual_manifest = _load_residual_manifest(residual_manifest_path.resolve(), config)
    verification = verify_e02_manifest(source_manifest, data_root.resolve())
    rows = _read_scores(scores_path.resolve())
    primary = [row for row in rows if row["split"] == config["evaluation"]["primary_split"]]
    conditions = sorted({(row["suite_id"], row["extractor"], row["aggregator"]) for row in primary})
    permutation = config["evaluation"]["permutation"]
    bootstrap = config["evaluation"]["bootstrap"]
    evaluations: list[dict[str, Any]] = []
    for condition_index, (suite_id, extractor, aggregator) in enumerate(conditions):
        subset = [
            row
            for row in primary
            if row["suite_id"] == suite_id
            and row["extractor"] == extractor
            and row["aggregator"] == aggregator
        ]
        labels = np.asarray([bool(row["label_match"]) for row in subset])
        values = np.asarray([float(row["raw_score"]) for row in subset])
        margins = _paired_margins(subset)
        ci_low, ci_high = _cluster_bootstrap_auroc(
            subset,
            draws=int(bootstrap["draws"]),
            confidence=float(bootstrap["confidence"]),
            seed=int(bootstrap["seed"]) + condition_index,
        )
        evaluations.append(
            {
                "suite_id": suite_id,
                "extractor": extractor,
                "aggregator": aggregator,
                "query_count": margins.size,
                "h1_count": int(np.count_nonzero(labels)),
                "h0_count": int(np.count_nonzero(~labels)),
                "h1_median": float(np.median(values[labels])),
                "h0_median": float(np.median(values[~labels])),
                "paired_margin_mean": float(np.mean(margins)),
                "paired_margin_median": float(np.median(margins)),
                "auroc": float(roc_auc_score(labels, values)),
                "average_precision": float(average_precision_score(labels, values)),
                "auroc_ci_low": ci_low,
                "auroc_ci_high": ci_high,
                "permutation_p": _permutation_pvalue(
                    margins,
                    draws=int(permutation["draws"]),
                    seed=int(permutation["seed"]) + condition_index,
                ),
            }
        )
    corrected = _holm_adjust([float(row["permutation_p"]) for row in evaluations])
    alpha = float(config["evaluation"]["alpha"])
    for row, corrected_p in zip(evaluations, corrected, strict=True):
        row["holm_corrected_p"] = corrected_p
        row["signal_detected"] = (
            corrected_p < alpha
            and float(row["auroc_ci_low"]) > 0.5
            and float(row["paired_margin_mean"]) > 0.0
        )
    completion = _completion_gate(
        config,
        source_manifest,
        verification,
        residual_manifest,
        evaluations,
    )
    passing_by_suite = {
        suite: sum(bool(row["signal_detected"]) for row in evaluations if row["suite_id"] == suite)
        for suite in config["mainline_gate"]["required_signal_suites"]
    }
    statistical_signal = all(
        count >= int(config["mainline_gate"]["minimum_passing_conditions_per_suite"])
        for count in passing_by_suite.values()
    )
    seed_metadata = bool(source_manifest["leakage_audit"]["seed_metadata_complete"])
    if not seed_metadata:
        decision = config["mainline_gate"]["decision_if_metadata_gate_fails"]
    elif not statistical_signal:
        decision = config["mainline_gate"]["decision_if_all_methods_fail"]
    else:
        decision = "proceed_to_e03_configuration_audit"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = output_dir / "condition-evaluation.csv"
    _write_csv(evaluation_path, evaluations)
    figures = _write_e02_figures(output_dir, evaluations)
    summary = {
        "schema": E02_RESULT_SCHEMA,
        "phase": "E02",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "claim_boundary": (
            "Exploratory same-source signal only: both public releases omit "
            "per-image seed identity."
        ),
        "configuration_sha256": sha256_file(config_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path.resolve()),
        "residual_manifest_sha256": sha256_file(residual_manifest_path.resolve()),
        "scores_sha256": sha256_file(scores_path.resolve()),
        "completion_gate": completion,
        "statistical_signal_gate": {
            "passed": statistical_signal,
            "passing_conditions_by_suite": passing_by_suite,
        },
        "mainline_decision": decision,
        "condition_count": len(evaluations),
        "artifacts": {
            "condition-evaluation.csv": sha256_file(evaluation_path),
            **{path.name: sha256_file(path) for path in figures},
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "ok": completion["passed"],
        "completed_execution": all(
            value
            for key, value in completion["checks"].items()
            if key != "seed_metadata_and_disjoint_splits"
        ),
        "summary": str(summary_path),
        "condition_evaluation": str(evaluation_path),
        "completion_gate": completion,
        "statistical_signal_gate": summary["statistical_signal_gate"],
        "mainline_decision": decision,
    }


def generate_e02_report(
    config_path: Path,
    source_manifest_path: Path,
    summary_path: Path,
    evaluation_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Generate a concise stage report from immutable machine-readable evidence."""

    config = load_e02_config(config_path.resolve())
    source_manifest = load_e02_manifest(source_manifest_path.resolve(), config)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with evaluation_path.open(newline="", encoding="utf-8") as stream:
        evaluations = list(csv.DictReader(stream))
    detected = [row for row in evaluations if row["signal_detected"] == "True"]
    best = sorted(evaluations, key=lambda row: float(row["auroc"]), reverse=True)[:8]
    suite_summaries = []
    for suite_id in sorted({row["suite_id"] for row in evaluations}):
        suite_rows = [row for row in evaluations if row["suite_id"] == suite_id]
        suite_summaries.append(
            {
                "suite_id": suite_id,
                "passing": sum(row["signal_detected"] == "True" for row in suite_rows),
                "total": len(suite_rows),
                "best": max(suite_rows, key=lambda row: float(row["auroc"])),
            }
        )
    zero_signal_suites = [row for row in suite_summaries if row["passing"] == 0]
    checks = summary["completion_gate"]["checks"]
    non_seed_checks = sum(
        bool(value) for key, value in checks.items() if key != "seed_metadata_and_disjoint_splits"
    )
    completion_label = "PASS" if summary["completion_gate"]["passed"] else "FAIL"
    signal_label = "PASS" if summary["statistical_signal_gate"]["passed"] else "FAIL"
    lines = [
        "# E02 Generator-Fingerprint Signal Validation",
        "",
        "## Outcome",
        "",
        (
            "- Execution-complete checks (excluding unavailable seed metadata): "
            f"**{non_seed_checks}/{len(checks) - 1}**."
        ),
        f"- Full confirmatory completion gate: **{completion_label}**.",
        f"- Statistical signal gate: **{signal_label}**.",
        f"- Mainline decision: `{summary['mainline_decision']}`.",
        "",
        (
            "The experiment is deliberately labeled exploratory. GenImage and the "
            "DMimageDetection public test release do not publish a per-image random-seed "
            "mapping, so the registered seed-isolation requirement cannot be verified. "
            "No result below is promoted to confirmatory generator-fingerprint evidence."
        ),
        "",
        "## Registered design",
        "",
        (
            f"The immutable manifest contains {source_manifest['record_count']} original PNG "
            "members: 12 template, 12 calibration, and 20 test images per source. "
            "DMimageDetection contributes four native 256×256 text-to-image sources. "
            "GenImage contributes independent 256×256 and 512×512 geometry suites with "
            "three sources each. Prompt/class groups and original archive members are "
            "disjoint across splits. G-FITS performs no resize, crop, recompression, EXIF "
            "transpose, or stored color conversion."
        ),
        "",
        (
            "The complete 4×4 method matrix compares wavelet, fixed SRM, RGB two-low-bit, "
            "and official Noiseprint residuals with arithmetic mean, PRNU MLE, pixel "
            "median, and 20% trimmed mean fingerprints. The primary score is signed "
            "zero-mean NCC. Each condition uses a one-sided paired query-level sign-flip "
            "test, query-clustered AUROC bootstrap, and Holm FWER correction across all "
            "48 conditions."
        ),
        "",
        "## Signal results",
        "",
        (
            f"{len(detected)} of {len(evaluations)} registered conditions met all three "
            "exploratory signal criteria (Holm-adjusted p < 0.05, AUROC lower 95% bound "
            "> 0.5, and positive paired margin)."
        ),
        "",
        "| Suite | Passing | Best method | Best AUROC | 95% CI | Holm p | Signal |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for suite in suite_summaries:
        row = suite["best"]
        lines.append(
            "| {suite_id} | {passing}/{total} | {extractor}/{aggregator} | {auroc:.4f} | "
            "[{low:.4f}, {high:.4f}] | {p:.4g} | {signal} |".format(
                suite_id=suite["suite_id"],
                passing=suite["passing"],
                total=suite["total"],
                extractor=row["extractor"],
                aggregator=row["aggregator"],
                auroc=float(row["auroc"]),
                low=float(row["auroc_ci_low"]),
                high=float(row["auroc_ci_high"]),
                p=float(row["holm_corrected_p"]),
                signal=row["signal_detected"],
            )
        )
    lines.extend(
        [
            "",
            "The eight highest-AUROC registered conditions are:",
            "",
            "| Suite | Extractor | Aggregator | AUROC | 95% CI | Holm p | Margin | Signal |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in best:
        lines.append(
            "| {suite_id} | {extractor} | {aggregator} | {auroc:.4f} | "
            "[{low:.4f}, {high:.4f}] | {p:.4g} | {margin:.4g} | {signal} |".format(
                suite_id=row["suite_id"],
                extractor=row["extractor"],
                aggregator=row["aggregator"],
                auroc=float(row["auroc"]),
                low=float(row["auroc_ci_low"]),
                high=float(row["auroc_ci_high"]),
                p=float(row["holm_corrected_p"]),
                margin=float(row["paired_margin_mean"]),
                signal=row["signal_detected"],
            )
        )
    lines.extend(
        [
            "",
            "## Evidence supporting the working hypothesis",
            "",
            (
                "- Positive conditions, when present, show that a fixed residual/aggregator "
                "pair ranks the matching generator above native-geometry nonmatches on "
                "held-out prompt/class groups."
            ),
            (
                "- The DM and GenImage suites are evaluated independently, preventing a "
                "single random dataset split from being mislabeled as cross-generator "
                "generalization."
            ),
            "",
            "## Evidence against or limiting the working hypothesis",
            "",
            (
                "- Per-image seed identity is unavailable in both public releases. The full "
                "completion gate therefore fails by construction, even if exploratory "
                "signal is strong."
            ),
            *(
                [
                    (
                        "- No registered condition passed in "
                        + ", ".join(str(row["suite_id"]) for row in zero_signal_suites)
                        + ". The best observed AUROC in those suites was "
                        + "; ".join(
                            "{suite}: {auroc:.4f} ({extractor}/{aggregator}, lower 95% "
                            "bound {low:.4f})".format(
                                suite=row["suite_id"],
                                auroc=float(row["best"]["auroc"]),
                                extractor=row["best"]["extractor"],
                                aggregator=row["best"]["aggregator"],
                                low=float(row["best"]["auroc_ci_low"]),
                            )
                            for row in zero_signal_suites
                        )
                        + ". The exploratory signal is therefore not native-geometry "
                        "general across the registered suites."
                    )
                ]
                if zero_signal_suites
                else []
            ),
            (
                "- GenImage was range-extracted from a revision-pinned unofficial byte "
                "mirror because the official release is distributed as very large split "
                "archives. Every selected member is independently SHA-256 hashed, but the "
                "mirror boundary remains explicit."
            ),
            (
                "- This phase applies no downstream software pipeline. It validates only "
                "whether a source-specific signal may exist; it does not test G-FITS "
                "calibration and is not a general AIGC detector result."
            ),
            (
                "- Any RGBA GenImage member is accepted only when its alpha plane is "
                "entirely 255; the unchanged RGB channels are analyzed and the original "
                "RGBA bytes remain the hashed evidence."
            ),
            "",
            "## Primary sources",
            "",
            (
                "- Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting "
                "AI-Generated Image* (NeurIPS 2023 Datasets and Benchmarks), plus the "
                "official GenImage repository and Dataset Terms."
            ),
            (
                "- Corvi et al., *On the Detection of Synthetic Images Generated by "
                "Diffusion Models* (ICASSP 2023), DOI "
                "10.1109/ICASSP49357.2023.10095167, and the official release repository."
            ),
            (
                "- Cozzolino and Verdoliva, *Noiseprint: A CNN-Based Camera Model "
                "Fingerprint* (IEEE TIFS 2020), DOI 10.1109/TIFS.2019.2916364, and the "
                "locked official implementation."
            ),
            (
                "- Bondi, Bestagini, and Bonettini, locked `polimi-ispl/prnu-python` "
                "implementation used for the wavelet residual baseline."
            ),
            "",
            "## Reproducibility and claim boundary",
            "",
            (
                "All selected input bytes, residual caches, fingerprints, scores, "
                "configuration, source revisions, and output tables are hash-addressed. "
                "Calibration rows are retained only for diagnostics; no extractor, "
                "aggregator, threshold, stabilizer, or test criterion was chosen from test "
                "outcomes."
            ),
            "",
            (
                "A controlled dataset that publishes prompt, seed, generator checkpoint, "
                "sampler, decoder/VAE, and original/derivative identity is required before "
                "the confirmatory E02 gate can pass. Until then, the registered mainline "
                "decision above governs subsequent work."
            ),
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "ok": True,
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path.resolve()),
    }
