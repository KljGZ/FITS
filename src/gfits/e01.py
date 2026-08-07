"""Real downstream-pipeline mechanism replication for Phase E01."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml
from PIL import Image
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

from gfits.manifest import sha256_file
from gfits.matching import fits, fits_plus, log_ratio, normalized_cross_correlation

E01_MANIFEST_SCHEMA = "gfits.e01-vision-manifest/v1"
E01_RESULT_SCHEMA = "gfits.e01-mechanism-validation/v1"


@dataclass(frozen=True)
class VisionRecord:
    """One immutable VISION source file selected by the E01 protocol."""

    device: str
    pipeline: str
    split: str
    source_index: int
    source_url: str
    relative_path: str
    expected_width: int
    expected_height: int


def _required(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise ValueError(f"missing configuration key: {name}")
    return mapping[name]


def load_e01_config(path: Path) -> dict[str, Any]:
    """Load and validate the immutable Phase E01 protocol."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("phase") != "E01":
        raise ValueError("configuration must be a Phase E01 mapping")
    if payload.get("status") != "pre_registered":
        raise ValueError("E01 configuration must be pre_registered")
    dataset = _required(payload, "dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a mapping")
    devices = dataset.get("devices")
    pipelines = dataset.get("pipelines")
    splits = dataset.get("splits")
    if not isinstance(devices, list) or len(devices) < 4 or len(set(devices)) != len(devices):
        raise ValueError("dataset.devices must contain at least four unique devices")
    if not isinstance(pipelines, dict) or len(pipelines) < 2:
        raise ValueError("dataset.pipelines must contain at least two pipelines")
    if not isinstance(splits, dict) or set(splits) != {"template", "calibration", "test"}:
        raise ValueError("dataset.splits must define template, calibration, and test")
    split_indices: dict[str, set[int]] = {}
    for split, bounds in splits.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"split {split} must be an inclusive [start, stop] pair")
        start, stop = (int(value) for value in bounds)
        if start <= 0 or stop < start:
            raise ValueError(f"invalid split bounds for {split}")
        split_indices[split] = set(range(start, stop + 1))
    for left, left_values in split_indices.items():
        for right, right_values in split_indices.items():
            if left < right and left_values & right_values:
                raise ValueError(f"source-index leakage between {left} and {right}")
    expected = len(devices) * len(pipelines) * sum(len(values) for values in split_indices.values())
    if int(dataset.get("expected_file_count", -1)) != expected:
        raise ValueError("dataset.expected_file_count does not match the registered selection")
    extractor = _required(payload, "extractor")
    if extractor.get("resize") or extractor.get("crop") or extractor.get("recompress"):
        raise ValueError("E01 forbids resize, crop, and recompression")
    return payload


def _vision_records(config: Mapping[str, Any]) -> list[VisionRecord]:
    dataset = config["dataset"]
    base_url = str(dataset["base_url"]).rstrip("/")
    records: list[VisionRecord] = []
    for device in dataset["devices"]:
        device_id = str(device).split("_", 1)[0]
        for pipeline, pipeline_config in dataset["pipelines"].items():
            directory = pipeline_config["directory"]
            token = pipeline_config["filename_token"]
            width, height = (int(value) for value in pipeline_config["resolution"])
            for split, bounds in dataset["splits"].items():
                for source_index in range(int(bounds[0]), int(bounds[1]) + 1):
                    filename = f"{device_id}_I_{token}_{source_index:04d}.jpg"
                    relative = f"{device}/{pipeline}/{filename}"
                    url = f"{base_url}/{device}/images/{directory}/{filename}"
                    records.append(
                        VisionRecord(
                            device=str(device),
                            pipeline=str(pipeline),
                            split=str(split),
                            source_index=source_index,
                            source_url=url,
                            relative_path=relative,
                            expected_width=width,
                            expected_height=height,
                        )
                    )
    return records


def _inspect_rgb(path: Path, record: VisionRecord) -> tuple[int, int, str]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
    if (width, height) != (record.expected_width, record.expected_height):
        raise ValueError(
            f"unexpected dimensions for {record.relative_path}: "
            f"{width}x{height}, expected {record.expected_width}x{record.expected_height}"
        )
    if mode != "RGB":
        raise ValueError(f"unexpected image mode for {record.relative_path}: {mode}")
    return width, height, mode


def _download_one(
    record: VisionRecord,
    data_root: Path,
    *,
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    target = data_root / Path(record.relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
        request = urllib.request.Request(
            record.source_url,
            headers={"User-Agent": "G-FITS-E01-research/1.0"},
        )
        last_error: BaseException | None = None
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
                temporary.replace(target)
                last_error = None
                break
            except (OSError, urllib.error.URLError) as error:
                last_error = error
                temporary.unlink(missing_ok=True)
                if attempt < retries:
                    time.sleep(min(2**attempt, 10))
        if last_error is not None:
            raise RuntimeError(f"failed to download {record.source_url}: {last_error}")
    width, height, mode = _inspect_rgb(target, record)
    return {
        "sample_id": f"{record.device}:{record.pipeline}:{record.source_index:04d}",
        "device": record.device,
        "pipeline": record.pipeline,
        "split": record.split,
        "source_index": record.source_index,
        "original_group_id": f"{record.device}:{record.source_index:04d}",
        "source_url": record.source_url,
        "relative_path": record.relative_path,
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "width": width,
        "height": height,
        "mode": mode,
        "transformations_by_gfits": [],
    }


def download_vision_e01(
    config_path: Path,
    data_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Download the registered VISION subset byte-for-byte and hash every input."""

    config_path = config_path.resolve()
    data_root = data_root.resolve()
    manifest_path = manifest_path.resolve()
    config = load_e01_config(config_path)
    records = _vision_records(config)
    dataset = config["dataset"]
    workers = int(dataset["download_workers"])
    retries = int(dataset["download_retries"])
    timeout = int(dataset["request_timeout_seconds"])
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                record,
                data_root,
                retries=retries,
                timeout=timeout,
            ): record
            for record in records
        }
        for future in as_completed(futures):
            completed.append(future.result())
            if len(completed) % 20 == 0 or len(completed) == len(records):
                print(f"downloaded_or_verified={len(completed)}/{len(records)}", flush=True)
    completed.sort(key=lambda item: item["sample_id"])
    payload: dict[str, Any] = {
        "schema": E01_MANIFEST_SCHEMA,
        "phase": "E01",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": dataset["name"],
        "license": dataset["license"],
        "license_url": dataset["license_url"],
        "citation_doi": dataset["citation_doi"],
        "source_root": str(data_root),
        "configuration_sha256": sha256_file(config_path),
        "record_count": len(completed),
        "records": completed,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": len(completed) == int(dataset["expected_file_count"]),
        "manifest": str(manifest_path),
        "record_count": len(completed),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _load_e01_manifest(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != E01_MANIFEST_SCHEMA:
        raise ValueError("unsupported E01 manifest schema")
    if payload.get("record_count") != len(payload.get("records", [])):
        raise ValueError("E01 manifest record count is inconsistent")
    if payload.get("configuration_sha256") != sha256_file(Path(config["_config_path"])):
        raise ValueError("E01 manifest was built from a different configuration")
    return payload


def _verify_e01_manifest(payload: Mapping[str, Any], data_root: Path) -> dict[str, Any]:
    issues: list[str] = []
    for record in payload["records"]:
        path = data_root / Path(record["relative_path"])
        if not path.is_file():
            issues.append(f"missing:{record['relative_path']}")
            continue
        if path.stat().st_size != int(record["size_bytes"]):
            issues.append(f"size:{record['relative_path']}")
            continue
        if sha256_file(path) != record["sha256"]:
            issues.append(f"sha256:{record['relative_path']}")
    return {"passed": not issues, "checked": len(payload["records"]), "issues": issues}


def _load_upstream(root: Path, extractor: Mapping[str, Any]) -> ModuleType:
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
    source = root / "prnu" / "functions.py"
    canonical_source = subprocess.run(
        ["git", "show", "HEAD:prnu/functions.py"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(canonical_source).hexdigest() != extractor["functions_sha256"]:
        raise ValueError("locked prnu-python functions.py SHA-256 mismatch")
    specification = importlib.util.spec_from_file_location("gfits_e01_prnu", source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load upstream module: {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _decode_rgb(path: Path, expected_width: int, expected_height: int) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"image is not native RGB: {path}")
        if image.size != (expected_width, expected_height):
            raise ValueError(f"image dimensions changed after manifest creation: {path}")
        array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"unsupported decoded representation: {path}")
    return array


def _fingerprint(
    paths: Sequence[Path],
    width: int,
    height: int,
    upstream: ModuleType,
    *,
    levels: int,
    sigma: float,
) -> np.ndarray:
    numerator = np.zeros((height, width, 3), dtype=np.float64)
    denominator = np.zeros((height, width, 3), dtype=np.float64)
    for path in paths:
        image = _decode_rgb(path, width, height)
        residual = upstream.noise_extract(image, levels, sigma)
        numerator += residual * image / 255.0
        denominator += np.square(upstream.inten_scale(image) * upstream.saturation(image))
    fingerprint = numerator / (denominator + 1.0)
    fingerprint = upstream.rgb2gray(fingerprint)
    fingerprint = upstream.zero_mean_total(fingerprint)
    return upstream.wiener_dft(fingerprint, fingerprint.std(ddof=1)).astype(np.float32)


def _query_residual(
    path: Path,
    width: int,
    height: int,
    upstream: ModuleType,
    *,
    levels: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    image = _decode_rgb(path, width, height)
    residual = upstream.extract_single(image, levels=levels, sigma=sigma)
    intensity = upstream.rgb2gray(image.astype(np.float32)) / 255.0
    return residual.astype(np.float32), intensity.astype(np.float32)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _score_rows(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    data_root: Path,
    upstream: ModuleType,
    cache_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = config["dataset"]
    extractor = config["extractor"]
    scoring = config["scoring"]
    devices = [str(value) for value in dataset["devices"]]
    by_key = {
        (record["device"], record["pipeline"], record["split"], int(record["source_index"])): record
        for record in manifest["records"]
    }
    score_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    levels = int(extractor["levels"])
    sigma = float(extractor["sigma"])
    stabilizer = float(scoring["stabilizer"])
    template_count = (
        int(dataset["splits"]["template"][1]) - int(dataset["splits"]["template"][0]) + 1
    )
    for pipeline, pipeline_config in dataset["pipelines"].items():
        width, height = (int(value) for value in pipeline_config["resolution"])
        fingerprints: dict[str, np.ndarray] = {}
        template_inputs: dict[str, tuple[list[Mapping[str, Any]], list[Path]]] = {}
        for device in devices:
            start, stop = dataset["splits"]["template"]
            template_records = [
                by_key[(device, pipeline, "template", index)]
                for index in range(int(start), int(stop) + 1)
            ]
            paths = [data_root / Path(record["relative_path"]) for record in template_records]
            template_inputs[device] = (template_records, paths)
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            template_futures = {
                device: executor.submit(
                    _fingerprint,
                    template_inputs[device][1],
                    width,
                    height,
                    upstream,
                    levels=levels,
                    sigma=sigma,
                )
                for device in devices
            }
            for device in devices:
                template_records, paths = template_inputs[device]
                fingerprint = template_futures[device].result()
                cache_path = cache_root / "fingerprints" / pipeline / f"{device}.npy"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, fingerprint, allow_pickle=False)
                fingerprints[device] = fingerprint
                fingerprint_rows.append(
                    {
                        "device": device,
                        "pipeline": pipeline,
                        "width": width,
                        "height": height,
                        "template_count": len(paths),
                        "template_sample_ids": "|".join(
                            record["sample_id"] for record in template_records
                        ),
                        "fingerprint_path": str(cache_path),
                        "fingerprint_sha256": sha256_file(cache_path),
                    }
                )
        for split in ("calibration", "test"):
            start, stop = dataset["splits"][split]
            query_inputs: list[tuple[str, int, Mapping[str, Any], Path]] = []
            for query_device in devices:
                for source_index in range(int(start), int(stop) + 1):
                    record = by_key[(query_device, pipeline, split, source_index)]
                    query_path = data_root / Path(record["relative_path"])
                    query_inputs.append((query_device, source_index, record, query_path))
            with ThreadPoolExecutor(max_workers=8) as executor:
                query_futures = [
                    executor.submit(
                        _query_residual,
                        query_path,
                        width,
                        height,
                        upstream,
                        levels=levels,
                        sigma=sigma,
                    )
                    for _, _, _, query_path in query_inputs
                ]
                for query_input, query_future in zip(
                    query_inputs,
                    query_futures,
                    strict=True,
                ):
                    query_device, source_index, record, _ = query_input
                    residual, intensity = query_future.result()
                    signed_ncc: dict[str, float] = {}
                    raw_scores: dict[str, float] = {}
                    for candidate in devices:
                        predicted = fingerprints[candidate] * intensity
                        ncc = normalized_cross_correlation(residual, predicted)
                        signed_ncc[candidate] = ncc
                        raw_scores[candidate] = ncc * ncc
                    for candidate in devices:
                        eligible = sorted(
                            device
                            for device in devices
                            if device != candidate and device != query_device
                        )
                        if not eligible:
                            raise RuntimeError("no eligible nuisance controls")
                        controls = np.asarray([raw_scores[device] for device in eligible])
                        single = float(controls[0])
                        median = float(np.median(controls))
                        candidate_score = raw_scores[candidate]
                        score_rows.append(
                            {
                                "query_id": record["sample_id"],
                                "query_source_id": query_device,
                                "candidate_source_id": candidate,
                                "control_source_ids": "|".join(eligible),
                                "control_type": "same_pipeline_different_device",
                                "pipeline_query": pipeline,
                                "pipeline_reference": pipeline,
                                "resolution_query": f"{width}x{height}",
                                "resolution_reference": f"{width}x{height}",
                                "extractor": "prnu_python_wavelet_mle",
                                "aggregator": f"mle_{template_count}",
                                "similarity": "squared_aligned_ncc",
                                "split": split,
                                "source_index": source_index,
                                "signed_ncc": signed_ncc[candidate],
                                "raw_score": candidate_score,
                                "control_mean": float(np.mean(controls)),
                                "control_median": median,
                                "control_mad": float(np.median(np.abs(controls - median))),
                                "single_control_score": single,
                                "subtraction_score": candidate_score - single,
                                "fits_score": fits(
                                    candidate_score,
                                    single,
                                    stabilizer=stabilizer,
                                ),
                                "fits_plus_score": fits_plus(
                                    candidate_score,
                                    controls,
                                    stabilizer=stabilizer,
                                ),
                                "log_ratio": log_ratio(
                                    candidate_score,
                                    median,
                                    stabilizer=stabilizer,
                                ),
                                "label_match": candidate == query_device,
                            }
                        )
            print(f"scored_pipeline={pipeline} split={split}", flush=True)
    return score_rows, fingerprint_rows


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pipelines = list(config["dataset"]["pipelines"])
    methods = [str(value) for value in config["evaluation"]["methods"]]
    target_far = float(config["evaluation"]["target_far"])
    method_column = {
        "raw_score": "raw_score",
        "subtraction_score": "subtraction_score",
        "fits_score": "fits_score",
        "fits_plus_score": "fits_plus_score",
    }
    evaluations: list[dict[str, Any]] = []
    thresholds_by_method: dict[str, list[float]] = {method: [] for method in methods}
    ks_by_method: dict[str, list[float]] = {method: [] for method in methods}
    for method in methods:
        column = method_column[method]
        h0_test_by_pipeline: dict[str, np.ndarray] = {}
        for pipeline in pipelines:
            calibration_h0 = np.asarray(
                [
                    float(row[column])
                    for row in rows
                    if row["pipeline_query"] == pipeline
                    and row["split"] == "calibration"
                    and not row["label_match"]
                ]
            )
            test_h0 = np.asarray(
                [
                    float(row[column])
                    for row in rows
                    if row["pipeline_query"] == pipeline
                    and row["split"] == "test"
                    and not row["label_match"]
                ]
            )
            test_h1 = np.asarray(
                [
                    float(row[column])
                    for row in rows
                    if row["pipeline_query"] == pipeline
                    and row["split"] == "test"
                    and row["label_match"]
                ]
            )
            threshold = float(np.quantile(calibration_h0, 1.0 - target_far, method="higher"))
            labels = np.concatenate([np.zeros(test_h0.size), np.ones(test_h1.size)])
            values = np.concatenate([test_h0, test_h1])
            evaluations.append(
                {
                    "method": method,
                    "pipeline": pipeline,
                    "target_far": target_far,
                    "threshold": threshold,
                    "calibration_h0_count": calibration_h0.size,
                    "test_h0_count": test_h0.size,
                    "test_h1_count": test_h1.size,
                    "observed_test_far": float(np.mean(test_h0 >= threshold)),
                    "observed_test_tpr": float(np.mean(test_h1 >= threshold)),
                    "test_h0_median": float(np.median(test_h0)),
                    "test_h1_median": float(np.median(test_h1)),
                    "auroc": float(roc_auc_score(labels, values)),
                    "average_precision": float(average_precision_score(labels, values)),
                }
            )
            thresholds_by_method[method].append(threshold)
            h0_test_by_pipeline[pipeline] = test_h0
        for left_index, left in enumerate(pipelines):
            for right in pipelines[left_index + 1 :]:
                ks_by_method[method].append(
                    float(ks_2samp(h0_test_by_pipeline[left], h0_test_by_pipeline[right]).statistic)
                )
    aggregate: dict[str, Any] = {"methods": {}}
    for method in methods:
        thresholds = np.asarray(thresholds_by_method[method])
        mean_threshold = float(np.mean(thresholds))
        threshold_cv = (
            float(np.std(thresholds, ddof=1) / abs(mean_threshold))
            if mean_threshold != 0.0
            else float("inf")
        )
        aggregate["methods"][method] = {
            "threshold_mean": mean_threshold,
            "threshold_cv": threshold_cv,
            "mean_pairwise_h0_ks": float(np.mean(ks_by_method[method])),
        }
    raw = aggregate["methods"]["raw_score"]
    calibrated = aggregate["methods"]["fits_plus_score"]
    aggregate["mechanism_hypothesis"] = {
        "fits_plus_threshold_cv_below_raw": (calibrated["threshold_cv"] < raw["threshold_cv"]),
        "fits_plus_mean_pairwise_h0_ks_below_raw": (
            calibrated["mean_pairwise_h0_ks"] < raw["mean_pairwise_h0_ks"]
        ),
    }
    aggregate["mechanism_hypothesis"]["passed"] = all(aggregate["mechanism_hypothesis"].values())
    return evaluations, aggregate


def _completion_gate(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    verification: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    registered = config["completion_gate"]
    records = manifest["records"]
    split_indices: dict[str, set[int]] = {name: set() for name in config["dataset"]["splits"]}
    for record in records:
        split_indices[record["split"]].add(int(record["source_index"]))
    disjoint = all(
        not (split_indices[left] & split_indices[right])
        for left in split_indices
        for right in split_indices
        if left < right
    )
    dimension_ok = all(
        [int(record["width"]), int(record["height"])]
        == [
            int(value) for value in config["dataset"]["pipelines"][record["pipeline"]]["resolution"]
        ]
        for record in records
    )
    minimum_counts = all(
        int(row["calibration_h0_count"]) >= int(registered["minimum_calibration_h0_per_pipeline"])
        and int(row["test_h0_count"]) >= int(registered["minimum_test_h0_per_pipeline"])
        and int(row["test_h1_count"]) >= int(registered["minimum_test_h1_per_pipeline"])
        for row in evaluations
    )
    checks = {
        "exact_file_count": len(records) == int(registered["exact_file_count"]),
        "all_sha256_verified": bool(verification["passed"]),
        "exact_registered_dimensions": dimension_ok,
        "disjoint_source_indices": disjoint,
        "minimum_evaluation_counts": minimum_counts,
        "no_gfits_transformations": all(
            not record["transformations_by_gfits"] for record in records
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _write_figures(
    output_directory: Path,
    rows: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [str(value) for value in config["evaluation"]["methods"]]
    pipelines = list(config["dataset"]["pipelines"])
    figure, axes = plt.subplots(
        len(methods),
        len(pipelines),
        figsize=(14, 10),
        constrained_layout=True,
    )
    for method_index, method in enumerate(methods):
        for pipeline_index, pipeline in enumerate(pipelines):
            axis = axes[method_index, pipeline_index]
            h0 = [
                float(row[method])
                for row in rows
                if row["pipeline_query"] == pipeline
                and row["split"] == "test"
                and not row["label_match"]
            ]
            h1 = [
                float(row[method])
                for row in rows
                if row["pipeline_query"] == pipeline
                and row["split"] == "test"
                and row["label_match"]
            ]
            threshold = next(
                float(row["threshold"])
                for row in evaluations
                if row["method"] == method and row["pipeline"] == pipeline
            )
            axis.hist(h0, bins=30, alpha=0.55, label="H0")
            axis.hist(h1, bins=30, alpha=0.55, label="H1")
            axis.axvline(threshold, color="black", linestyle="--", linewidth=1)
            axis.set_title(f"{method} / {pipeline}")
            if method_index == 0 and pipeline_index == 0:
                axis.legend()
    distribution_path = output_directory / "h0-h1-histograms.png"
    figure.suptitle("E01 VISION downstream-pipeline score distributions")
    figure.savefig(distribution_path, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, method in zip(axes.flat, methods, strict=True):
        for pipeline in pipelines:
            values = np.sort(
                [
                    float(row[method])
                    for row in rows
                    if row["pipeline_query"] == pipeline
                    and row["split"] == "test"
                    and not row["label_match"]
                ]
            )
            ecdf = np.arange(1, values.size + 1) / values.size
            axis.plot(values, ecdf, label=pipeline)
        axis.set(title=method, xlabel="H0 score", ylabel="ECDF")
        axis.legend(fontsize=7)
    ecdf_path = output_directory / "h0-ecdf.png"
    figure.suptitle("E01 cross-pipeline H0 alignment")
    figure.savefig(ecdf_path, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    threshold_cv = [aggregate["methods"][method]["threshold_cv"] for method in methods]
    pairwise_ks = [aggregate["methods"][method]["mean_pairwise_h0_ks"] for method in methods]
    axes[0].bar(methods, threshold_cv)
    axes[0].set(title="Threshold CV across pipelines", ylabel="CV")
    axes[1].bar(methods, pairwise_ks)
    axes[1].set(title="Mean pairwise H0 KS", ylabel="KS statistic")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    diagnostics_path = output_directory / "calibration-diagnostics.png"
    figure.savefig(diagnostics_path, dpi=160)
    plt.close(figure)
    return [distribution_path, ecdf_path, diagnostics_path]


def _git_metadata(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"base_commit": commit, "tracked_worktree_dirty": dirty}


def run_e01(
    config_path: Path,
    manifest_path: Path,
    data_root: Path,
    upstream_root: Path,
    output_directory: Path,
    cache_root: Path,
) -> dict[str, Any]:
    """Execute the complete pre-registered E01 real-pipeline experiment."""

    config_path = config_path.resolve()
    manifest_path = manifest_path.resolve()
    data_root = data_root.resolve()
    output_directory = output_directory.resolve()
    cache_root = cache_root.resolve()
    config = load_e01_config(config_path)
    config["_config_path"] = str(config_path)
    manifest = _load_e01_manifest(manifest_path, config)
    verification = _verify_e01_manifest(manifest, data_root)
    if not verification["passed"]:
        raise ValueError(f"E01 dataset verification failed: {verification['issues'][:3]}")
    upstream = _load_upstream(upstream_root, config["extractor"])
    output_directory.mkdir(parents=True, exist_ok=True)
    score_rows, fingerprint_rows = _score_rows(
        config,
        manifest,
        data_root,
        upstream,
        cache_root,
    )
    evaluations, aggregate = _evaluate(score_rows, config)
    completion = _completion_gate(config, manifest, verification, evaluations)
    score_path = output_directory / "scores.csv"
    fingerprint_path = output_directory / "fingerprints.csv"
    evaluation_path = output_directory / "threshold-evaluation.csv"
    _write_csv(score_path, score_rows)
    _write_csv(fingerprint_path, fingerprint_rows)
    _write_csv(evaluation_path, evaluations)
    figures = _write_figures(
        output_directory,
        score_rows,
        evaluations,
        aggregate,
        config,
    )
    repository_root = Path(__file__).resolve().parents[2]
    summary: dict[str, Any] = {
        "schema": E01_RESULT_SCHEMA,
        "phase": "E01",
        "claim": config["claim"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "passed": completion["passed"],
        "configuration_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": {
            "src/gfits/e01.py": sha256_file(repository_root / "src" / "gfits" / "e01.py"),
            "src/gfits/matching.py": sha256_file(repository_root / "src" / "gfits" / "matching.py"),
            "configs/e01.yaml": sha256_file(config_path),
        },
        "git": _git_metadata(repository_root),
        "dataset": {
            "name": config["dataset"]["name"],
            "record_count": len(manifest["records"]),
            "license": manifest["license"],
            "verification": verification,
            "transformations_by_gfits": [],
        },
        "extractor": config["extractor"],
        "scoring": config["scoring"],
        "completion_gate": completion,
        "aggregate": aggregate,
        "artifacts": {},
    }
    for path in [score_path, fingerprint_path, evaluation_path, *figures]:
        summary["artifacts"][path.name] = sha256_file(path)
    summary_path = output_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
