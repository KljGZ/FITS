"""Controlled factorial data and provenance utilities for Phase E02b."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from gfits.manifest import sha256_file

E02B_GENERATION_SCHEMA = "gfits.e02b-generation-manifest/v1"
E02B_DERIVATIVE_SCHEMA = "gfits.e02b-derivative-manifest/v1"

REQUIRED_SAMPLE_FIELDS = (
    "sample_id",
    "suite_ids",
    "model_id",
    "model_hash",
    "base_model",
    "checkpoint_hash",
    "vae",
    "vae_hash",
    "sampler",
    "scheduler",
    "steps",
    "guidance_scale",
    "prompt_id",
    "prompt",
    "negative_prompt",
    "seed",
    "split",
    "native_resolution",
    "generation_code_commit",
    "generation_tree_clean",
    "tensor_dtype",
    "tensor_sha256",
    "quantization_method",
    "writer_id",
    "writer_library",
    "writer_version",
    "relative_path",
    "output_sha256",
)


@dataclass(frozen=True)
class RepositoryState:
    """Git state that is embedded in every formal E02b sample."""

    commit: str
    clean: bool
    source_sha256: dict[str, str]


def _require(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise ValueError(f"missing E02b configuration key: {name}")
    return mapping[name]


def load_e02b_config(path: Path) -> dict[str, Any]:
    """Load the pre-registered E02b configuration and validate its full design."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("phase") != "E02b":
        raise ValueError("configuration must be a Phase E02b mapping")
    if payload.get("status") != "pre_registered":
        raise ValueError("E02b configuration must be pre_registered")
    design = _require(payload, "design")
    subjects = list(_require(design, "subjects"))
    scenes = list(_require(design, "scenes"))
    prompt_count = int(_require(design, "prompt_count"))
    if len(subjects) * len(scenes) != prompt_count:
        raise ValueError("subjects x scenes must equal design.prompt_count")
    split_counts = _require(design, "prompt_split_counts")
    if list(split_counts) != ["template", "calibration", "test"]:
        raise ValueError("prompt splits must be template, calibration, and test in order")
    if sum(int(value) for value in split_counts.values()) != prompt_count:
        raise ValueError("prompt split counts must consume every prompt")
    seed_splits = _require(design, "seeds")
    if list(seed_splits) != ["template", "calibration", "test"]:
        raise ValueError("seed splits must be template, calibration, and test in order")
    flattened_seeds = [int(seed) for values in seed_splits.values() for seed in values]
    if len(flattened_seeds) != len(set(flattened_seeds)) or len(flattened_seeds) < 4:
        raise ValueError("E02b requires at least four split-disjoint seeds")
    resolutions = [int(value) for value in _require(design, "resolutions")]
    if resolutions != [256, 512, 768, 1024]:
        raise ValueError("E02b resolutions must remain the registered 256/512/768/1024 grid")
    groups = _require(payload, "model_groups")
    if set(groups) != {"cross_family", "near_family"}:
        raise ValueError("E02b needs independent cross_family and near_family suites")
    models = _require(payload, "models")
    for group_id, group in groups.items():
        members = list(_require(group, "models"))
        if len(members) < 4 or len(set(members)) != len(members):
            raise ValueError(f"{group_id} must contain at least four unique models")
        if not set(members) <= set(models):
            raise ValueError(f"{group_id} references an unknown model")
    cross_families = {models[name]["family"] for name in groups["cross_family"]["models"]}
    if len(cross_families) < 3:
        raise ValueError("cross_family may not consist only of one generator family")
    template_sizes = list(payload["spatial_branch"]["template_sizes"])
    if template_sizes != [1, 3, 5, 10, 20, 50]:
        raise ValueError("the registered template-size curve is incomplete")
    signatures = set(payload["statistical_signature_branch"]["signatures"])
    expected_signatures = {
        "power_spectrum_2d",
        "phase_spectrum",
        "radial_power_spectrum",
        "autocorrelation",
        "cepstrum",
        "srm_style_cooccurrence",
        "patch_gram_covariance",
        "low_high_frequency_partition",
    }
    if signatures != expected_signatures:
        raise ValueError("the registered statistical-signature branch is incomplete")
    if set(payload["counterfactuals"]) != {
        "canonical_uniform_export",
        "random_lsb_pm1",
        "bit_depth_6",
        "png_reencode_level_9",
    }:
        raise ValueError("the registered exporter/low-bit counterfactual matrix is incomplete")
    resolution_profile = payload["statistics"]["resolution_profile"]
    if [int(value) for value in resolution_profile["resolutions"]] != resolutions:
        raise ValueError("the secondary resolution profile must cover the full native grid")
    if resolution_profile["gate_effect"] != "none_descriptive_secondary_analysis":
        raise ValueError("the resolution profile may not alter the E02b Gate")
    payload["_config_path"] = str(path.resolve())
    return payload


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository_state(root: Path, config: Mapping[str, Any]) -> RepositoryState:
    """Return the committed source state and fail if formal execution is dirty."""

    root = root.resolve()
    commit = _git(root, "rev-parse", "HEAD")
    unstaged_clean = subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode == 0
    staged_clean = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode == 0
    untracked = [
        line
        for line in _git(root, "status", "--porcelain").splitlines()
        if line and not line.endswith(" main.py")
    ]
    clean = unstaged_clean and staged_clean and not untracked
    if bool(config["governance"]["require_committed_clean_tree"]) and not clean:
        raise ValueError(f"formal E02b execution requires a clean committed tree: {untracked[:5]}")
    hashes: dict[str, str] = {}
    for relative in config["governance"]["code_hash_paths"]:
        content = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    return RepositoryState(commit=commit, clean=clean, source_sha256=hashes)


def _rank(seed: int, *parts: str) -> str:
    value = "|".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def controlled_prompts(config: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return the versioned 20 x 10 prompt factorial and disjoint splits."""

    design = config["design"]
    rows: list[dict[str, str]] = []
    for subject_index, subject in enumerate(design["subjects"]):
        for scene_index, scene in enumerate(design["scenes"]):
            prompt_id = f"p{subject_index:02d}{scene_index:02d}"
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "subject": str(subject),
                    "scene": str(scene),
                    "prompt": str(design["prompt_template"]).format(subject=subject, scene=scene),
                    "negative_prompt": str(design["negative_prompt"]),
                }
            )
    rows.sort(key=lambda row: _rank(int(design["split_rank_seed"]), str(row["prompt_id"])))
    offset = 0
    for split, raw_count in design["prompt_split_counts"].items():
        count = int(raw_count)
        for row in rows[offset : offset + count]:
            row["split"] = split
        offset += count
    return sorted(rows, key=lambda row: row["prompt_id"])


def build_generation_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the complete Model x Prompt x Seed x Resolution plan."""

    prompts = controlled_prompts(config)
    design = config["design"]
    seeds = [int(seed) for values in design["seeds"].values() for seed in values]
    seed_split = {int(seed): split for split, values in design["seeds"].items() for seed in values}
    memberships: dict[str, list[str]] = defaultdict(list)
    for suite_id, suite in config["model_groups"].items():
        for model_id in suite["models"]:
            memberships[str(model_id)].append(str(suite_id))
    plan: list[dict[str, Any]] = []
    for model_id in sorted(memberships):
        for prompt in prompts:
            for seed in seeds:
                for resolution in design["resolutions"]:
                    split = (
                        str(prompt["split"]) if seed_split[seed] == prompt["split"] else "auxiliary"
                    )
                    sample_id = f"{model_id}__{prompt['prompt_id']}__s{seed}__r{resolution}"
                    plan.append(
                        {
                            **prompt,
                            "sample_id": sample_id,
                            "model_id": model_id,
                            "suite_ids": sorted(memberships[model_id]),
                            "seed": seed,
                            "seed_split": seed_split[seed],
                            "split": split,
                            "native_resolution": int(resolution),
                            "relative_path": (
                                f"{model_id}/{resolution}/{prompt['prompt_id']}/{seed}.png"
                            ),
                        }
                    )
    return plan


def canonical_quantize(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the one registered float image to RGB uint8 conversion."""

    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"generator output must be HxWx3, observed {array.shape}")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("generator output must contain finite floating-point RGB values")
    float32 = np.ascontiguousarray(array, dtype=np.float32)
    tensor_sha = hashlib.sha256(float32.tobytes(order="C")).hexdigest()
    clipped = np.clip(float32, 0.0, 1.0)
    quantized = np.rint(clipped * 255.0).astype(np.uint8)
    metadata = {
        "tensor_dtype": str(float32.dtype),
        "tensor_shape": list(float32.shape),
        "tensor_min": float(np.min(float32)),
        "tensor_max": float(np.max(float32)),
        "tensor_sha256": tensor_sha,
        "quantized_rgb_sha256": hashlib.sha256(quantized.tobytes(order="C")).hexdigest(),
        "quantization_method": "clip_0_1_then_rint_x255_to_uint8",
    }
    return quantized, metadata


def canonical_png_bytes(rgb: np.ndarray, writer: Mapping[str, Any]) -> bytes:
    """Encode RGB bytes through the single registered Pillow PNG writer."""

    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("canonical PNG writer requires an HxWx3 uint8 array")
    stream = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(
        stream,
        format="PNG",
        compress_level=int(writer["png_compress_level"]),
        optimize=bool(writer["png_optimize"]),
    )
    return stream.getvalue()


def write_canonical_output(
    image: np.ndarray,
    target: Path,
    writer: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically quantize and write a generated image with hash evidence."""

    rgb, metadata = canonical_quantize(image)
    payload = canonical_png_bytes(rgb, writer)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return {
        **metadata,
        "writer_id": writer["id"],
        "writer_library": writer["library"],
        "writer_version": writer["version"],
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "output_size_bytes": len(payload),
    }


def decode_manifest_rgb(path: Path, record: Mapping[str, Any]) -> np.ndarray:
    """Decode a manifest image without geometry, color, or orientation transforms."""

    if sha256_file(path) != record["output_sha256"]:
        raise ValueError(f"sample SHA-256 mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.format != "PNG":
            raise ValueError(f"non-canonical image representation: {path}")
        if image.size != (int(record["native_resolution"]), int(record["native_resolution"])):
            raise ValueError(f"native geometry mismatch: {path}")
        array = np.asarray(image)
    return np.ascontiguousarray(array)


def _counterfactual_rgb(
    rgb: np.ndarray,
    counterfactual: Mapping[str, Any],
    *,
    sample_id: str,
) -> np.ndarray:
    kind = str(counterfactual["kind"])
    array = np.asarray(rgb, dtype=np.uint8)
    if kind == "canonical_requantize":
        return np.rint(array.astype(np.float32) / 255.0 * 255.0).astype(np.uint8)
    if kind == "lsb_jitter":
        seed_material = f"{counterfactual['seed']}|{sample_id}".encode()
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        changes = rng.choice(np.asarray([-1, 1], dtype=np.int16), size=array.shape)
        return np.clip(array.astype(np.int16) + changes, 0, 255).astype(np.uint8)
    if kind == "bit_depth":
        bits = int(counterfactual["bits"])
        if not 1 <= bits < 8:
            raise ValueError("bit-depth counterfactual requires 1..7 bits")
        levels = (1 << bits) - 1
        return np.rint(np.rint(array / 255.0 * levels) / levels * 255.0).astype(np.uint8)
    if kind == "png_reencode":
        return array.copy()
    raise ValueError(f"unsupported E02b counterfactual: {kind}")


def write_counterfactual(
    source: Path,
    target: Path,
    record: Mapping[str, Any],
    counterfactual_id: str,
    settings: Mapping[str, Any],
    writer: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one deterministic counterfactual and record pixel/byte invariance."""

    original = decode_manifest_rgb(source, record)
    transformed = _counterfactual_rgb(original, settings, sample_id=str(record["sample_id"]))
    derived_writer = dict(writer)
    if settings["kind"] == "png_reencode":
        derived_writer["png_compress_level"] = int(settings["compress_level"])
        derived_writer["png_optimize"] = bool(settings["optimize"])
    payload = canonical_png_bytes(transformed, derived_writer)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return {
        "sample_id": f"{record['sample_id']}__{counterfactual_id}",
        "parent_sample_id": record["sample_id"],
        "parent_sha256": record["output_sha256"],
        "counterfactual_id": counterfactual_id,
        "counterfactual": dict(settings),
        "relative_path": str(target),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "pixel_sha256": hashlib.sha256(transformed.tobytes(order="C")).hexdigest(),
        "parent_pixel_sha256": hashlib.sha256(original.tobytes(order="C")).hexdigest(),
        "pixel_identical": bool(np.array_equal(original, transformed)),
        "changed_pixel_elements": int(np.count_nonzero(original != transformed)),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_generation_manifest(
    manifest_path: Path,
    data_root: Path,
    config: Mapping[str, Any],
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate metadata completeness, factorial coverage, and split isolation."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != E02B_GENERATION_SCHEMA:
        raise ValueError("unsupported E02b generation manifest schema")
    if payload.get("configuration_sha256") != sha256_file(Path(config["_config_path"])):
        raise ValueError("generation manifest belongs to a different E02b configuration")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("generation manifest has no records")
    missing = Counter()
    file_failures: list[str] = []
    keys: set[tuple[str, str, int, int]] = set()
    split_prompts: dict[str, set[str]] = defaultdict(set)
    split_seeds: dict[str, set[int]] = defaultdict(set)
    prompt_model_coverage: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for record in records:
        for field in REQUIRED_SAMPLE_FIELDS:
            if field not in record or record[field] is None or record[field] == "":
                missing[field] += 1
        key = (
            str(record["model_id"]),
            str(record["prompt_id"]),
            int(record["seed"]),
            int(record["native_resolution"]),
        )
        if key in keys:
            raise ValueError(f"duplicate E02b factorial cell: {key}")
        keys.add(key)
        if record["split"] != "auxiliary":
            split_prompts[str(record["split"])].add(str(record["prompt_id"]))
            split_seeds[str(record["split"])].add(int(record["seed"]))
        for suite_id in record["suite_ids"]:
            prompt_model_coverage[
                (str(suite_id), str(record["prompt_id"]), int(record["native_resolution"]))
            ].add(str(record["model_id"]))
        if verify_files:
            path = data_root / str(record["relative_path"])
            if not path.is_file() or sha256_file(path) != record["output_sha256"]:
                file_failures.append(str(record["sample_id"]))
    if missing:
        raise ValueError(f"incomplete E02b generation metadata: {dict(missing)}")
    expected_plan = build_generation_plan(config)
    expected_keys = {
        (row["model_id"], row["prompt_id"], row["seed"], row["native_resolution"])
        for row in expected_plan
    }
    missing_cells = sorted(expected_keys - keys)
    extra_cells = sorted(keys - expected_keys)
    prompt_disjoint = all(
        not (split_prompts[left] & split_prompts[right])
        for index, left in enumerate(("template", "calibration", "test"))
        for right in ("template", "calibration", "test")[index + 1 :]
    )
    seed_disjoint = all(
        not (split_seeds[left] & split_seeds[right])
        for index, left in enumerate(("template", "calibration", "test"))
        for right in ("template", "calibration", "test")[index + 1 :]
    )
    coverage_failures: list[str] = []
    for (suite_id, prompt_id, resolution), observed_models in prompt_model_coverage.items():
        expected_models = set(config["model_groups"][suite_id]["models"])
        if observed_models != expected_models:
            coverage_failures.append(f"{suite_id}/{prompt_id}/{resolution}")
    checks = {
        "metadata_complete": not missing,
        "complete_factorial": not missing_cells and not extra_cells,
        "prompt_split_disjoint": prompt_disjoint,
        "seed_split_disjoint": seed_disjoint,
        "shared_prompt_model_coverage": not coverage_failures,
        "files_sha256_verified": not file_failures if verify_files else None,
        "single_writer": len({str(row["writer_id"]) for row in records}) == 1,
        "single_quantization": len({str(row["quantization_method"]) for row in records}) == 1,
        "generation_tree_clean": all(bool(row["generation_tree_clean"]) for row in records),
    }
    passed = all(value for value in checks.values() if value is not None)
    return {
        "passed": passed,
        "checks": checks,
        "record_count": len(records),
        "expected_record_count": len(expected_plan),
        "missing_factorial_cells": missing_cells[:20],
        "extra_factorial_cells": extra_cells[:20],
        "coverage_failures": coverage_failures[:20],
        "file_failures": file_failures[:20],
        "field_completeness": {
            field: 1.0 - missing[field] / len(records) for field in REQUIRED_SAMPLE_FIELDS
        },
        "split_prompt_counts": {key: len(value) for key, value in split_prompts.items()},
        "split_seed_values": {key: sorted(value) for key, value in split_seeds.items()},
    }


def repository_state_dict(state: RepositoryState) -> dict[str, Any]:
    """Return a serialization helper without leaking any credentials."""

    return asdict(state)
