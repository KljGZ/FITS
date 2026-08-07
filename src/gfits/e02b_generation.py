"""Revision-locked multi-generator image production for controlled E02b data."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gfits.e02b_data import (
    E02B_GENERATION_SCHEMA,
    build_generation_plan,
    load_e02b_config,
    repository_state,
    repository_state_dict,
    validate_generation_manifest,
    write_canonical_output,
    write_json_atomic,
)
from gfits.manifest import sha256_file

E02B_FRAGMENT_SCHEMA = "gfits.e02b-generation-fragment/v1"


def _snapshot_allow_patterns(model: Mapping[str, Any]) -> list[str]:
    """Return the minimal audited config/tokenizer/weight file policy."""

    patterns = [
        "*.json",
        "**/*.json",
        "*.txt",
        "**/*.txt",
        "*.model",
        "**/*.model",
        "*.py",
        "**/*.py",
    ]
    weight_format = str(model["weight_format"])
    variant = model.get("variant")
    if weight_format == "safetensors":
        suffix = f".{variant}.safetensors" if variant else ".safetensors"
    elif weight_format == "pytorch_bin":
        if variant:
            raise ValueError("binary E02b weights may not declare a variant")
        suffix = ".bin"
    else:
        raise ValueError(f"unsupported E02b weight format: {weight_format}")
    patterns.extend((f"*{suffix}", f"**/*{suffix}"))
    return patterns


def _inventory(root: Path, *, subdirectory: str | None = None) -> dict[str, Any]:
    target = root / subdirectory if subdirectory else root
    if not target.is_dir():
        return {"root": str(target), "file_count": 0, "size_bytes": 0, "sha256": None}
    rows: list[str] = []
    size = 0
    for path in sorted(value for value in target.rglob("*") if value.is_file()):
        relative = path.relative_to(target).as_posix()
        file_size = path.stat().st_size
        size += file_size
        rows.append(f"{relative}\t{file_size}\t{sha256_file(path)}")
    digest = hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()
    return {
        "root": str(target.resolve()),
        "file_count": len(rows),
        "size_bytes": size,
        "sha256": digest,
    }


def _download_repositories(model: Mapping[str, Any], cache_dir: Path) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError("E02b generation requires environment-e02b.yml") from error
    repositories: list[dict[str, Any]] = []
    for repository in model["repositories"]:
        root = Path(
            snapshot_download(
                repo_id=str(repository["repo_id"]),
                revision=str(repository["revision"]),
                cache_dir=str(cache_dir.resolve()),
                allow_patterns=_snapshot_allow_patterns(model),
            )
        )
        inventory = _inventory(root)
        repositories.append(
            {
                "repo_id": repository["repo_id"],
                "revision": repository["revision"],
                "snapshot_root": str(root.resolve()),
                "inventory": inventory,
                "vae_inventory": _inventory(root, subdirectory="vae"),
                "movq_inventory": _inventory(root, subdirectory="movq"),
            }
        )
    return repositories


def _combined_hash(repositories: Sequence[Mapping[str, Any]], key: str) -> str:
    rows = []
    for repository in repositories:
        value = repository[key]
        if isinstance(value, Mapping):
            value = value.get("sha256")
        rows.append(f"{repository['repo_id']}@{repository['revision']}={value}")
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def _load_pipeline(
    model: Mapping[str, Any],
    repositories: Sequence[Mapping[str, Any]],
    device: str,
) -> Any:
    try:
        import torch
        from diffusers import (
            DPMSolverMultistepScheduler,
            KandinskyV22Pipeline,
            KandinskyV22PriorPipeline,
            PixArtSigmaPipeline,
            StableDiffusionPipeline,
        )
    except ImportError as error:
        raise ImportError("E02b generation requires environment-e02b.yml") from error
    dtype = torch.float16 if model["compute_dtype"] == "float16" else torch.float32
    adapter = str(model["adapter"])
    common = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "use_safetensors": model["weight_format"] == "safetensors",
    }
    if model.get("variant"):
        common["variant"] = str(model["variant"])
    if adapter == "stable_diffusion":
        pipeline = StableDiffusionPipeline.from_pretrained(
            repositories[0]["snapshot_root"],
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            **common,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            **dict(model["scheduler_parameters"]),
        )
        pipeline.set_progress_bar_config(disable=True)
        return pipeline.to(device)
    if adapter == "kandinsky22":
        prior = KandinskyV22PriorPipeline.from_pretrained(
            repositories[0]["snapshot_root"], **common
        ).to(device)
        decoder = KandinskyV22Pipeline.from_pretrained(
            repositories[1]["snapshot_root"], **common
        ).to(device)
        prior.set_progress_bar_config(disable=True)
        decoder.set_progress_bar_config(disable=True)
        return prior, decoder
    if adapter == "pixart_sigma":
        pipeline = PixArtSigmaPipeline.from_pretrained(
            repositories[0]["snapshot_root"], **common
        ).to(device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline
    raise ValueError(f"unsupported E02b model adapter: {adapter}")


def _torch_generators(device: str, seeds: Sequence[int]) -> list[Any]:
    import torch

    return [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]


def _generate_batch(
    pipeline: Any,
    model: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    resolution: int,
    device: str,
) -> np.ndarray:
    prompts = [str(row["prompt"]) for row in rows]
    negative = [str(row["negative_prompt"]) for row in rows]
    seeds = [int(row["seed"]) for row in rows]
    adapter = str(model["adapter"])
    if adapter == "stable_diffusion":
        output = pipeline(
            prompt=prompts,
            negative_prompt=negative,
            generator=_torch_generators(device, seeds),
            height=resolution,
            width=resolution,
            num_inference_steps=int(model["steps"]),
            guidance_scale=float(model["guidance_scale"]),
            output_type="np",
        )
        return np.asarray(output.images)
    if adapter == "pixart_sigma":
        output = pipeline(
            prompt=prompts,
            negative_prompt=negative,
            generator=_torch_generators(device, seeds),
            height=resolution,
            width=resolution,
            num_inference_steps=int(model["steps"]),
            guidance_scale=float(model["guidance_scale"]),
            output_type="np",
            use_resolution_binning=False,
        )
        return np.asarray(output.images)
    if adapter == "kandinsky22":
        if len(rows) != 1:
            raise ValueError("the deterministic Kandinsky adapter requires batch_size=1")
        prior, decoder = pipeline
        prior_output = prior(
            prompt=prompts[0],
            negative_prompt=negative[0],
            generator=_torch_generators(device, seeds)[0],
            num_inference_steps=int(model["prior_steps"]),
            guidance_scale=float(model["prior_guidance_scale"]),
        )
        output = decoder(
            image_embeds=prior_output.image_embeds,
            negative_image_embeds=prior_output.negative_image_embeds,
            generator=_torch_generators(device, seeds)[0],
            height=resolution,
            width=resolution,
            num_inference_steps=int(model["steps"]),
            guidance_scale=float(model["guidance_scale"]),
            output_type="np",
        )
        return np.asarray(output.images)
    raise ValueError(f"unsupported E02b model adapter: {adapter}")


def _runtime_metadata(device: str) -> dict[str, Any]:
    import torch

    packages = (
        "accelerate",
        "diffusers",
        "huggingface-hub",
        "numpy",
        "Pillow",
        "safetensors",
        "torch",
        "transformers",
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "cuda": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }


def generate_e02b_shard(
    config_path: Path,
    repository_root: Path,
    data_root: Path,
    cache_root: Path,
    fragment_path: Path,
    *,
    model_id: str,
    resolution: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Generate one model-resolution shard and persist resumable hash evidence."""

    config = load_e02b_config(config_path.resolve())
    state = repository_state(repository_root.resolve(), config)
    if model_id not in config["models"]:
        raise ValueError(f"unknown E02b model: {model_id}")
    if int(resolution) not in config["design"]["resolutions"]:
        raise ValueError(f"unregistered E02b resolution: {resolution}")
    model = config["models"][model_id]
    registered_batch_size = int(config["design"]["generation_batch_size"])
    if batch_size != registered_batch_size:
        raise ValueError(
            f"formal generation batch_size must be {registered_batch_size}, got {batch_size}"
        )
    repositories = _download_repositories(model, cache_root.resolve())
    model_hash = _combined_hash(repositories, "inventory")
    component_rows = []
    for repository in repositories:
        component = repository["vae_inventory"]
        if not component["sha256"]:
            component = repository["movq_inventory"]
        component_rows.append({**repository, "component_inventory": component})
    vae_hash = _combined_hash(component_rows, "component_inventory")
    plan = [
        row
        for row in build_generation_plan(config)
        if row["model_id"] == model_id and row["native_resolution"] == int(resolution)
    ]
    existing: dict[str, dict[str, Any]] = {}
    if fragment_path.is_file():
        previous = json.loads(fragment_path.read_text(encoding="utf-8"))
        if previous.get("schema") != E02B_FRAGMENT_SCHEMA:
            raise ValueError("unsupported E02b fragment schema")
        if previous.get("configuration_sha256") != sha256_file(config_path.resolve()):
            raise ValueError("existing fragment belongs to another E02b configuration")
        if previous.get("repository_state", {}).get("commit") != state.commit:
            raise ValueError("existing fragment was generated by another code commit")
        existing = {str(row["sample_id"]): row for row in previous["records"]}
    complete: dict[str, dict[str, Any]] = {}
    for row in plan:
        old = existing.get(str(row["sample_id"]))
        if old:
            path = data_root / str(old["relative_path"])
            if path.is_file() and sha256_file(path) == old["output_sha256"]:
                complete[str(row["sample_id"])] = old
    pending = [row for row in plan if row["sample_id"] not in complete]
    pipeline = _load_pipeline(model, repositories, device) if pending else None
    started = time.time()
    writer = config["design"]["writer"]
    observed_writer_version = importlib.metadata.version(str(writer["library"]))
    if observed_writer_version != str(writer["version"]):
        raise ValueError(
            f"canonical writer version mismatch: {observed_writer_version} != {writer['version']}"
        )
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        generated = _generate_batch(pipeline, model, batch, int(resolution), device)
        if generated.shape[0] != len(batch):
            raise ValueError("generator returned an unexpected batch dimension")
        for row, image in zip(batch, generated, strict=True):
            target = data_root / str(row["relative_path"])
            output = write_canonical_output(image, target, writer)
            complete[str(row["sample_id"])] = {
                **row,
                "model_hash": model_hash,
                "base_model": model["base_model"],
                "checkpoint_hash": model_hash,
                "vae": model["vae"],
                "vae_hash": vae_hash,
                "sampler": model["sampler"],
                "scheduler": model["scheduler"],
                "scheduler_parameters": dict(model.get("scheduler_parameters", {})),
                "steps": int(model["steps"]),
                "guidance_scale": float(model["guidance_scale"]),
                "generation_code_commit": state.commit,
                "generation_tree_clean": state.clean,
                **output,
            }
        fragment = {
            "schema": E02B_FRAGMENT_SCHEMA,
            "configuration_sha256": sha256_file(config_path.resolve()),
            "model_id": model_id,
            "resolution": int(resolution),
            "repository_state": repository_state_dict(state),
            "repositories": repositories,
            "runtime": _runtime_metadata(device),
            "generation_batch_size": registered_batch_size,
            "elapsed_seconds": time.time() - started,
            "records": [complete[key] for key in sorted(complete)],
        }
        write_json_atomic(fragment_path.resolve(), fragment)
    return {
        "ok": len(complete) == len(plan),
        "fragment": str(fragment_path.resolve()),
        "model_id": model_id,
        "resolution": int(resolution),
        "record_count": len(complete),
        "expected_record_count": len(plan),
        "repository_commit": state.commit,
        "tree_clean": state.clean,
        "model_hash": model_hash,
        "vae_hash": vae_hash,
    }


def merge_e02b_fragments(
    config_path: Path,
    repository_root: Path,
    data_root: Path,
    fragment_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Merge exact shard fragments into the formal controlled-data manifest."""

    config = load_e02b_config(config_path.resolve())
    state = repository_state(repository_root.resolve(), config)
    fragments = sorted(fragment_root.resolve().glob("*.json"))
    if not fragments:
        raise ValueError("no E02b generation fragments were found")
    records: dict[str, dict[str, Any]] = {}
    fragment_evidence = []
    for path in fragments:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != E02B_FRAGMENT_SCHEMA:
            raise ValueError(f"unsupported fragment: {path}")
        if payload.get("configuration_sha256") != sha256_file(config_path.resolve()):
            raise ValueError(f"fragment configuration mismatch: {path}")
        if payload.get("repository_state", {}).get("commit") != state.commit:
            raise ValueError(f"fragment code commit mismatch: {path}")
        for record in payload["records"]:
            sample_id = str(record["sample_id"])
            if sample_id in records:
                raise ValueError(f"duplicate generated sample across fragments: {sample_id}")
            records[sample_id] = record
        fragment_evidence.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "record_count": len(payload["records"]),
            }
        )
    manifest = {
        "schema": E02B_GENERATION_SCHEMA,
        "configuration_sha256": sha256_file(config_path.resolve()),
        "repository_state": repository_state_dict(state),
        "data_root": str(data_root.resolve()),
        "fragments": fragment_evidence,
        "records": [records[key] for key in sorted(records)],
    }
    write_json_atomic(output_path.resolve(), manifest)
    validation = validate_generation_manifest(
        output_path.resolve(), data_root.resolve(), config, verify_files=True
    )
    if not validation["passed"]:
        raise ValueError(f"merged E02b generation manifest failed: {validation}")
    return {
        "ok": True,
        "manifest": str(output_path.resolve()),
        "manifest_sha256": sha256_file(output_path.resolve()),
        "record_count": len(records),
        "validation": validation,
    }
