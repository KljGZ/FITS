"""Pre-registered residual-level synthetic validation for Phase E00."""

from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gfits.manifest import sha256_file
from gfits.matching import fits, fits_plus, log_ratio, normalized_cross_correlation

SYNTHETIC_MODELS = ("additive", "multiplicative", "nonadditive")


def _required(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise ValueError(f"missing configuration key: {name}")
    return mapping[name]


def load_e00_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the E00 YAML configuration."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("phase") != "E00":
        raise ValueError("configuration must be a Phase E00 mapping")
    synthetic = _required(payload, "synthetic")
    gate = _required(payload, "gate")
    if not isinstance(synthetic, dict) or not isinstance(gate, dict):
        raise ValueError("synthetic and gate sections must be mappings")
    if tuple(synthetic.get("models", ())) != SYNTHETIC_MODELS:
        raise ValueError(f"synthetic.models must be {SYNTHETIC_MODELS}")
    return payload


def _standard_pattern(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    pattern = rng.normal(size=shape)
    pattern -= np.mean(pattern)
    rms = float(np.sqrt(np.mean(np.square(pattern))))
    if rms == 0.0:
        raise RuntimeError("random pattern unexpectedly has zero energy")
    return pattern / rms


def _residual_signal(
    source: np.ndarray,
    software: np.ndarray,
    *,
    model: str,
    gamma: float,
    beta: float,
) -> np.ndarray:
    if model == "additive":
        return gamma * source + beta * software
    if model == "multiplicative":
        return (1.0 + gamma * source) * (1.0 + beta * software) - 1.0
    if model == "nonadditive":
        return np.tanh(gamma * source + beta * software)
    raise ValueError(f"unsupported synthetic model: {model}")


def _observation(
    source: np.ndarray,
    software: np.ndarray,
    rng: np.random.Generator,
    *,
    model: str,
    gamma: float,
    beta: float,
    noise_std: float,
    averaged_samples: int = 1,
) -> np.ndarray:
    if averaged_samples <= 0:
        raise ValueError("averaged_samples must be positive")
    signal = _residual_signal(source, software, model=model, gamma=gamma, beta=beta)
    effective_noise = noise_std / np.sqrt(averaged_samples)
    return signal + rng.normal(scale=effective_noise, size=source.shape)


def _positive_ncc(first: np.ndarray, second: np.ndarray) -> float:
    score = normalized_cross_correlation(first, second)
    if score <= 0.0:
        raise RuntimeError(
            "synthetic nuisance correlation was non-positive; log-ratio is undefined"
        )
    return score


def _hypothesis_rows(
    rng: np.random.Generator,
    sources: Sequence[np.ndarray],
    software: np.ndarray,
    *,
    model: str,
    trials: int,
    controls_nz: int,
    reference_samples: int,
    gamma: float,
    beta: float,
    noise_std: float,
    stabilizer: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in range(trials):
        selected = rng.choice(len(sources), size=controls_nz + 2, replace=False)
        candidate_id = int(selected[0])
        h0_query_id = int(selected[1])
        control_ids = [int(value) for value in selected[2:]]
        candidate_reference = _observation(
            sources[candidate_id],
            software,
            rng,
            model=model,
            gamma=gamma,
            beta=beta,
            noise_std=noise_std,
            averaged_samples=reference_samples,
        )
        control_references = [
            _observation(
                sources[control_id],
                software,
                rng,
                model=model,
                gamma=gamma,
                beta=beta,
                noise_std=noise_std,
                averaged_samples=reference_samples,
            )
            for control_id in control_ids
        ]
        for label, query_id in (("H0", h0_query_id), ("H1", candidate_id)):
            query = _observation(
                sources[query_id],
                software,
                rng,
                model=model,
                gamma=gamma,
                beta=beta,
                noise_std=noise_std,
            )
            candidate_score = _positive_ncc(query, candidate_reference)
            control_scores = np.asarray(
                [_positive_ncc(query, reference) for reference in control_references]
            )
            control_median = float(np.median(control_scores))
            resolution = f"{query.shape[1]}x{query.shape[0]}"
            rows.append(
                {
                    "query_id": f"{model}:hypothesis:{trial}:{label}",
                    "query_source_id": query_id,
                    "candidate_source_id": candidate_id,
                    "control_type": "same_pipeline_different_source",
                    "pipeline_query": f"synthetic:{model}",
                    "pipeline_reference": f"synthetic:{model}",
                    "resolution_query": resolution,
                    "resolution_reference": resolution,
                    "extractor": "identity_residual",
                    "aggregator": f"mean_{reference_samples}",
                    "similarity": "ncc",
                    "seed": seed,
                    "model": model,
                    "trial": trial,
                    "hypothesis": label,
                    "label_match": label == "H1",
                    "raw_score": candidate_score,
                    "candidate_score": candidate_score,
                    "control_mean": float(np.mean(control_scores)),
                    "control_median": control_median,
                    "control_mad": float(np.median(np.abs(control_scores - control_median))),
                    "n_controls": controls_nz,
                    "fits_score": fits(
                        candidate_score,
                        float(control_scores[0]),
                        stabilizer=stabilizer,
                    ),
                    "fits_plus_score": fits_plus(
                        candidate_score,
                        control_scores,
                        stabilizer=stabilizer,
                    ),
                    "log_ratio": log_ratio(
                        candidate_score,
                        control_median,
                        stabilizer=stabilizer,
                    ),
                }
            )
    return rows


def _variance_rows(
    rng: np.random.Generator,
    sources: Sequence[np.ndarray],
    software: np.ndarray,
    *,
    model: str,
    replicates: int,
    nz_values: Sequence[int],
    reference_samples: int,
    gamma: float,
    beta: float,
    noise_std: float,
    stabilizer: float,
    seed: int,
) -> list[dict[str, Any]]:
    query_id, candidate_id = 0, 1
    eligible_controls = np.arange(2, len(sources))
    query = _observation(
        sources[query_id],
        software,
        rng,
        model=model,
        gamma=gamma,
        beta=beta,
        noise_std=noise_std,
    )
    candidate_reference = _observation(
        sources[candidate_id],
        software,
        rng,
        model=model,
        gamma=gamma,
        beta=beta,
        noise_std=noise_std,
        averaged_samples=reference_samples,
    )
    candidate_score = _positive_ncc(query, candidate_reference)
    maximum_nz = max(nz_values)
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        control_ids = rng.choice(eligible_controls, size=maximum_nz, replace=True)
        control_scores = np.asarray(
            [
                _positive_ncc(
                    query,
                    _observation(
                        sources[int(control_id)],
                        software,
                        rng,
                        model=model,
                        gamma=gamma,
                        beta=beta,
                        noise_std=noise_std,
                        averaged_samples=reference_samples,
                    ),
                )
                for control_id in control_ids
            ]
        )
        for nz in nz_values:
            selected_scores = control_scores[:nz]
            median_score = float(np.median(selected_scores))
            resolution = f"{query.shape[1]}x{query.shape[0]}"
            rows.append(
                {
                    "query_id": f"{model}:variance-query:0",
                    "query_source_id": query_id,
                    "candidate_source_id": candidate_id,
                    "control_type": "same_pipeline_different_source",
                    "pipeline_query": f"synthetic:{model}",
                    "pipeline_reference": f"synthetic:{model}",
                    "resolution_query": resolution,
                    "resolution_reference": resolution,
                    "extractor": "identity_residual",
                    "aggregator": f"mean_{reference_samples}",
                    "similarity": "ncc",
                    "seed": seed,
                    "model": model,
                    "replicate": replicate,
                    "n_controls": nz,
                    "label_match": False,
                    "raw_score": candidate_score,
                    "candidate_score_fixed": candidate_score,
                    "control_mean": float(np.mean(selected_scores)),
                    "control_median": median_score,
                    "control_mad": float(np.median(np.abs(selected_scores - median_score))),
                    "fits_score": fits(
                        candidate_score,
                        float(selected_scores[0]),
                        stabilizer=stabilizer,
                    ),
                    "fits_plus_score": fits_plus(
                        candidate_score,
                        selected_scores,
                        stabilizer=stabilizer,
                    ),
                    "log_ratio": log_ratio(
                        candidate_score,
                        median_score,
                        stabilizer=stabilizer,
                    ),
                }
            )
    return rows


def _summarize(
    hypothesis_rows: Sequence[Mapping[str, Any]],
    variance_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    h0_low, h0_high = (float(value) for value in gate["h0_fits_plus_median_interval"])
    h1_minimum = float(gate["h1_fits_plus_median_min"])
    variance_ratio_maximum = float(gate["variance_final_to_initial_max"])
    slope_maximum = float(gate["variance_log_slope_max"])
    by_model: dict[str, Any] = {}
    all_passed = True
    for model in SYNTHETIC_MODELS:
        h0 = np.asarray(
            [
                row["fits_plus_score"]
                for row in hypothesis_rows
                if row["model"] == model and row["hypothesis"] == "H0"
            ]
        )
        h1 = np.asarray(
            [
                row["fits_plus_score"]
                for row in hypothesis_rows
                if row["model"] == model and row["hypothesis"] == "H1"
            ]
        )
        h0_log = np.asarray(
            [
                row["log_ratio"]
                for row in hypothesis_rows
                if row["model"] == model and row["hypothesis"] == "H0"
            ]
        )
        h1_log = np.asarray(
            [
                row["log_ratio"]
                for row in hypothesis_rows
                if row["model"] == model and row["hypothesis"] == "H1"
            ]
        )
        nz_values = sorted(
            {int(row["n_controls"]) for row in variance_rows if row["model"] == model}
        )
        variances = {
            str(nz): float(
                np.var(
                    [
                        row["log_ratio"]
                        for row in variance_rows
                        if row["model"] == model and int(row["n_controls"]) == nz
                    ],
                    ddof=1,
                )
            )
            for nz in nz_values
        }
        initial_variance = variances[str(nz_values[0])]
        final_variance = variances[str(nz_values[-1])]
        final_to_initial = final_variance / initial_variance
        log_slope = float(
            np.polyfit(
                np.log(np.asarray(nz_values, dtype=np.float64)),
                np.log(np.asarray([variances[str(nz)] for nz in nz_values])),
                deg=1,
            )[0]
        )
        h0_median = float(np.median(h0))
        h1_median = float(np.median(h1))
        checks = {
            "h0_centered_near_one": h0_low <= h0_median <= h0_high,
            "h1_above_minimum": h1_median >= h1_minimum,
            "h1_above_h0": h1_median > h0_median,
            "variance_final_to_initial": final_to_initial <= variance_ratio_maximum,
            "variance_log_slope": log_slope <= slope_maximum,
        }
        model_passed = all(checks.values())
        all_passed = all_passed and model_passed
        by_model[model] = {
            "h0_fits_plus_median": h0_median,
            "h1_fits_plus_median": h1_median,
            "h0_log_ratio_median": float(np.median(h0_log)),
            "h1_log_ratio_median": float(np.median(h1_log)),
            "log_ratio_variance_by_nz": variances,
            "variance_final_to_initial": final_to_initial,
            "variance_log_slope": log_slope,
            "checks": checks,
            "passed": model_passed,
        }
    return {"passed": all_passed, "models": by_model}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_diagnostics(
    path: Path,
    hypothesis_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {"H0": "#4C78A8", "H1": "#E45756"}
    for model in SYNTHETIC_MODELS:
        for hypothesis in ("H0", "H1"):
            values = [
                row["fits_plus_score"]
                for row in hypothesis_rows
                if row["model"] == model and row["hypothesis"] == hypothesis
            ]
            axes[0, 0].hist(
                values,
                bins=35,
                alpha=0.22,
                color=colors[hypothesis],
                label=hypothesis if model == SYNTHETIC_MODELS[0] else None,
            )
        axes[0, 1].scatter(
            [summary["models"][model]["h0_fits_plus_median"]],
            [summary["models"][model]["h1_fits_plus_median"]],
            label=model,
            s=65,
        )
        variances = summary["models"][model]["log_ratio_variance_by_nz"]
        nz_values = [int(value) for value in variances]
        axes[1, 0].plot(
            nz_values,
            [variances[str(value)] for value in nz_values],
            marker="o",
            label=model,
        )
    axes[0, 0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(title="FITS+ score distributions", xlabel="FITS+", ylabel="count")
    axes[0, 0].legend()
    axes[0, 1].axvline(1.0, color="black", linestyle=":", linewidth=1)
    axes[0, 1].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[0, 1].set(
        title="Median separation by residual model",
        xlabel="H0 median FITS+",
        ylabel="H1 median FITS+",
    )
    axes[0, 1].legend()
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(
        title="Control-estimator variance",
        xlabel="number of controls nZ",
        ylabel="variance of log-ratio",
    )
    axes[1, 0].legend()
    axes[1, 1].axis("off")
    lines = ["Pre-registered E00 gate", f"overall: {'PASS' if summary['passed'] else 'FAIL'}"]
    for model in SYNTHETIC_MODELS:
        result = summary["models"][model]
        lines.append(
            f"{model}: H0={result['h0_fits_plus_median']:.3f}, "
            f"H1={result['h1_fits_plus_median']:.3f}, "
            f"var ratio={result['variance_final_to_initial']:.3f}"
        )
    axes[1, 1].text(0.02, 0.98, "\n".join(lines), va="top", family="monospace")
    figure.suptitle("G-FITS Phase E00 synthetic mechanism validation")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


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


def validate_synthetic_fits(
    config_path: Path,
    output_directory: Path,
    *,
    profile: str = "gate",
) -> dict[str, Any]:
    """Execute the configured E00 synthetic validation and write raw evidence."""

    config_path = config_path.resolve()
    output_directory = output_directory.resolve()
    config = load_e00_config(config_path)
    synthetic = config["synthetic"]
    gate = config["gate"]
    if profile not in {"development", "gate"}:
        raise ValueError("profile must be 'development' or 'gate'")
    seed = int(synthetic[f"{profile}_seed"])
    rng = np.random.default_rng(seed)
    shape = tuple(int(value) for value in synthetic["shape"])
    if len(shape) != 2:
        raise ValueError("synthetic.shape must have two dimensions")
    source_count = int(synthetic["source_count"])
    controls_nz = int(synthetic["hypothesis_controls_nz"])
    nz_values = tuple(int(value) for value in synthetic["variance_nz_values"])
    if source_count < max(controls_nz + 2, max(nz_values) + 2):
        raise ValueError("source_count is too small for configured controls")
    sources = [_standard_pattern(rng, shape) for _ in range(source_count)]
    gamma = float(synthetic["gamma"])
    beta = float(synthetic["beta"])
    noise_std = float(synthetic["noise_std"])
    reference_samples = int(synthetic["reference_samples"])
    stabilizer = float(synthetic["stabilizer"])
    hypothesis_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    for model in SYNTHETIC_MODELS:
        software = _standard_pattern(rng, shape)
        hypothesis_rows.extend(
            _hypothesis_rows(
                rng,
                sources,
                software,
                model=model,
                trials=int(synthetic["hypothesis_trials"]),
                controls_nz=controls_nz,
                reference_samples=reference_samples,
                gamma=gamma,
                beta=beta,
                noise_std=noise_std,
                stabilizer=stabilizer,
                seed=seed,
            )
        )
        variance_rows.extend(
            _variance_rows(
                rng,
                sources,
                software,
                model=model,
                replicates=int(synthetic["variance_replicates"]),
                nz_values=nz_values,
                reference_samples=reference_samples,
                gamma=gamma,
                beta=beta,
                noise_std=noise_std,
                stabilizer=stabilizer,
                seed=seed,
            )
        )
    gate_summary = _summarize(hypothesis_rows, variance_rows, gate)
    hypothesis_path = output_directory / "synthetic_scores.csv"
    variance_path = output_directory / "variance_by_nz.csv"
    diagnostics_path = output_directory / "diagnostics.png"
    _write_csv(hypothesis_path, hypothesis_rows)
    _write_csv(variance_path, variance_rows)
    _write_diagnostics(diagnostics_path, hypothesis_rows, gate_summary)

    repository_root = Path(__file__).resolve().parents[2]
    source_paths = [
        repository_root / "src" / "gfits" / "matching.py",
        repository_root / "src" / "gfits" / "synthetic.py",
        config_path,
    ]
    summary: dict[str, Any] = {
        "schema": "gfits.e00-synthetic-validation/v1",
        "phase": "E00",
        "profile": profile,
        "passed": gate_summary["passed"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "configuration": str(config_path),
        "configuration_sha256": sha256_file(config_path),
        "source_sha256": {
            str(path.relative_to(repository_root).as_posix()): sha256_file(path)
            for path in source_paths
        },
        "git": _git_metadata(repository_root),
        "definitions": config["definitions"],
        "gate_thresholds": gate,
        "gate_summary": gate_summary,
        "artifacts": {
            hypothesis_path.name: sha256_file(hypothesis_path),
            variance_path.name: sha256_file(variance_path),
            diagnostics_path.name: sha256_file(diagnostics_path),
        },
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
