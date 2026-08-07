"""Cross-validation against the immutable ``prnu-python`` implementation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from gfits.manifest import sha256_file
from gfits.matching import cross_correlation_map, prnu_python_signed_pce

EXPECTED_PRNU_COMMIT = "91e1585a39287e26f9e770b71cf9124c35d9248d"


def _upstream_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_functions(root: Path) -> ModuleType:
    source = root / "prnu" / "functions.py"
    if not source.is_file():
        raise ValueError(f"upstream functions.py not found: {source}")
    specification = importlib.util.spec_from_file_location("gfits_locked_prnu_functions", source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load upstream module: {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_prnu(upstream_root: Path, output_path: Path) -> dict[str, Any]:
    """Compare CC maps and legacy signed PCE with locked upstream functions."""

    upstream_root = upstream_root.resolve()
    output_path = output_path.resolve()
    commit = _upstream_commit(upstream_root)
    if commit != EXPECTED_PRNU_COMMIT:
        raise ValueError(
            f"unexpected prnu-python commit: {commit}; expected {EXPECTED_PRNU_COMMIT}"
        )
    upstream = _load_functions(upstream_root)
    rng = np.random.default_rng(20260808)
    correlation_cases: list[dict[str, Any]] = []
    correlation_passed = True
    for case, shapes in enumerate(
        (
            ((17, 19), (17, 19)),
            ((23, 15), (19, 13)),
            ((16, 21), (12, 21)),
            ((31, 18), (31, 14)),
        )
    ):
        first = rng.normal(size=shapes[0]).astype(np.float32)
        second = rng.normal(size=shapes[1]).astype(np.float32)
        upstream_result = upstream.crosscorr_2d(first.copy(), second.copy())
        gfits_result = cross_correlation_map(first, second).astype(np.float32)
        difference = np.abs(upstream_result.astype(np.float64) - gfits_result.astype(np.float64))
        passed = bool(np.allclose(upstream_result, gfits_result, rtol=5e-5, atol=5e-3))
        correlation_passed = correlation_passed and passed
        correlation_cases.append(
            {
                "case": case,
                "first_shape": list(shapes[0]),
                "second_shape": list(shapes[1]),
                "max_abs_error": float(np.max(difference)),
                "mean_abs_error": float(np.mean(difference)),
                "passed": passed,
            }
        )

    pce_cases: list[dict[str, Any]] = []
    pce_passed = True
    for case, peak in enumerate(((8, 9), (12, 14), (17, 7), (20, 20))):
        correlation = rng.normal(scale=0.2, size=(29, 31))
        correlation[peak] = 8.0 + case
        upstream_result = upstream.pce(correlation.copy(), neigh_radius=2)
        gfits_result = prnu_python_signed_pce(correlation, neighborhood_radius=2)
        value_error = abs(float(upstream_result["pce"]) - gfits_result.value)
        floor_from_upstream = float(
            np.square(float(upstream_result["cc"])) / abs(float(upstream_result["pce"]))
        )
        floor_error = abs(floor_from_upstream - gfits_result.floor_energy)
        passed = (
            tuple(int(value) for value in upstream_result["peak"]) == gfits_result.peak
            and value_error <= 1e-10
            and floor_error <= 1e-12
        )
        pce_passed = pce_passed and passed
        pce_cases.append(
            {
                "case": case,
                "peak": list(peak),
                "value_error": value_error,
                "floor_energy_error": floor_error,
                "passed": passed,
            }
        )

    source_path = upstream_root / "prnu" / "functions.py"
    result: dict[str, Any] = {
        "schema": "gfits.e00-prnu-cross-validation/v1",
        "phase": "E00",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "passed": correlation_passed and pce_passed,
        "upstream": {
            "repository": "https://github.com/polimi-ispl/prnu-python",
            "commit": commit,
            "functions_sha256": sha256_file(source_path),
        },
        "conventions": {
            "cross_correlation": (
                "zero-center, zero-pad to maximum shape, FFT with 180-degree rotation"
            ),
            "pce_compatibility": (
                "upstream half-open peak slice and zero-filled full-map floor energy"
            ),
            "primary_pce_difference": (
                "G-FITS primary PCE uses an inclusive circular exclusion and mn-|N| " "floor mean"
            ),
        },
        "cross_correlation_cases": correlation_cases,
        "pce_cases": pce_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
