"""Explicit score semantics for controlled E02b and later G-FITS stages."""

from __future__ import annotations

from typing import Literal

import numpy as np

from gfits.matching import DegenerateStatisticError


def _score(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def paper_fits_ratio(candidate_score: float, control_score: float) -> float:
    """Return the unstabilized FITS ratio defined in the CCS 2023 paper."""

    candidate = _score(candidate_score, "candidate_score")
    control = _score(control_score, "control_score")
    if control == 0.0:
        raise DegenerateStatisticError("FITS denominator is zero")
    return candidate / control


def paper_fits_plus_c(
    candidate_score: float,
    control_score: float,
    *,
    constant: float,
) -> float:
    """Return paper FITS+ ``(candidate + C) / (control + C)``.

    Choosing the explicit non-negative ``C`` is a calibration decision and
    must be completed without test-set access.
    """

    candidate = _score(candidate_score, "candidate_score")
    control = _score(control_score, "control_score")
    offset = _score(constant, "constant")
    if offset < 0.0:
        raise ValueError("constant must be non-negative")
    denominator = control + offset
    if denominator == 0.0:
        raise DegenerateStatisticError("FITS+ denominator is zero")
    return (candidate + offset) / denominator


def median_control_ratio(
    candidate_score: float,
    control_scores: np.ndarray,
    *,
    constant: float = 0.0,
) -> float:
    """Return a project-defined ratio against a median control statistic."""

    controls = np.asarray(control_scores, dtype=np.float64)
    if controls.ndim != 1 or controls.size < 3:
        raise ValueError("median control requires at least three one-dimensional scores")
    if not np.isfinite(controls).all():
        raise ValueError("control_scores contains a non-finite value")
    return paper_fits_plus_c(
        candidate_score,
        float(np.median(controls)),
        constant=constant,
    )


def gallery_complement_ratio(
    candidate_score: float,
    other_gallery_scores: np.ndarray,
    *,
    reducer: Literal["mean", "median"] = "mean",
    constant: float = 0.0,
) -> float:
    """Return a closed-gallery contrast, explicitly not a FITS control.

    Competing candidate templates do not isolate a software-only nuisance
    term. This relative score is therefore descriptive and cannot support a
    software-noise calibration claim.
    """

    controls = np.asarray(other_gallery_scores, dtype=np.float64)
    if controls.ndim != 1 or controls.size == 0:
        raise ValueError("other_gallery_scores must be a non-empty one-dimensional array")
    if not np.isfinite(controls).all():
        raise ValueError("other_gallery_scores contains a non-finite value")
    if reducer == "mean":
        control = float(np.mean(controls))
    elif reducer == "median":
        control = float(np.median(controls))
    else:
        raise ValueError(f"unsupported gallery reducer: {reducer}")
    return paper_fits_plus_c(candidate_score, control, constant=constant)


def robust_zscore(candidate_score: float, control_scores: np.ndarray) -> float:
    """Standardize a candidate against independent controls using median/MAD."""

    candidate = _score(candidate_score, "candidate_score")
    controls = np.asarray(control_scores, dtype=np.float64)
    if controls.ndim != 1 or controls.size < 3:
        raise ValueError("robust z-score requires at least three one-dimensional controls")
    if not np.isfinite(controls).all():
        raise ValueError("control_scores contains a non-finite value")
    center = float(np.median(controls))
    scale = float(1.4826 * np.median(np.abs(controls - center)))
    if scale == 0.0:
        raise DegenerateStatisticError("robust z-score control MAD is zero")
    return (candidate - center) / scale


def fit_nuisance_subspace(
    control_vectors: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a centered nuisance basis from independent control vectors only."""

    controls = np.asarray(control_vectors, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[0] < 3:
        raise ValueError("nuisance subspace requires at least three control vectors")
    if not np.isfinite(controls).all():
        raise ValueError("control_vectors contains a non-finite value")
    maximum_rank = min(controls.shape[0] - 1, controls.shape[1])
    if not 1 <= int(rank) <= maximum_rank:
        raise ValueError(f"rank must be between 1 and {maximum_rank}")
    center = np.mean(controls, axis=0)
    _, _, right = np.linalg.svd(controls - center, full_matrices=False)
    return center, right[: int(rank)]


def nuisance_subspace_residual_score(
    candidate_vector: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
) -> float:
    """Return residual energy after removing a fitted nuisance subspace."""

    candidate = np.asarray(candidate_vector, dtype=np.float64)
    location = np.asarray(center, dtype=np.float64)
    components = np.asarray(basis, dtype=np.float64)
    if candidate.ndim != 1 or location.shape != candidate.shape:
        raise ValueError("candidate and nuisance center must be equal-length vectors")
    if components.ndim != 2 or components.shape[1] != candidate.size:
        raise ValueError("nuisance basis has an incompatible shape")
    if not all(np.isfinite(value).all() for value in (candidate, location, components)):
        raise ValueError("nuisance subspace inputs must be finite")
    gram = components @ components.T
    if not np.allclose(gram, np.eye(components.shape[0]), atol=1.0e-8):
        raise ValueError("nuisance basis must have orthonormal rows")
    centered = candidate - location
    residual = centered - components.T @ (components @ centered)
    return float(np.dot(residual, residual))
