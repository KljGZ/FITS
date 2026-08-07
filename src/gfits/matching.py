"""Numerical matching primitives used by the G-FITS staged experiments.

The original FITS division is a candidate test statistic divided by a
same-pipeline, different-source control statistic. ``fits_plus`` is explicitly
a G-FITS project extension: it replaces a single control with a robust median
over multiple controls. It is not presented as terminology from the CCS 2023
paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


class DegenerateStatisticError(ValueError):
    """Raised when a statistic is undefined because its energy is zero."""


@dataclass(frozen=True)
class PceResult:
    """Signed peak-to-correlation-energy result."""

    peak: tuple[int, int]
    peak_value: float
    floor_energy: float
    value: float
    excluded_samples: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def mean_center(value: np.ndarray) -> np.ndarray:
    """Return a float64, zero-mean copy of a two-dimensional signal."""

    array = _matrix(value, "value")
    return array - np.mean(array, dtype=np.float64)


def cross_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return aligned, zero-mean cross-correlation (CC)."""

    first_centered = mean_center(first)
    second_centered = mean_center(second)
    if first_centered.shape != second_centered.shape:
        raise ValueError("aligned CC requires equal shapes")
    return float(np.sum(first_centered * second_centered, dtype=np.float64))


def normalized_cross_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return aligned, zero-mean normalized cross-correlation (NCC)."""

    first_centered = mean_center(first)
    second_centered = mean_center(second)
    if first_centered.shape != second_centered.shape:
        raise ValueError("aligned NCC requires equal shapes")
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator == 0.0:
        raise DegenerateStatisticError("NCC is undefined for a zero-energy signal")
    return float(np.sum(first_centered * second_centered, dtype=np.float64) / denominator)


def cross_correlation_map(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the circular two-dimensional CC map used by ``prnu-python``.

    Inputs are independently zero-centered and zero-padded to their maximum
    height and width. The orientation follows the locked upstream implementation
    (FFT of the first signal times FFT of the second signal rotated by 180
    degrees). Unlike upstream, this function never mutates its inputs and keeps
    float64 output for the primary implementation.
    """

    first_centered = mean_center(first)
    second_centered = mean_center(second)
    height = max(first_centered.shape[0], second_centered.shape[0])
    width = max(first_centered.shape[1], second_centered.shape[1])
    first_padded = np.pad(
        first_centered,
        ((0, height - first_centered.shape[0]), (0, width - first_centered.shape[1])),
        mode="constant",
    )
    second_padded = np.pad(
        second_centered,
        ((0, height - second_centered.shape[0]), (0, width - second_centered.shape[1])),
        mode="constant",
    )
    first_fft = np.fft.fft2(first_padded)
    second_fft = np.fft.fft2(np.rot90(second_padded, 2))
    return np.real(np.fft.ifft2(first_fft * second_fft)).astype(np.float64, copy=False)


def aligned_peak_position(shape: tuple[int, int]) -> tuple[int, int]:
    """Return the zero-shift peak index for :func:`cross_correlation_map`."""

    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape must contain two positive dimensions")
    return shape[0] - 1, shape[1] - 1


def _peak_position(
    correlation: np.ndarray,
    peak_position: tuple[int, int] | None,
    peak_mode: Literal["max", "absolute"],
) -> tuple[int, int]:
    if peak_position is None:
        if peak_mode == "max":
            flat_index = int(np.argmax(correlation))
        elif peak_mode == "absolute":
            flat_index = int(np.argmax(np.abs(correlation)))
        else:
            raise ValueError(f"unsupported peak_mode: {peak_mode}")
        return tuple(int(value) for value in np.unravel_index(flat_index, correlation.shape))
    if len(peak_position) != 2:
        raise ValueError("peak_position must contain two indices")
    row, column = peak_position
    if not 0 <= row < correlation.shape[0] or not 0 <= column < correlation.shape[1]:
        raise ValueError("peak_position is outside the correlation map")
    return int(row), int(column)


def signed_pce(
    correlation_map_value: np.ndarray,
    *,
    neighborhood_radius: int = 2,
    peak_position: tuple[int, int] | None = None,
    peak_mode: Literal["max", "absolute"] = "max",
) -> PceResult:
    """Compute signed PCE with a true excluded-neighborhood denominator.

    The square neighborhood contains ``(2 * radius + 1)^2`` samples and wraps
    at the circular map boundary. Floor energy is the mean squared correlation
    over samples outside that neighborhood, matching the mathematical
    denominator ``mn - |N|`` rather than averaging inserted zeros.
    """

    correlation = _matrix(correlation_map_value, "correlation_map")
    if not isinstance(neighborhood_radius, int) or neighborhood_radius < 0:
        raise ValueError("neighborhood_radius must be a non-negative integer")
    peak = _peak_position(correlation, peak_position, peak_mode)
    excluded = np.zeros(correlation.shape, dtype=bool)
    rows = (np.arange(-neighborhood_radius, neighborhood_radius + 1) + peak[0]) % correlation.shape[
        0
    ]
    columns = (
        np.arange(-neighborhood_radius, neighborhood_radius + 1) + peak[1]
    ) % correlation.shape[1]
    excluded[np.ix_(rows, columns)] = True
    floor = correlation[~excluded]
    if floor.size == 0:
        raise DegenerateStatisticError("PCE neighborhood excludes the complete map")
    floor_energy = float(np.mean(np.square(floor), dtype=np.float64))
    if floor_energy == 0.0:
        raise DegenerateStatisticError("PCE is undefined for zero floor energy")
    peak_value = float(correlation[peak])
    value = float(np.sign(peak_value) * np.square(peak_value) / floor_energy)
    return PceResult(
        peak=peak,
        peak_value=peak_value,
        floor_energy=floor_energy,
        value=value,
        excluded_samples=int(np.count_nonzero(excluded)),
    )


def prnu_python_signed_pce(
    correlation_map_value: np.ndarray,
    *,
    neighborhood_radius: int = 2,
) -> PceResult:
    """Reproduce the locked ``prnu-python`` PCE convention exactly.

    This compatibility path intentionally preserves the upstream half-open
    ``[peak-radius:peak+radius]`` slice and zero-filled full-map energy. Use
    :func:`signed_pce` for new experiments; this function exists only for
    numerical cross-validation and historical comparability.
    """

    correlation = _matrix(correlation_map_value, "correlation_map")
    if not isinstance(neighborhood_radius, int) or neighborhood_radius < 0:
        raise ValueError("neighborhood_radius must be a non-negative integer")
    peak = _peak_position(correlation, None, "max")
    without_peak = correlation.copy()
    without_peak[
        peak[0] - neighborhood_radius : peak[0] + neighborhood_radius,
        peak[1] - neighborhood_radius : peak[1] + neighborhood_radius,
    ] = 0
    floor_energy = float(np.mean(np.square(without_peak), dtype=np.float64))
    if floor_energy == 0.0:
        raise DegenerateStatisticError("PCE is undefined for zero floor energy")
    peak_value = float(correlation[peak])
    value = float(np.sign(peak_value) * np.square(peak_value) / floor_energy)
    return PceResult(
        peak=peak,
        peak_value=peak_value,
        floor_energy=floor_energy,
        value=value,
        excluded_samples=int(np.count_nonzero(correlation != without_peak)),
    )


def _score(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def fits(candidate_score: float, control_score: float, *, stabilizer: float = 0.0) -> float:
    """Return the original single-control FITS division."""

    candidate = _score(candidate_score, "candidate_score")
    control = _score(control_score, "control_score")
    constant = _score(stabilizer, "stabilizer")
    if constant < 0:
        raise ValueError("stabilizer must be non-negative")
    denominator = control + constant
    if denominator == 0.0:
        raise DegenerateStatisticError("FITS denominator is zero")
    return (candidate + constant) / denominator


def fits_plus(
    candidate_score: float,
    control_scores: np.ndarray,
    *,
    stabilizer: float = 0.0,
) -> float:
    """Return the G-FITS multi-control extension using a median denominator."""

    controls = np.asarray(control_scores, dtype=np.float64)
    if controls.ndim != 1 or controls.size == 0:
        raise ValueError("control_scores must be a non-empty one-dimensional array")
    if not np.isfinite(controls).all():
        raise ValueError("control_scores contains a non-finite value")
    return fits(candidate_score, float(np.median(controls)), stabilizer=stabilizer)


def log_ratio(candidate_score: float, control_score: float, *, stabilizer: float = 0.0) -> float:
    """Return the natural logarithm of a stabilized candidate/control ratio."""

    candidate = _score(candidate_score, "candidate_score")
    control = _score(control_score, "control_score")
    constant = _score(stabilizer, "stabilizer")
    if constant < 0:
        raise ValueError("stabilizer must be non-negative")
    numerator = candidate + constant
    denominator = control + constant
    if numerator <= 0.0 or denominator <= 0.0:
        raise DegenerateStatisticError("log-ratio requires positive stabilized scores")
    return float(np.log(numerator) - np.log(denominator))
