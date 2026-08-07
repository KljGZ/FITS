"""Residual, fixed-fingerprint, and statistical-signature primitives for E02b."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import convolve, zoom
from scipy.stats import chi2, linregress, trim_mean

from gfits.matching import DegenerateStatisticError, normalized_cross_correlation

_LUMINANCE = np.asarray([0.29893602, 0.58704307, 0.11402090], dtype=np.float32)
_HIGH_PASS_KERNELS = np.asarray(
    [
        [[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]],
        [[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]],
    ],
    dtype=np.float32,
)
_HIGH_PASS_KERNELS[2] /= 4.0


def luminance(image: np.ndarray) -> np.ndarray:
    """Return normalized BT.601-like luminance without changing image geometry."""

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("luminance requires an HxWx3 uint8 image")
    return np.tensordot(array.astype(np.float32), _LUMINANCE, axes=([-1], [0])) / 255.0


def fixed_three_kernel_high_pass_residual_bank(image: np.ndarray) -> np.ndarray:
    """Return the explicitly limited three-kernel bank (not full SRM)."""

    gray = luminance(image)
    return np.stack(
        [convolve(gray, kernel, mode="reflect") for kernel in _HIGH_PASS_KERNELS],
        axis=-1,
    ).astype(np.float32)


def low_bit_residual(image: np.ndarray, bit_count: int = 2) -> np.ndarray:
    """Return centered RGB low-bit planes as an exporter-sensitive baseline."""

    if not 1 <= int(bit_count) <= 8:
        raise ValueError("bit_count must be between 1 and 8")
    mask = (1 << int(bit_count)) - 1
    return (np.bitwise_and(image, mask).astype(np.float32) / float(mask) - 0.5).astype(np.float32)


def _channels(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim == 2:
        value = value[..., np.newaxis]
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError("residual must be a finite HxW or HxWxC array")
    return value


def _ncc_any(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"NCC shapes differ: {left.shape} and {right.shape}")
    return normalized_cross_correlation(left.reshape(1, -1), right.reshape(1, -1))


def global_ncc(first: np.ndarray, second: np.ndarray) -> float:
    """Compute one NCC after flattening every spatial/channel element."""

    return _ncc_any(first, second)


def channel_mean_ncc(first: np.ndarray, second: np.ndarray) -> float:
    """Average independently centered NCC values over channels."""

    left = _channels(first)
    right = _channels(second)
    if left.shape != right.shape:
        raise ValueError("channel-wise NCC requires equal residual shapes")
    values = [_ncc_any(left[..., index], right[..., index]) for index in range(left.shape[-1])]
    return float(np.mean(values))


@dataclass(frozen=True)
class ChannelWhitening:
    """Source-independent channel scaling fitted on calibration data only."""

    split: str
    means: tuple[float, ...]
    scales: tuple[float, ...]
    sample_count: int

    def transform(self, value: np.ndarray) -> np.ndarray:
        channels = _channels(value)
        if channels.shape[-1] != len(self.means):
            raise ValueError("whitening channel count does not match residual")
        mean = np.asarray(self.means, dtype=np.float64).reshape(1, 1, -1)
        scale = np.asarray(self.scales, dtype=np.float64).reshape(1, 1, -1)
        return ((channels - mean) / scale).astype(np.float32)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_channel_whitening(
    calibration_residuals: Sequence[np.ndarray],
    *,
    split: str,
) -> ChannelWhitening:
    """Fit pooled channel parameters; formal use requires ``split=calibration``."""

    if split != "calibration":
        raise ValueError("E02b channel whitening may only be fit on calibration data")
    if not calibration_residuals:
        raise ValueError("channel whitening requires calibration residuals")
    arrays = [_channels(value) for value in calibration_residuals]
    channel_count = arrays[0].shape[-1]
    if any(value.shape[-1] != channel_count for value in arrays):
        raise ValueError("calibration residual channel counts differ")
    sums = np.zeros(channel_count, dtype=np.float64)
    squares = np.zeros(channel_count, dtype=np.float64)
    count = 0
    for value in arrays:
        flat = value.reshape(-1, channel_count)
        sums += np.sum(flat, axis=0)
        squares += np.sum(np.square(flat), axis=0)
        count += flat.shape[0]
    means = sums / count
    variances = np.maximum(squares / count - np.square(means), 1.0e-12)
    return ChannelWhitening(
        split=split,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in np.sqrt(variances)),
        sample_count=len(arrays),
    )


def whitened_global_ncc(
    first: np.ndarray,
    second: np.ndarray,
    whitening: ChannelWhitening,
) -> float:
    """Compute global NCC after calibration-only channel whitening."""

    return global_ncc(whitening.transform(first), whitening.transform(second))


def aggregate_template(
    residuals: np.ndarray,
    intensities: np.ndarray | None,
    name: str,
    *,
    extractor: str,
    trimmed_fraction: float = 0.1,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    """Aggregate residuals with names that expose physical assumptions."""

    values = np.asarray(residuals, dtype=np.float32)
    if values.ndim < 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("residuals must be a non-empty finite stack")
    if name == "mean":
        result = np.mean(values, axis=0, dtype=np.float64)
    elif name == "median":
        result = np.median(values, axis=0)
    elif name == "trimmed_mean_10":
        result = trim_mean(values, proportiontocut=trimmed_fraction, axis=0)
    elif name == "standardized_mean":
        flattened = values.reshape(values.shape[0], -1, values.shape[-1] if values.ndim == 4 else 1)
        means = np.mean(flattened, axis=1, keepdims=True)
        scales = np.std(flattened, axis=1, keepdims=True)
        standardized = (flattened - means) / np.maximum(scales, epsilon)
        result = np.mean(standardized.reshape(values.shape), axis=0, dtype=np.float64)
    elif name == "spectral_whitened_mean":
        channels = values[..., np.newaxis] if values.ndim == 3 else values
        whitened = []
        for sample in channels:
            transformed = np.fft.fft2(sample, axes=(0, 1))
            normalized = transformed / np.maximum(np.abs(transformed), epsilon)
            whitened.append(np.real(np.fft.ifft2(normalized, axes=(0, 1))))
        result = np.mean(np.asarray(whitened), axis=0, dtype=np.float64)
        if values.ndim == 3:
            result = result[..., 0]
    elif name in {"prnu_mle", "intensity_weighted"}:
        if name == "prnu_mle" and extractor != "wavelet":
            raise ValueError("prnu_mle is reserved for the wavelet PRNU residual")
        if name == "intensity_weighted" and extractor == "wavelet":
            raise ValueError("wavelet multiplicative aggregation must be named prnu_mle")
        if intensities is None:
            raise ValueError(f"{name} requires aligned intensity maps")
        weights = np.asarray(intensities, dtype=np.float32)
        if values.ndim == 4:
            weights = weights[..., np.newaxis]
        numerator = np.sum(values * weights, axis=0, dtype=np.float64)
        denominator = np.sum(np.square(weights), axis=0, dtype=np.float64)
        result = numerator / (denominator + epsilon)
    else:
        raise ValueError(f"unsupported E02b aggregator: {name}")
    return np.asarray(result, dtype=np.float32)


def _normalized_grid(value: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    factors = (size / array.shape[0], size / array.shape[1]) + (1.0,) * (array.ndim - 2)
    return zoom(array, factors, order=1, mode="nearest", prefilter=False).astype(np.float32)


def _fft_channels(residual: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(_channels(residual), axes=(0, 1)), axes=(0, 1))


def power_spectrum_2d(residual: np.ndarray, *, grid_size: int = 128) -> np.ndarray:
    """Return a normalized-grid log 2-D power signature."""

    power = np.mean(np.square(np.abs(_fft_channels(residual))), axis=-1)
    return _normalized_grid(np.log1p(power), grid_size)


def phase_spectrum(residual: np.ndarray, *, grid_size: int = 128) -> np.ndarray:
    """Encode circular phase as cosine/sine channels on a normalized grid."""

    phase = np.angle(_fft_channels(residual))
    encoded = np.concatenate(
        (
            np.mean(np.cos(phase), axis=-1, keepdims=True),
            np.mean(np.sin(phase), axis=-1, keepdims=True),
        ),
        axis=-1,
    )
    return _normalized_grid(encoded, grid_size)


def radial_power_spectrum(residual: np.ndarray, *, bins: int = 128) -> np.ndarray:
    """Return radial means on normalized spatial-frequency radius."""

    power = np.mean(np.square(np.abs(_fft_channels(residual))), axis=-1)
    height, width = power.shape
    yy, xx = np.indices(power.shape)
    radius = np.sqrt(
        ((yy - (height - 1) / 2) / max(height, 1)) ** 2
        + ((xx - (width - 1) / 2) / max(width, 1)) ** 2
    )
    edges = np.linspace(0.0, np.sqrt(0.5), bins + 1)
    indices = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, bins - 1)
    sums = np.bincount(indices, weights=np.log1p(power).ravel(), minlength=bins)
    counts = np.maximum(np.bincount(indices, minlength=bins), 1)
    return (sums / counts).astype(np.float32)


def autocorrelation_signature(residual: np.ndarray, *, grid_size: int = 128) -> np.ndarray:
    """Return the mean Wiener--Khinchin autocorrelation surface."""

    transformed = np.fft.fft2(_channels(residual), axes=(0, 1))
    correlation = np.fft.fftshift(
        np.real(np.fft.ifft2(np.square(np.abs(transformed)), axes=(0, 1))), axes=(0, 1)
    )
    return _normalized_grid(np.mean(correlation, axis=-1), grid_size)


def cepstrum_signature(residual: np.ndarray, *, grid_size: int = 128) -> np.ndarray:
    """Return a real cepstral surface averaged across channels."""

    transformed = np.fft.fft2(_channels(residual), axes=(0, 1))
    cepstrum = np.fft.fftshift(
        np.real(np.fft.ifft2(np.log(np.abs(transformed) + 1.0e-8), axes=(0, 1))), axes=(0, 1)
    )
    return _normalized_grid(np.mean(cepstrum, axis=-1), grid_size)


def srm_style_cooccurrence(residual: np.ndarray, *, levels: int = 8) -> np.ndarray:
    """Return a compact SRM-style co-occurrence, not a full SRM implementation."""

    channels = _channels(residual)
    features = []
    for index in range(channels.shape[-1]):
        value = channels[..., index]
        median = np.median(value)
        scale = 1.4826 * np.median(np.abs(value - median)) + 1.0e-8
        quantized = np.clip(
            np.floor((np.clip((value - median) / scale, -3, 3) + 3) / 6 * levels), 0, levels - 1
        ).astype(int)
        matrix = np.zeros((levels, levels), dtype=np.float64)
        for left, right in (
            (quantized[:, :-1], quantized[:, 1:]),
            (quantized[:-1, :], quantized[1:, :]),
        ):
            np.add.at(matrix, (left.ravel(), right.ravel()), 1)
        matrix /= max(float(np.sum(matrix)), 1.0)
        features.append(matrix.ravel())
    return np.concatenate(features).astype(np.float32)


def patch_gram_covariance(residual: np.ndarray, *, patch_size: int = 32) -> np.ndarray:
    """Return covariance/Gram statistics of patch-level residual summaries."""

    channels = _channels(residual)
    height = channels.shape[0] - channels.shape[0] % patch_size
    width = channels.shape[1] - channels.shape[1] % patch_size
    if height < patch_size or width < patch_size:
        raise ValueError("residual is smaller than the registered patch size")
    rows = []
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            patch = channels[top : top + patch_size, left : left + patch_size]
            horizontal = np.diff(patch, axis=1)
            vertical = np.diff(patch, axis=0)
            rows.append(
                np.concatenate(
                    (
                        np.mean(patch, axis=(0, 1)),
                        np.std(patch, axis=(0, 1)),
                        np.mean(np.square(horizontal), axis=(0, 1)),
                        np.mean(np.square(vertical), axis=(0, 1)),
                    )
                )
            )
    matrix = np.asarray(rows, dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    covariance = np.cov(matrix, rowvar=False, bias=True)
    gram = matrix.T @ matrix / matrix.shape[0]
    return np.concatenate((mean, covariance.ravel(), gram.ravel())).astype(np.float32)


def low_high_frequency_partition(residual: np.ndarray) -> np.ndarray:
    """Return per-channel normalized low/high frequency energy statistics."""

    transformed = _fft_channels(residual)
    power = np.square(np.abs(transformed))
    height, width, _ = power.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt(
        ((yy - (height - 1) / 2) / max(height, 1)) ** 2
        + ((xx - (width - 1) / 2) / max(width, 1)) ** 2
    )
    low = radius <= 0.25
    features = []
    for channel in range(power.shape[-1]):
        values = power[..., channel]
        total = float(np.sum(values)) + 1.0e-12
        for mask in (low, ~low):
            selected = np.log1p(values[mask])
            features.extend(
                (
                    float(np.sum(values[mask]) / total),
                    float(np.mean(selected)),
                    float(np.std(selected)),
                    float(np.quantile(selected, 0.5)),
                    float(np.quantile(selected, 0.9)),
                )
            )
    return np.asarray(features, dtype=np.float32)


def statistical_signature(name: str, residual: np.ndarray, *, grid_size: int = 128) -> np.ndarray:
    """Dispatch one explicitly named statistical-signature representation."""

    functions = {
        "power_spectrum_2d": lambda value: power_spectrum_2d(value, grid_size=grid_size),
        "phase_spectrum": lambda value: phase_spectrum(value, grid_size=grid_size),
        "radial_power_spectrum": radial_power_spectrum,
        "autocorrelation": lambda value: autocorrelation_signature(value, grid_size=grid_size),
        "cepstrum": lambda value: cepstrum_signature(value, grid_size=grid_size),
        "srm_style_cooccurrence": srm_style_cooccurrence,
        "patch_gram_covariance": patch_gram_covariance,
        "low_high_frequency_partition": low_high_frequency_partition,
    }
    if name not in functions:
        raise ValueError(f"unsupported statistical signature: {name}")
    result = np.asarray(functions[name](residual), dtype=np.float32)
    if not np.isfinite(result).all() or result.size == 0:
        raise DegenerateStatisticError(f"invalid statistical signature: {name}")
    return result


def intensity_multiplicativity_diagnostics(
    residuals: Sequence[np.ndarray],
    intensities: Sequence[np.ndarray],
    *,
    bins: int = 16,
    max_points: int = 1_000_000,
    seed: int = 2026080711,
) -> dict[str, Any]:
    """Test intensity-linked variance using calibration residuals only."""

    if not residuals or len(residuals) != len(intensities):
        raise ValueError("aligned residual and intensity calibration samples are required")
    x_values = []
    y_values = []
    for residual, intensity in zip(residuals, intensities, strict=True):
        value = _channels(residual)
        image = np.asarray(intensity, dtype=np.float64)
        if image.shape != value.shape[:2]:
            raise ValueError("intensity and residual geometry differ")
        x_values.append(np.repeat(image.ravel(), value.shape[-1]))
        y_values.append(np.square(value).ravel())
    x = np.concatenate(x_values)
    y = np.concatenate(y_values)
    if x.size > max_points:
        rng = np.random.default_rng(seed)
        selected = rng.choice(x.size, size=max_points, replace=False)
        x = x[selected]
        y = y[selected]
    regression = linregress(x, y)
    fitted = regression.intercept + regression.slope * x
    errors = y - fitted
    auxiliary = linregress(x, np.square(errors))
    bp_statistic = float(x.size * auxiliary.rvalue**2)
    bp_p = float(chi2.sf(bp_statistic, 1))
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_rows = []
    for index in range(bins):
        mask = (x >= edges[index]) & (
            x < edges[index + 1] if index + 1 < bins else x <= edges[index + 1]
        )
        bin_rows.append(
            {
                "bin": index,
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "count": int(np.count_nonzero(mask)),
                "residual_energy_mean": float(np.mean(y[mask])) if np.any(mask) else None,
                "residual_energy_variance": float(np.var(y[mask])) if np.any(mask) else None,
            }
        )
    return {
        "sample_count": len(residuals),
        "point_count": int(x.size),
        "regression": {
            "slope": float(regression.slope),
            "intercept": float(regression.intercept),
            "rvalue": float(regression.rvalue),
            "pvalue": float(regression.pvalue),
            "stderr": float(regression.stderr),
        },
        "breusch_pagan": {"statistic": bp_statistic, "df": 1, "pvalue": bp_p},
        "bins": bin_rows,
    }


def multiplicative_modulation_supported(
    diagnostics: Mapping[str, Any], settings: Mapping[str, Any]
) -> bool:
    """Apply the pre-registered calibration-only modulation gate."""

    rules = settings["modulation_requires_all"]
    regression = diagnostics["regression"]
    bp = diagnostics["breusch_pagan"]
    return (
        (not rules["slope_positive"] or float(regression["slope"]) > 0.0)
        and float(regression["pvalue"]) < float(rules["regression_p_below"])
        and float(bp["pvalue"]) < float(rules["breusch_pagan_p_below"])
    )
