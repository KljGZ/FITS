from __future__ import annotations

import numpy as np
import pytest

from gfits.explicit_scoring import (
    fit_nuisance_subspace,
    gallery_complement_ratio,
    median_control_ratio,
    nuisance_subspace_residual_score,
    paper_fits_plus_c,
    paper_fits_ratio,
    robust_zscore,
)
from gfits.matching import (
    DegenerateStatisticError,
    aligned_peak_position,
    cross_correlation,
    cross_correlation_map,
    fits,
    fits_plus,
    log_ratio,
    normalized_cross_correlation,
    prnu_python_signed_pce,
    signed_pce,
)


def test_aligned_cc_and_ncc_are_zero_mean() -> None:
    first = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    second = first + 100.0

    assert cross_correlation(first, second) == pytest.approx(5.0)
    assert normalized_cross_correlation(first, second) == pytest.approx(1.0)
    assert normalized_cross_correlation(first, -second) == pytest.approx(-1.0)


def test_correlation_map_has_aligned_peak_without_mutation() -> None:
    rng = np.random.default_rng(7)
    first = rng.normal(size=(11, 13))
    original = first.copy()

    correlation = cross_correlation_map(first, first)

    assert np.array_equal(first, original)
    assert np.unravel_index(np.argmax(correlation), correlation.shape) == aligned_peak_position(
        correlation.shape
    )


def test_primary_signed_pce_excludes_true_circular_neighborhood() -> None:
    correlation = np.arange(81, dtype=np.float64).reshape(9, 9) / 100.0
    correlation[4, 4] = 5.0
    result = signed_pce(correlation, neighborhood_radius=1, peak_position=(4, 4))
    mask = np.zeros_like(correlation, dtype=bool)
    mask[3:6, 3:6] = True
    expected_energy = np.mean(np.square(correlation[~mask]))

    assert result.peak == (4, 4)
    assert result.excluded_samples == 9
    assert result.floor_energy == pytest.approx(expected_energy)
    assert result.value == pytest.approx(25.0 / expected_energy)


def test_signed_pce_preserves_explicit_negative_peak_sign() -> None:
    correlation = np.linspace(-0.2, 0.2, 121).reshape(11, 11)
    correlation[5, 5] = -4.0

    result = signed_pce(correlation, neighborhood_radius=1, peak_position=(5, 5))

    assert result.value < 0.0
    assert result.peak_value == -4.0


def test_prnu_compatibility_pce_uses_zero_filled_half_open_slice() -> None:
    correlation = np.arange(225, dtype=np.float64).reshape(15, 15) / 100.0
    correlation[8, 9] = 10.0
    result = prnu_python_signed_pce(correlation, neighborhood_radius=2)
    legacy_floor = correlation.copy()
    legacy_floor[6:10, 7:11] = 0.0
    expected_energy = np.mean(np.square(legacy_floor))

    assert result.peak == (8, 9)
    assert result.floor_energy == pytest.approx(expected_energy)
    assert result.value == pytest.approx(100.0 / expected_energy)


def test_fits_family_uses_explicit_control_definition() -> None:
    controls = np.asarray([1.5, 2.0, 10.0])

    assert paper_fits_ratio(4.0, 2.0) == pytest.approx(2.0)
    assert paper_fits_plus_c(4.0, 2.0, constant=1.0) == pytest.approx(5.0 / 3.0)
    assert median_control_ratio(4.0, controls) == pytest.approx(2.0)
    assert gallery_complement_ratio(4.0, controls, reducer="mean") == pytest.approx(4.0 / 4.5)
    assert gallery_complement_ratio(4.0, controls, reducer="median") == pytest.approx(2.0)

    # Historical aliases remain reproducible for the frozen E00--E02 artifacts.
    assert fits(4.0, 2.0) == pytest.approx(2.0)
    assert fits_plus(4.0, controls) == pytest.approx(2.0)
    assert log_ratio(4.0, 2.0) == pytest.approx(np.log(2.0))
    assert fits(4.0, 2.0, stabilizer=1.0) == pytest.approx(5.0 / 3.0)


def test_degenerate_statistics_fail_closed() -> None:
    with pytest.raises(DegenerateStatisticError):
        normalized_cross_correlation(np.ones((3, 3)), np.ones((3, 3)))
    with pytest.raises(DegenerateStatisticError):
        paper_fits_ratio(1.0, 0.0)
    with pytest.raises(DegenerateStatisticError):
        paper_fits_plus_c(1.0, 0.0, constant=0.0)
    with pytest.raises(DegenerateStatisticError):
        log_ratio(-1.0, 2.0)


def test_robust_and_subspace_controls_are_explicit() -> None:
    assert robust_zscore(7.0, np.asarray([1.0, 2.0, 3.0])) > 0.0
    controls = np.asarray([[1.0, 0.0, 2.0], [2.0, 0.0, 1.0], [3.0, 0.0, 0.0], [4.0, 0.0, -1.0]])
    center, basis = fit_nuisance_subspace(controls, rank=1)
    assert basis.shape == (1, 3)
    score = nuisance_subspace_residual_score(np.asarray([2.0, 5.0, 1.0]), center, basis)
    assert score == pytest.approx(25.0)
