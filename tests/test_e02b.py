from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gfits.controls import (
    NuisanceControlBank,
    assert_candidate_independent_denominator,
    calibrate_candidates,
)
from gfits.e02b_data import (
    _counterfactual_rgb,
    build_generation_plan,
    canonical_png_bytes,
    canonical_quantize,
    controlled_prompts,
    load_e02b_config,
)
from gfits.e02b_features import (
    aggregate_template,
    channel_mean_ncc,
    fit_channel_whitening,
    fixed_three_kernel_high_pass_residual_bank,
    global_ncc,
    intensity_multiplicativity_diagnostics,
    low_bit_residual,
    statistical_signature,
    whitened_global_ncc,
)
from gfits.e02b_generation import _snapshot_allow_patterns
from gfits.e02b_statistics import (
    attribution_metrics,
    auxiliary_sign_flip_test,
    e02b_gate,
    hierarchical_family_prompt_permutation,
    prompt_block_source_label_permutation,
    template_source_label_permutation,
    two_way_bootstrap,
)
from gfits.explicit_scoring import gallery_complement_ratio

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "e02b.yaml"
TRACEABILITY = ROOT / "reports" / "e02b" / "PRO_AUDIT_TRACEABILITY.md"


def test_e02b_full_factorial_and_split_isolation() -> None:
    config = load_e02b_config(CONFIG)
    prompts = controlled_prompts(config)
    plan = build_generation_plan(config)

    assert len(prompts) == 200
    assert len({row["prompt"] for row in prompts}) == 200
    assert len(plan) == 7 * 200 * 4 * 4
    prompt_splits = {
        split: {row["prompt_id"] for row in prompts if row["split"] == split}
        for split in ("template", "calibration", "test")
    }
    seed_splits = {key: set(value) for key, value in config["design"]["seeds"].items()}
    assert not prompt_splits["template"] & prompt_splits["calibration"]
    assert not prompt_splits["template"] & prompt_splits["test"]
    assert not prompt_splits["calibration"] & prompt_splits["test"]
    assert not seed_splits["template"] & seed_splits["calibration"]
    assert not seed_splits["template"] & seed_splits["test"]
    assert not seed_splits["calibration"] & seed_splits["test"]
    assert {row["native_resolution"] for row in plan} == {256, 512, 768, 1024}
    assert set(config["model_groups"]) == {"cross_family", "near_family"}
    assert config["design"]["generation_batch_size"] == 1


def test_model_snapshot_policy_avoids_duplicate_weight_formats() -> None:
    config = load_e02b_config(CONFIG)
    sd_patterns = _snapshot_allow_patterns(config["models"]["sd14"])
    tiny_patterns = _snapshot_allow_patterns(config["models"]["tiny_sd"])
    assert "**/*.fp16.safetensors" in sd_patterns
    assert not any(pattern.endswith(".bin") for pattern in sd_patterns)
    assert "**/*.bin" in tiny_patterns
    assert not any(pattern.endswith(".safetensors") for pattern in tiny_patterns)
    for model in config["models"].values():
        if model["adapter"] == "stable_diffusion":
            assert model["scheduler_parameters"] == {
                "algorithm_type": "dpmsolver++",
                "solver_order": 2,
                "final_sigmas_type": "zero",
            }


def test_pro_audit_traceability_is_exactly_one_to_one() -> None:
    identifiers = []
    for line in TRACEABILITY.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) > 2 and parts[1].strip().isdigit():
            identifiers.append(int(parts[1].strip()))
    assert identifiers == list(range(1, 74))


def test_canonical_writer_and_counterfactual_pixel_contracts() -> None:
    floating = np.linspace(-0.1, 1.1, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    rgb, metadata = canonical_quantize(floating)
    writer = load_e02b_config(CONFIG)["design"]["writer"]
    first = canonical_png_bytes(rgb, writer)
    second = canonical_png_bytes(rgb, writer)

    assert first == second
    assert metadata["tensor_dtype"] == "float32"
    assert metadata["quantization_method"] == "clip_0_1_then_rint_x255_to_uint8"
    with Image.open(io.BytesIO(first)) as image:
        np.testing.assert_array_equal(np.asarray(image), rgb)
    identity = _counterfactual_rgb(rgb, {"kind": "canonical_requantize"}, sample_id="sample")
    reencode = _counterfactual_rgb(rgb, {"kind": "png_reencode"}, sample_id="sample")
    jitter = _counterfactual_rgb(rgb, {"kind": "lsb_jitter", "seed": 17}, sample_id="sample")
    reduced = _counterfactual_rgb(rgb, {"kind": "bit_depth", "bits": 6}, sample_id="sample")
    np.testing.assert_array_equal(identity, rgb)
    np.testing.assert_array_equal(reencode, rgb)
    assert np.count_nonzero(jitter != rgb) > 0
    assert np.count_nonzero(reduced != rgb) > 0
    np.testing.assert_array_equal(
        jitter,
        _counterfactual_rgb(rgb, {"kind": "lsb_jitter", "seed": 17}, sample_id="sample"),
    )


def test_nuisance_control_bank_is_independent_and_candidate_invariant() -> None:
    bank = NuisanceControlBank(
        pipeline_id="jpeg95",
        geometry_id="512x512",
        donor_source_ids=("z1", "z2", "z3"),
        candidate_source_ids=("g1", "g2"),
        donor_image_sha256=("a", "b", "c"),
    )
    bank.validate_query(
        query_source_id="q",
        query_image_sha256="query",
        pipeline_id="jpeg95",
        geometry_id="512x512",
    )
    denominator = bank.denominator({"z1": 1.0, "z2": 2.0, "z3": 9.0})
    calibrated = calibrate_candidates({"g1": 4.0, "g2": 1.0}, denominator, constant=1.0)
    assert denominator == 2.0
    assert calibrated == pytest.approx({"g1": 5.0 / 3.0, "g2": 2.0 / 3.0})
    assert_candidate_independent_denominator(
        [
            {"query_id": "q", "candidate_source_id": "g1", "nuisance_denominator": 2.0},
            {"query_id": "q", "candidate_source_id": "g2", "nuisance_denominator": 2.0},
        ]
    )
    with pytest.raises(ValueError):
        NuisanceControlBank(
            pipeline_id="p",
            geometry_id="g",
            donor_source_ids=("g1", "z2", "z3"),
            candidate_source_ids=("g1", "g2"),
        )
    with pytest.raises(ValueError):
        bank.validate_query(
            query_source_id="z1",
            query_image_sha256="query",
            pipeline_id="jpeg95",
            geometry_id="512x512",
        )
    with pytest.raises(ValueError):
        assert_candidate_independent_denominator(
            [
                {"query_id": "q", "nuisance_denominator": 1.0},
                {"query_id": "q", "nuisance_denominator": 2.0},
            ]
        )


@pytest.mark.parametrize("candidate_count", [3, 4, 8])
def test_gallery_complement_is_top1_monotone(candidate_count: int) -> None:
    rng = np.random.default_rng(candidate_count)
    for _ in range(50):
        energies = rng.uniform(0.01, 10.0, size=candidate_count)
        ratios = np.asarray(
            [
                gallery_complement_ratio(
                    value, np.delete(energies, index), reducer="mean", constant=0.0
                )
                for index, value in enumerate(energies)
            ]
        )
        assert int(np.argmax(ratios)) == int(np.argmax(energies))


def test_residual_names_aggregators_and_channel_scores_are_explicit() -> None:
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    high_pass = fixed_three_kernel_high_pass_residual_bank(image)
    low_bit = low_bit_residual(image)
    assert high_pass.shape == (16, 16, 3)
    assert low_bit.shape == image.shape
    stack = np.stack((high_pass, high_pass * 2, high_pass * 3))
    intensity = np.ones((3, 16, 16), dtype=np.float32)
    assert (
        aggregate_template(
            stack, None, "median", extractor="fixed_three_kernel_high_pass_residual_bank"
        ).shape
        == high_pass.shape
    )
    assert (
        aggregate_template(
            stack,
            intensity,
            "intensity_weighted",
            extractor="fixed_three_kernel_high_pass_residual_bank",
        ).shape
        == high_pass.shape
    )
    with pytest.raises(ValueError):
        aggregate_template(
            stack,
            intensity,
            "prnu_mle",
            extractor="fixed_three_kernel_high_pass_residual_bank",
        )
    first = np.stack((np.arange(16).reshape(4, 4), np.arange(16).reshape(4, 4) * 100), axis=-1)
    second = np.stack((first[..., 0], -first[..., 1]), axis=-1)
    assert global_ncc(first, second) != pytest.approx(channel_mean_ncc(first, second))


def test_channel_whitening_is_calibration_only() -> None:
    calibration = [np.ones((4, 4, 2)), np.dstack((np.ones((4, 4)) * 3, np.ones((4, 4)) * 7))]
    whitening = fit_channel_whitening(calibration, split="calibration")
    assert whitening.split == "calibration"
    assert whitening.sample_count == 2
    value = np.dstack((np.arange(16).reshape(4, 4), np.arange(16).reshape(4, 4) * 2))
    assert np.isfinite(whitened_global_ncc(value, value + 1, whitening))
    with pytest.raises(ValueError):
        fit_channel_whitening(calibration, split="test")


def test_vector_signatures_support_mean_and_median_aggregation() -> None:
    vector_stack = np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    median = aggregate_template(
        vector_stack,
        None,
        "median",
        extractor="fixed_three_kernel_high_pass_residual_bank",
    )
    assert np.allclose(median, [2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "name",
    [
        "power_spectrum_2d",
        "phase_spectrum",
        "radial_power_spectrum",
        "autocorrelation",
        "cepstrum",
        "srm_style_cooccurrence",
        "patch_gram_covariance",
        "low_high_frequency_partition",
    ],
)
def test_all_statistical_signatures_are_finite_and_named(name: str) -> None:
    rng = np.random.default_rng(9)
    residual = rng.normal(size=(64, 64, 3)).astype(np.float32)
    signature = statistical_signature(name, residual, grid_size=32)
    assert signature.size > 0
    assert signature.dtype == np.float32
    assert np.isfinite(signature).all()


def test_multiplicativity_diagnostics_detect_intensity_link() -> None:
    rng = np.random.default_rng(12)
    residuals = []
    intensities = []
    for _ in range(8):
        intensity = rng.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)
        residual = intensity * rng.normal(0.0, 0.5, size=(64, 64)).astype(np.float32)
        intensities.append(intensity)
        residuals.append(residual)
    result = intensity_multiplicativity_diagnostics(
        residuals, intensities, bins=8, max_points=100_000
    )
    assert result["regression"]["slope"] > 0.0
    assert result["regression"]["pvalue"] < 0.05
    assert result["breusch_pagan"]["pvalue"] < 0.05


def _toy_rows() -> list[dict[str, object]]:
    rows = []
    for prompt in ("p1", "p2", "p3"):
        for source in ("a", "b", "c"):
            query = f"{prompt}-{source}"
            for candidate in ("a", "b", "c"):
                rows.append(
                    {
                        "query_id": query,
                        "prompt_id": prompt,
                        "source_id": source,
                        "candidate_source_id": candidate,
                        "score": 1.0 if source == candidate else 0.1,
                    }
                )
    return rows


def test_blocked_statistics_and_two_way_bootstrap() -> None:
    rows = _toy_rows()
    metrics = attribution_metrics(rows)
    assert metrics["pooled_pairwise_auroc"] == 1.0
    assert metrics["macro_source_auroc"] == 1.0
    assert metrics["rank1"] == 1.0
    prompt_test = prompt_block_source_label_permutation(rows, draws=99, seed=1)
    template_test = template_source_label_permutation(rows, draws=99, seed=2)
    hierarchy_test = hierarchical_family_prompt_permutation(
        rows, {"a": "f", "b": "f", "c": "f"}, draws=99, seed=3
    )
    sign_flip = auxiliary_sign_flip_test(rows, draws=99, seed=4)
    assert prompt_test["kind"] == "prompt_block_source_label_permutation"
    assert prompt_test["rank1_observed"] == 1.0
    assert prompt_test["rank1_pvalue"] <= 0.05
    assert template_test["kind"] == "template_source_label_permutation"
    assert hierarchy_test["kind"] == "hierarchical_family_prompt_permutation"
    assert "assumption" in sign_flip

    def builder(templates: dict[str, list[str]], prompts: list[str]) -> list[dict[str, object]]:
        selected = []
        for occurrence, prompt in enumerate(prompts):
            for row in rows:
                if row["prompt_id"] == prompt:
                    copy = dict(row)
                    copy["query_id"] = f"{row['query_id']}-{occurrence}"
                    selected.append(copy)
        assert all(templates.values())
        return selected

    bootstrap = two_way_bootstrap(
        {"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1", "c2"]},
        ["p1", "p2", "p3"],
        builder,
        draws=20,
        confidence=0.95,
        seed=5,
    )
    assert set(bootstrap["intervals"]) == {"template_only", "query_only", "two_way"}
    assert bootstrap["intervals"]["two_way"]["low"] == 1.0


def test_e02b_gate_reports_each_of_six_checks() -> None:
    suite_result = {
        "bootstrap": {"intervals": {"two_way": {"low": 0.7}}},
        "prompt_block_permutation": {"pvalue": 0.01, "rank1_pvalue": 0.01},
        "holm_confirmatory_pvalues": {"prompt_block_rank1_permutation": 0.01},
        "metrics": {
            "rank1": 0.8,
            "chance_rank1": 0.25,
            "per_source": {source: {"auroc": 0.8} for source in ("a", "b", "c", "d")},
        },
    }
    curves = [
        {"n_template": n, "macro_source_auroc": 0.6 + index * 0.02}
        for index, n in enumerate((1, 3, 5, 10, 20, 50))
    ]
    gate = e02b_gate(
        {"cross_family": suite_result, "near_family": suite_result},
        {"cross_family": curves, "near_family": curves},
        {
            "cross_family": {"bootstrap_ci_low": 0.6},
            "near_family": {"bootstrap_ci_low": 0.6},
        },
        alpha=0.05,
    )
    assert gate["passed"] is True
    assert len(gate["checks"]) == 6
