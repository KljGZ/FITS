"""Blocked inference and source-level attribution metrics for controlled E02b."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _labels_scores(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(
        [str(row["source_id"]) == str(row["candidate_source_id"]) for row in rows],
        dtype=bool,
    )
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    if not labels.any() or labels.all() or not np.isfinite(scores).all():
        raise ValueError("pairwise metric needs finite H1 and H0 scores")
    return labels, scores


def query_margins(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return one match-minus-mean-nonmatch margin per query."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    margins = []
    for query_id in sorted(grouped):
        block = grouped[query_id]
        matched = [
            float(row["score"]) for row in block if row["source_id"] == row["candidate_source_id"]
        ]
        nonmatched = [
            float(row["score"]) for row in block if row["source_id"] != row["candidate_source_id"]
        ]
        if len(matched) != 1 or not nonmatched:
            raise ValueError(f"query {query_id} lacks one-match/full-gallery scores")
        margins.append(matched[0] - float(np.mean(nonmatched)))
    return np.asarray(margins, dtype=np.float64)


def attribution_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute pooled, macro/per-source, ranking, confusion, and bias metrics."""

    labels, scores = _labels_scores(rows)
    sources = sorted({str(row["candidate_source_id"]) for row in rows})
    per_source = {}
    for source in sources:
        selected = [row for row in rows if str(row["candidate_source_id"]) == source]
        source_labels, source_scores = _labels_scores(selected)
        per_source[source] = {
            "auroc": float(roc_auc_score(source_labels, source_scores)),
            "h1_count": int(np.count_nonzero(source_labels)),
            "h0_count": int(np.count_nonzero(~source_labels)),
            "h1_mean": float(np.mean(source_scores[source_labels])),
            "h0_mean": float(np.mean(source_scores[~source_labels])),
            "h0_std": float(np.std(source_scores[~source_labels])),
        }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    reciprocal_ranks = []
    rank1 = 0
    rank5 = 0
    predicted = Counter()
    source_rank1 = Counter()
    source_queries = Counter()
    for query_id in sorted(grouped):
        ranked = sorted(grouped[query_id], key=lambda row: float(row["score"]), reverse=True)
        true_source = str(ranked[0]["source_id"])
        predicted_source = str(ranked[0]["candidate_source_id"])
        predicted[predicted_source] += 1
        confusion[true_source][predicted_source] += 1
        source_queries[true_source] += 1
        true_rank = next(
            index + 1
            for index, row in enumerate(ranked)
            if str(row["candidate_source_id"]) == true_source
        )
        reciprocal_ranks.append(1.0 / true_rank)
        rank1 += int(true_rank == 1)
        rank5 += int(true_rank <= 5)
        source_rank1[true_source] += int(true_rank == 1)
    query_count = len(grouped)
    candidate_bias = {}
    for source in sources:
        h0 = [
            float(row["score"])
            for row in rows
            if str(row["candidate_source_id"]) == source and str(row["source_id"]) != source
        ]
        candidate_bias[source] = {
            "h0_mean": float(np.mean(h0)),
            "h0_std": float(np.std(h0)),
            "predicted_count": int(predicted[source]),
            "predicted_fraction": float(predicted[source] / query_count),
        }
    return {
        "pooled_pairwise_auroc": float(roc_auc_score(labels, scores)),
        "macro_source_auroc": float(np.mean([value["auroc"] for value in per_source.values()])),
        "per_source": per_source,
        "query_count": query_count,
        "chance_rank1": 1.0 / len(sources),
        "rank1": float(rank1 / query_count),
        "rank5": float(rank5 / query_count),
        "mean_average_precision": float(np.mean(reciprocal_ranks)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "per_source_rank1": {
            source: float(source_rank1[source] / source_queries[source]) for source in sources
        },
        "confusion_matrix": {
            actual: {candidate: int(confusion[actual][candidate]) for candidate in sources}
            for actual in sources
        },
        "candidate_template_bias": candidate_bias,
        "paired_margin_mean": float(np.mean(query_margins(rows))),
    }


def _permutation_p(observed: float, null: np.ndarray) -> float:
    return float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))


def auxiliary_sign_flip_test(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    """Retain the historical sign-flip only as an explicitly auxiliary test."""

    margins = query_margins(rows)
    observed = float(np.mean(margins))
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        null[index] = float(np.mean(margins * rng.choice((-1.0, 1.0), size=margins.size)))
    return {
        "kind": "auxiliary_query_margin_sign_flip",
        "assumption": "null margins are symmetric about zero",
        "observed": observed,
        "draws": draws,
        "seed": seed,
        "pvalue": _permutation_p(observed, null),
    }


def _margin_with_labels(
    rows: Sequence[Mapping[str, Any]],
    query_labels: Mapping[str, str],
    template_labels: Mapping[str, str] | None = None,
) -> float:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    values = []
    for query_id, block in grouped.items():
        true_source = query_labels[query_id]
        matched = []
        nonmatched = []
        for row in block:
            candidate = str(row["candidate_source_id"])
            assigned = template_labels[candidate] if template_labels else candidate
            target = matched if assigned == true_source else nonmatched
            target.append(float(row["score"]))
        if len(matched) == 1 and nonmatched:
            values.append(matched[0] - float(np.mean(nonmatched)))
    if not values:
        raise ValueError("permutation produced no valid one-match query blocks")
    return float(np.mean(values))


def prompt_block_source_label_permutation(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    """Permute query labels within prompts for margin and Rank-1 nulls."""

    query_rows: dict[str, Mapping[str, Any]] = {}
    prompt_queries: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        query_id = str(row["query_id"])
        query_rows.setdefault(query_id, row)
    for query_id, row in query_rows.items():
        prompt_queries[str(row["prompt_id"])].append(query_id)
    observed_labels = {query: str(row["source_id"]) for query, row in query_rows.items()}
    observed = _margin_with_labels(rows, observed_labels)
    predicted_labels: dict[str, str] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    for query_id, block in grouped.items():
        predicted_labels[query_id] = str(
            max(block, key=lambda value: float(value["score"]))["candidate_source_id"]
        )

    def rank1(labels: Mapping[str, str]) -> float:
        return float(
            np.mean([predicted_labels[query] == label for query, label in sorted(labels.items())])
        )

    observed_rank1 = rank1(observed_labels)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    rank1_null = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        labels = dict(observed_labels)
        for queries in prompt_queries.values():
            values = [observed_labels[query] for query in queries]
            shuffled = rng.permutation(values)
            labels.update(zip(queries, shuffled, strict=True))
        null[index] = _margin_with_labels(rows, labels)
        rank1_null[index] = rank1(labels)
    return {
        "kind": "prompt_block_source_label_permutation",
        "exchange_unit": "source labels within prompt_id",
        "observed": observed,
        "draws": draws,
        "seed": seed,
        "pvalue": _permutation_p(observed, null),
        "rank1_observed": observed_rank1,
        "rank1_pvalue": _permutation_p(observed_rank1, rank1_null),
    }


def template_source_label_permutation(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    """Permute complete template-source identities and recompute query margins."""

    sources = sorted({str(row["candidate_source_id"]) for row in rows})
    query_labels = {}
    for row in rows:
        query_labels[str(row["query_id"])] = str(row["source_id"])
    observed = _margin_with_labels(rows, query_labels)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        shuffled = rng.permutation(sources)
        mapping = dict(zip(sources, shuffled, strict=True))
        null[index] = _margin_with_labels(rows, query_labels, mapping)
    return {
        "kind": "template_source_label_permutation",
        "exchange_unit": "complete candidate templates",
        "observed": observed,
        "draws": draws,
        "seed": seed,
        "pvalue": _permutation_p(observed, null),
    }


def hierarchical_family_prompt_permutation(
    rows: Sequence[Mapping[str, Any]],
    source_families: Mapping[str, str],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Permute model identities within family while retaining prompt blocks."""

    query_rows: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        query_rows.setdefault(str(row["query_id"]), row)
    query_labels = {query: str(row["source_id"]) for query, row in query_rows.items()}
    prompt_queries: dict[str, list[str]] = defaultdict(list)
    for query, row in query_rows.items():
        prompt_queries[str(row["prompt_id"])].append(query)
    family_sources: dict[str, list[str]] = defaultdict(list)
    for source in sorted(set(query_labels.values())):
        family_sources[str(source_families[source])].append(source)
    observed = _margin_with_labels(rows, query_labels)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        labels = dict(query_labels)
        for queries in prompt_queries.values():
            by_family: dict[str, list[str]] = defaultdict(list)
            for query in queries:
                by_family[str(source_families[query_labels[query]])].append(query)
            for family, family_queries in by_family.items():
                if len(family_sources[family]) > 1:
                    shuffled = rng.permutation([query_labels[value] for value in family_queries])
                    labels.update(zip(family_queries, shuffled, strict=True))
        null[index] = _margin_with_labels(rows, labels)
    return {
        "kind": "hierarchical_family_prompt_permutation",
        "exchange_unit": "model labels within family and prompt_id",
        "families": {key: value for key, value in sorted(family_sources.items())},
        "observed": observed,
        "draws": draws,
        "seed": seed,
        "pvalue": _permutation_p(observed, null),
    }


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Return Holm FWER-adjusted p-values under arbitrary dependence."""

    values = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(values[index]) * (count - rank)))
        adjusted[index] = running
    return [float(value) for value in adjusted]


ScoreBuilder = Callable[[Mapping[str, Sequence[str]], Sequence[str]], Sequence[Mapping[str, Any]]]


def two_way_bootstrap(
    template_image_ids: Mapping[str, Sequence[str]],
    query_prompt_ids: Sequence[str],
    score_builder: ScoreBuilder,
    *,
    draws: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Rebuild templates and resample shared query prompts in each bootstrap draw."""

    if not template_image_ids or not query_prompt_ids or draws <= 0:
        raise ValueError("two-way bootstrap requires template IDs, prompt IDs, and draws")
    rng = np.random.default_rng(seed)
    metrics = {"template_only": [], "query_only": [], "two_way": []}
    fixed_templates = {key: list(values) for key, values in template_image_ids.items()}
    fixed_prompts = list(query_prompt_ids)
    for _ in range(draws):
        sampled_templates = {
            source: list(rng.choice(ids, size=len(ids), replace=True))
            for source, ids in fixed_templates.items()
        }
        sampled_prompts = list(rng.choice(fixed_prompts, size=len(fixed_prompts), replace=True))
        for kind, templates, prompts in (
            ("template_only", sampled_templates, fixed_prompts),
            ("query_only", fixed_templates, sampled_prompts),
            ("two_way", sampled_templates, sampled_prompts),
        ):
            value = attribution_metrics(score_builder(templates, prompts))["macro_source_auroc"]
            metrics[kind].append(value)
    alpha = (1.0 - confidence) / 2.0
    return {
        "kind": "two_way_template_image_and_query_prompt_group",
        "draws": draws,
        "confidence": confidence,
        "seed": seed,
        "template_resampling_unit": "template image within source",
        "query_resampling_unit": "prompt_id across all sources",
        "intervals": {
            kind: {
                "mean": float(np.mean(values)),
                "low": float(np.quantile(values, alpha)),
                "high": float(np.quantile(values, 1.0 - alpha)),
            }
            for kind, values in metrics.items()
        },
    }


def e02b_gate(
    suite_results: Mapping[str, Mapping[str, Any]],
    template_curves: Mapping[str, Sequence[Mapping[str, float]]],
    low_bit_uniform_results: Mapping[str, Mapping[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    """Apply the six pre-registered E02b confirmation requirements."""

    required = {"cross_family", "near_family"}
    suite_present = required <= set(suite_results)
    macro_ci = suite_present and all(
        float(suite_results[suite]["bootstrap"]["intervals"]["two_way"]["low"]) > 0.5
        for suite in required
    )
    rank1 = suite_present and all(
        float(suite_results[suite]["holm_confirmatory_pvalues"]["prompt_block_rank1_permutation"])
        < alpha
        and float(suite_results[suite]["metrics"]["rank1"])
        > float(suite_results[suite]["metrics"]["chance_rank1"])
        for suite in required
    )
    no_dominance = suite_present and all(
        min(float(row["auroc"]) for row in suite_results[suite]["metrics"]["per_source"].values())
        > 0.5
        for suite in required
    )
    curve_checks = []
    for suite in required:
        rows = sorted(template_curves.get(suite, ()), key=lambda row: float(row["n_template"]))
        if len(rows) != 6:
            curve_checks.append(False)
            continue
        correlation = spearmanr(
            [float(row["n_template"]) for row in rows],
            [float(row["macro_source_auroc"]) for row in rows],
        )
        curve_checks.append(
            float(correlation.statistic) > 0.5
            and float(rows[-1]["macro_source_auroc"]) >= float(rows[0]["macro_source_auroc"])
        )
    template_improvement = len(curve_checks) == 2 and all(curve_checks)
    low_bit = required <= set(low_bit_uniform_results) and all(
        float(low_bit_uniform_results[suite]["bootstrap_ci_low"]) > 0.5 for suite in required
    )
    checks = {
        "cross_and_near_family_suites": suite_present,
        "macro_source_auroc_two_way_ci_lower_above_chance": bool(macro_ci),
        "rank1_above_random_with_block_permutation": bool(rank1),
        "no_single_source_dominance": bool(no_dominance),
        "stable_template_size_improvement": bool(template_improvement),
        "low_bit_survives_uniform_export": bool(low_bit),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "status": "PASS" if passed else "RESTRICTED_CLAIM",
        "checks": checks,
        "decision": (
            "proceed_to_e03_configuration_and_hierarchy_analysis"
            if passed
            else "stop_or_restrict_fixed_generator_fingerprint_mainline"
        ),
    }
