"""Independent nuisance-control contracts for later G-FITS phases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from gfits.explicit_scoring import paper_fits_plus_c


@dataclass(frozen=True)
class NuisanceControlBank:
    """A candidate-independent, same-pipeline and same-geometry donor bank."""

    pipeline_id: str
    geometry_id: str
    donor_source_ids: tuple[str, ...]
    candidate_source_ids: tuple[str, ...]
    donor_image_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        donors = set(self.donor_source_ids)
        candidates = set(self.candidate_source_ids)
        if not self.pipeline_id or not self.geometry_id:
            raise ValueError("nuisance control bank requires pipeline_id and geometry_id")
        if len(donors) < 3:
            raise ValueError("mean/median nuisance comparison requires at least three donors")
        if len(donors) != len(self.donor_source_ids):
            raise ValueError("donor source IDs must be unique")
        if not candidates or len(candidates) != len(self.candidate_source_ids):
            raise ValueError("candidate source IDs must be non-empty and unique")
        overlap = donors & candidates
        if overlap:
            raise ValueError(f"nuisance donors overlap candidate sources: {sorted(overlap)}")
        if len(set(self.donor_image_sha256)) != len(self.donor_image_sha256):
            raise ValueError("nuisance donor image hashes must be unique")

    def validate_query(
        self,
        *,
        query_source_id: str,
        query_image_sha256: str,
        pipeline_id: str,
        geometry_id: str,
    ) -> None:
        """Fail closed on source, image, pipeline, or geometry leakage."""

        if query_source_id in set(self.donor_source_ids):
            raise ValueError("query source is present in the nuisance donor bank")
        if query_image_sha256 in set(self.donor_image_sha256):
            raise ValueError("query image is present in the nuisance donor bank")
        if pipeline_id != self.pipeline_id:
            raise ValueError("query and nuisance control pipeline IDs do not match")
        if geometry_id != self.geometry_id:
            raise ValueError("query and nuisance control geometry IDs do not match")

    def denominator(
        self,
        donor_scores: Mapping[str, float],
        *,
        reducer: str = "median",
    ) -> float:
        """Return one denominator shared by every candidate for a query."""

        if set(donor_scores) != set(self.donor_source_ids):
            raise ValueError("donor scores do not exactly match the registered control bank")
        values = np.asarray([donor_scores[source] for source in self.donor_source_ids], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("nuisance donor scores must be finite")
        if reducer == "median":
            return float(np.median(values))
        if reducer == "mean":
            return float(np.mean(values))
        raise ValueError(f"unsupported nuisance reducer: {reducer}")


def calibrate_candidates(
    candidate_scores: Mapping[str, float],
    nuisance_denominator: float,
    *,
    constant: float,
) -> dict[str, float]:
    """Apply a single candidate-independent denominator to all candidates."""

    if not candidate_scores:
        raise ValueError("candidate_scores must not be empty")
    denominator = float(nuisance_denominator)
    return {
        source: paper_fits_plus_c(score, denominator, constant=constant)
        for source, score in candidate_scores.items()
    }


def assert_candidate_independent_denominator(rows: Sequence[Mapping[str, object]]) -> None:
    """Verify that each query has one nuisance denominator across candidates."""

    grouped: dict[str, set[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["query_id"]), set()).add(float(row["nuisance_denominator"]))
    invalid = {query: values for query, values in grouped.items() if len(values) != 1}
    if invalid:
        raise ValueError(f"candidate-dependent nuisance denominators detected: {invalid}")
