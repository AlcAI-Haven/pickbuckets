from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from pickbuckets.core._utils import (
    make_labels,
    missing_count,
    most_frequent_label,
    numeric_bin_counts,
    numeric_values,
    validate_boundary_strategy,
    validate_missing_strategy,
    validate_n_bins,
)
from pickbuckets.core.base import BaseBucket
from pickbuckets.exceptions import InvalidBucketingError
from pickbuckets.rules import Rule
from pickbuckets.rules.schema import BoundaryStrategy, MissingStrategy


class EqualFrequencyBucket(BaseBucket):
    def __init__(
        self,
        n_bins: int = 5,
        labels: str | Sequence[Any] = "ordinal",
        *,
        duplicates: str = "raise",
        feature_name: str | None = None,
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
    ) -> None:
        validate_n_bins(n_bins)
        validate_missing_strategy(missing_strategy)
        validate_boundary_strategy(boundary_strategy)
        if not isinstance(duplicates, str) or duplicates not in {"raise", "drop"}:
            raise InvalidBucketingError("duplicates must be either 'raise' or 'drop'.")
        self.n_bins = n_bins
        self.labels = labels
        self.duplicates = duplicates
        self.feature_name = feature_name
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label

    def fit(self, values: Iterable[Any]) -> EqualFrequencyBucket:
        raw = list(values)
        clean = sorted(numeric_values(raw))
        edges = [
            _quantile(clean, index / self.n_bins)
            for index in range(self.n_bins + 1)
        ]
        deduped = _dedupe_edges(edges)
        if len(deduped) != len(edges):
            if self.duplicates == "raise":
                raise InvalidBucketingError(
                    "Duplicate quantile edges produced; use duplicates='drop' to keep "
                    "the unique intervals."
                )
            edges = deduped if len(deduped) > 1 else [deduped[0], deduped[0]]
        if len(edges) < 2:
            raise InvalidBucketingError("Could not produce at least one interval.")
        labels = make_labels(edges, self.labels)
        bin_counts = numeric_bin_counts(
            clean,
            edges,
            labels,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
        )
        missing_label = self.missing_label
        if self.missing_strategy == "most_frequent":
            missing_label = most_frequent_label(bin_counts)

        self.rules_ = Rule(
            kind="numeric",
            feature_name=self.feature_name,
            edges=edges,
            labels=labels,
            missing_strategy=self.missing_strategy,
            missing_label=missing_label,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
            fit_stats={
                "algorithm": "equal_frequency",
                "n_observations": len(clean),
                "n_missing": missing_count(raw),
                "requested_bins": self.n_bins,
                "actual_bins": len(edges) - 1,
                "bin_counts": bin_counts,
            },
        )
        return self


def _quantile(values: Sequence[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _dedupe_edges(edges: Sequence[float]) -> list[float]:
    result: list[float] = []
    for edge in edges:
        if not result or edge != result[-1]:
            result.append(edge)
    return result
