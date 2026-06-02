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
    validate_edges,
    validate_missing_strategy,
)
from pickbuckets.core.base import BaseBucket
from pickbuckets.exceptions import InvalidBucketingError
from pickbuckets.rules import Rule
from pickbuckets.rules.schema import BoundaryStrategy, MissingStrategy


class CustomBoundaryBucket(BaseBucket):
    def __init__(
        self,
        edges: Sequence[float],
        labels: str | Sequence[Any] = "ordinal",
        *,
        feature_name: str | None = None,
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
    ) -> None:
        validate_missing_strategy(missing_strategy)
        validate_boundary_strategy(boundary_strategy)
        self.edges = validate_edges(edges)
        make_labels(self.edges, labels)
        self.labels = labels
        self.feature_name = feature_name
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label

    def fit(self, values: Iterable[Any] | None = None) -> CustomBoundaryBucket:
        labels = make_labels(self.edges, self.labels)
        missing_label = self.missing_label
        fit_stats: dict[str, Any] = {"algorithm": "custom_boundary"}
        if values is not None:
            raw = list(values)
            clean = numeric_values(raw)
            if self.boundary_strategy == "error":
                below = any(value < self.edges[0] for value in clean)
                above = any(value > self.edges[-1] for value in clean)
                if below or above:
                    raise InvalidBucketingError(
                        "Fit values fall outside custom edges while "
                        "boundary_strategy='error'."
                    )
            bin_counts = numeric_bin_counts(
                clean,
                self.edges,
                labels,
                boundary_strategy=self.boundary_strategy,
                underflow_label=self.underflow_label,
                overflow_label=self.overflow_label,
            )
            fit_stats["n_observations"] = len(clean)
            fit_stats["n_missing"] = missing_count(raw)
            fit_stats["bin_counts"] = bin_counts
            if self.missing_strategy == "most_frequent":
                missing_label = most_frequent_label(bin_counts)
        elif self.missing_strategy == "most_frequent":
            raise InvalidBucketingError(
                "CustomBoundaryBucket.fit(values) requires data when "
                "missing_strategy='most_frequent'."
            )
        self.rules_ = Rule(
            kind="numeric",
            feature_name=self.feature_name,
            edges=self.edges,
            labels=labels,
            missing_strategy=self.missing_strategy,
            missing_label=missing_label,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
            fit_stats=fit_stats,
        )
        return self
