from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from pickbuckets.core._utils import make_labels, numeric_values, validate_edges
from pickbuckets.core.base import BaseBucket
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
    ) -> None:
        self.edges = validate_edges(edges)
        make_labels(self.edges, labels)
        self.labels = labels
        self.feature_name = feature_name
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy

    def fit(self, values: Iterable[Any] | None = None) -> CustomBoundaryBucket:
        fit_stats: dict[str, Any] = {"algorithm": "custom_boundary"}
        if values is not None:
            clean = numeric_values(values)
            fit_stats["n_observations"] = len(clean)
        self.rules_ = Rule(
            kind="numeric",
            feature_name=self.feature_name,
            edges=self.edges,
            labels=make_labels(self.edges, self.labels),
            missing_strategy=self.missing_strategy,
            missing_label=self.missing_label,
            boundary_strategy=self.boundary_strategy,
            fit_stats=fit_stats,
        )
        return self
