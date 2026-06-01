from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from pickbuckets.core._utils import make_labels, numeric_values, validate_n_bins
from pickbuckets.core.base import BaseBucket
from pickbuckets.rules import Rule


class EqualWidthBucket(BaseBucket):
    def __init__(
        self,
        n_bins: int = 5,
        labels: str | Sequence[Any] = "ordinal",
        *,
        feature_name: str | None = None,
        missing_strategy: str = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: str = "clip",
    ) -> None:
        validate_n_bins(n_bins)
        self.n_bins = n_bins
        self.labels = labels
        self.feature_name = feature_name
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy

    def fit(self, values: Iterable[Any]) -> EqualWidthBucket:
        clean = numeric_values(values)
        minimum = min(clean)
        maximum = max(clean)
        if minimum == maximum:
            edges = [minimum, maximum]
        else:
            width = (maximum - minimum) / self.n_bins
            edges = [minimum + width * index for index in range(self.n_bins + 1)]
            edges[-1] = maximum

        self.rules_ = Rule(
            kind="numeric",
            feature_name=self.feature_name,
            edges=edges,
            labels=make_labels(edges, self.labels),
            missing_strategy=self.missing_strategy,
            missing_label=self.missing_label,
            boundary_strategy=self.boundary_strategy,
            fit_stats={
                "algorithm": "equal_width",
                "n_observations": len(clean),
                "minimum": minimum,
                "maximum": maximum,
            },
        )
        return self

