from __future__ import annotations

from bisect import insort
from collections.abc import Iterable, Sequence
from math import isfinite
from typing import Any

from pickbuckets.core._utils import (
    make_labels,
    most_frequent_label,
    validate_n_bins,
)
from pickbuckets.core.base import BaseBucket
from pickbuckets.exceptions import InvalidBucketingError, NotFittedError
from pickbuckets.rules import Rule
from pickbuckets.rules.schema import BoundaryStrategy, MissingStrategy
from pickbuckets.runtime.apply import is_missing

__all__ = ["StreamingEqualFrequencyBucket", "StreamingHistogram"]


class StreamingHistogram:
    """Bounded-memory streaming quantile sketch.

    The sketch keeps at most ``max_centroids`` weighted centroids. Each update
    inserts a unit-weight centroid and, when the budget is exceeded, merges the
    two adjacent centroids with the smallest value gap. This is the streaming
    histogram of Ben-Haim and Tom-Tov (2010), reduced to the parts pickbuckets
    needs: bounded memory, mergeable state, and approximate quantiles.

    Quantiles use a weighted type-7 order statistic over the centroids (the same
    convention as :class:`~pickbuckets.EqualFrequencyBucket` and NumPy). When no
    merging has happened — ``max_centroids`` at or above the number of distinct
    values — the centroids represent the data exactly and the quantiles match the
    exact bucketer. As centroids merge, their means introduce approximation that
    shrinks again as ``max_centroids`` grows.
    """

    def __init__(self, max_centroids: int = 256) -> None:
        if (
            isinstance(max_centroids, bool)
            or not isinstance(max_centroids, int)
            or max_centroids < 2
        ):
            raise InvalidBucketingError("max_centroids must be an integer >= 2.")
        self.max_centroids = max_centroids
        self.centroids: list[list[float]] = []
        self.total = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, value: float, count: int = 1) -> None:
        if count <= 0:
            return
        self.total += count
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

        index = self._index_of(value)
        if index is not None:
            self.centroids[index][1] += count
            return
        insort(self.centroids, [value, float(count)])
        self._shrink()

    def merge(self, other: StreamingHistogram) -> None:
        for value, count in other.centroids:
            self.update(value, int(count))

    def quantile(self, q: float) -> float:
        if self.total == 0:
            raise InvalidBucketingError("Cannot compute quantiles from no data.")
        if q <= 0.0:
            return self.minimum
        if q >= 1.0:
            return self.maximum

        # Type-7 (linear) quantile over the weighted centroids, matching
        # EqualFrequencyBucket and NumPy's default.
        position = q * (self.total - 1)
        lower = int(position)
        fraction = position - lower
        low_value = self._value_at_rank(lower)
        if fraction <= 0.0:
            return low_value
        high_value = self._value_at_rank(lower + 1)
        return low_value + fraction * (high_value - low_value)

    def estimated_count_below(self, edge: float) -> float:
        """Number of observations strictly below ``edge`` (exact when lossless)."""

        if self.total == 0:
            return 0.0
        count = 0.0
        for value, weight in self.centroids:
            if value < edge:
                count += weight
            else:
                break
        return count

    def _value_at_rank(self, rank: int) -> float:
        """Return the ``rank``-th smallest value (0-indexed) over the centroids."""

        cumulative = 0.0
        for value, weight in self.centroids:
            if rank < cumulative + weight:
                return value
            cumulative += weight
        return self.centroids[-1][0]

    def _index_of(self, value: float) -> int | None:
        for index, centroid in enumerate(self.centroids):
            if centroid[0] == value:
                return index
            if centroid[0] > value:
                return None
        return None

    def _shrink(self) -> None:
        while len(self.centroids) > self.max_centroids:
            gaps = [
                (right[0] - left[0], index)
                for index, (left, right) in enumerate(
                    zip(self.centroids, self.centroids[1:])
                )
            ]
            _, merge_index = min(gaps)
            left = self.centroids[merge_index]
            right = self.centroids[merge_index + 1]
            total = left[1] + right[1]
            merged_value = (left[0] * left[1] + right[0] * right[1]) / total
            self.centroids[merge_index] = [merged_value, total]
            del self.centroids[merge_index + 1]


class StreamingEqualFrequencyBucket(BaseBucket):
    """Approximate equal-frequency bucketing for data that does not fit in memory.

    This is an *online* approximation kept deliberately separate from the exact
    :class:`~pickbuckets.EqualFrequencyBucket`. Feed data in chunks with
    :meth:`partial_fit`, then call :meth:`finalize` (or simply ``transform``)
    to materialise a standard portable :class:`~pickbuckets.Rule`. The resulting
    rule applies through the same pure-Python runtime as every other bucketer;
    only fitting is approximate.

    Approximation error is bounded by ``max_centroids``: edges converge to the
    exact quantiles as it grows, and are exact once it reaches the number of
    distinct values seen.
    """

    def __init__(
        self,
        n_bins: int = 5,
        labels: str | Sequence[Any] = "ordinal",
        *,
        max_centroids: int = 256,
        duplicates: str = "drop",
        feature_name: str | None = None,
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
    ) -> None:
        validate_n_bins(n_bins)
        if not isinstance(duplicates, str) or duplicates not in {"raise", "drop"}:
            raise InvalidBucketingError("duplicates must be either 'raise' or 'drop'.")
        self.n_bins = n_bins
        self.labels = labels
        self.max_centroids = max_centroids
        self.duplicates = duplicates
        self.feature_name = feature_name
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label
        self._sketch = StreamingHistogram(max_centroids)
        self._n_missing = 0
        self._dirty = True

    def partial_fit(
        self, values: Iterable[Any]
    ) -> StreamingEqualFrequencyBucket:
        """Update the sketch with another chunk of values."""

        for value in values:
            if is_missing(value):
                self._n_missing += 1
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise InvalidBucketingError(
                    f"Non-numeric value encountered: {value!r}"
                ) from exc
            if not isfinite(number):
                raise InvalidBucketingError(f"Non-finite value encountered: {value!r}")
            self._sketch.update(number)
        self._dirty = True
        return self

    def fit(self, values: Iterable[Any]) -> StreamingEqualFrequencyBucket:
        """Reset the sketch and fit from a single chunk."""

        self._sketch = StreamingHistogram(self.max_centroids)
        self._n_missing = 0
        self.partial_fit(values)
        return self.finalize()

    def finalize(self) -> StreamingEqualFrequencyBucket:
        """Materialise the current sketch into a portable rule."""

        if self._sketch.total == 0:
            raise InvalidBucketingError(
                "At least one non-missing numeric value is required."
            )
        edges = [
            self._sketch.quantile(index / self.n_bins)
            for index in range(self.n_bins + 1)
        ]
        deduped = _dedupe_edges(edges)
        if len(deduped) != len(edges):
            if self.duplicates == "raise":
                raise InvalidBucketingError(
                    "Duplicate approximate edges produced; use duplicates='drop' to "
                    "keep the unique intervals."
                )
            edges = deduped
        if len(edges) < 2:
            edges = [self._sketch.minimum, self._sketch.maximum]
            if edges[0] == edges[1]:
                raise InvalidBucketingError(
                    "Could not produce at least one interval from constant input."
                )
        labels = make_labels(edges, self.labels)
        bin_counts = self._approximate_bin_counts(edges, labels)
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
                "algorithm": "streaming_equal_frequency",
                "approximate": True,
                "n_observations": self._sketch.total,
                "n_missing": self._n_missing,
                "requested_bins": self.n_bins,
                "actual_bins": len(edges) - 1,
                "max_centroids": self.max_centroids,
                "n_centroids": len(self._sketch.centroids),
                "bin_counts": bin_counts,
            },
        )
        self._dirty = False
        return self

    def transform(self, values: Iterable[Any]) -> Any:
        if self._dirty or not hasattr(self, "rules_"):
            if self._sketch.total == 0:
                raise NotFittedError(
                    "This bucketer must be fitted before transform()."
                )
            self.finalize()
        return super().transform(values)

    def _approximate_bin_counts(
        self, edges: Sequence[float], labels: Sequence[Any]
    ) -> list[dict[str, Any]]:
        count_labels = list(labels)
        if self.boundary_strategy == "underflow_overflow":
            count_labels = [self.underflow_label, *count_labels, self.overflow_label]
        below = [self._sketch.estimated_count_below(edge) for edge in edges]
        interior = [
            max(0.0, below[index + 1] - below[index])
            for index in range(len(edges) - 1)
        ]
        # Anchor the final bin so the estimated mass sums to the total seen.
        interior[-1] = max(0.0, self._sketch.total - sum(interior[:-1]))
        rounded = [int(round(mass)) for mass in interior]
        if self.boundary_strategy == "underflow_overflow":
            rounded = [0, *rounded, 0]
        return [
            {"label": label, "count": count}
            for label, count in zip(count_labels, rounded)
        ]


def _dedupe_edges(edges: Sequence[float]) -> list[float]:
    result: list[float] = []
    for edge in edges:
        if not result or edge != result[-1]:
            result.append(edge)
    return result
