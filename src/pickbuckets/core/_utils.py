from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite, isinf, isnan
from typing import Any

from pickbuckets.exceptions import InvalidBucketingError
from pickbuckets.runtime.apply import is_missing

MISSING_STRATEGIES = {"separate", "most_frequent", "error", "propagate"}
BOUNDARY_STRATEGIES = {"clip", "underflow_overflow", "error"}
UNKNOWN_CATEGORY_STRATEGIES = {"other", "missing", "error", "keep"}


def numeric_values(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if is_missing(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            message = f"Non-numeric value encountered: {value!r}"
            raise InvalidBucketingError(message) from exc
        if not isfinite(number):
            raise InvalidBucketingError(f"Non-finite value encountered: {value!r}")
        result.append(number)
    if not result:
        raise InvalidBucketingError(
            "At least one non-missing numeric value is required."
        )
    return result


def validate_n_bins(n_bins: int) -> None:
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise InvalidBucketingError("n_bins must be a positive integer.")


def validate_min_frequency(min_frequency: int | float) -> None:
    if (
        not isinstance(min_frequency, (int, float))
        or isinstance(min_frequency, bool)
        or min_frequency <= 0
    ):
        raise InvalidBucketingError(
            "min_frequency must be a positive count or ratio."
        )
    if isinstance(min_frequency, float) and min_frequency > 1:
        raise InvalidBucketingError(
            "Float min_frequency must be in the range (0, 1]."
        )


def validate_missing_strategy(strategy: str) -> None:
    if not isinstance(strategy, str) or strategy not in MISSING_STRATEGIES:
        allowed = "', '".join(sorted(MISSING_STRATEGIES))
        raise InvalidBucketingError(
            f"missing_strategy must be one of '{allowed}'."
        )


def validate_boundary_strategy(strategy: str) -> None:
    if not isinstance(strategy, str) or strategy not in BOUNDARY_STRATEGIES:
        allowed = "', '".join(sorted(BOUNDARY_STRATEGIES))
        raise InvalidBucketingError(
            f"boundary_strategy must be one of '{allowed}'."
        )


def validate_unknown_category_strategy(strategy: str) -> None:
    if not isinstance(strategy, str) or strategy not in UNKNOWN_CATEGORY_STRATEGIES:
        allowed = "', '".join(sorted(UNKNOWN_CATEGORY_STRATEGIES))
        raise InvalidBucketingError(
            f"unknown_category_strategy must be one of '{allowed}'."
        )


def validate_edges(edges: Sequence[float]) -> list[float]:
    if len(edges) < 2:
        raise InvalidBucketingError("At least two edges are required.")
    converted: list[float] = []
    for edge in edges:
        try:
            number = float(edge)
        except (TypeError, ValueError) as exc:
            raise InvalidBucketingError(f"Invalid edge value: {edge!r}") from exc
        if isnan(number):
            raise InvalidBucketingError("Edges cannot contain NaN.")
        converted.append(number)
    if any(left >= right for left, right in zip(converted, converted[1:])):
        raise InvalidBucketingError("Edges must be sorted and unique.")
    return converted


def make_labels(edges: Sequence[float], labels: str | Sequence[Any]) -> list[Any]:
    n_intervals = len(edges) - 1
    if labels == "ordinal":
        return list(range(n_intervals))
    if labels == "interval":
        result: list[str] = []
        for index, (left, right) in enumerate(zip(edges, edges[1:])):
            close = "]" if index == n_intervals - 1 else ")"
            result.append(f"[{_format_edge(left)}, {_format_edge(right)}{close}")
        return result
    if isinstance(labels, str):
        raise InvalidBucketingError(
            "labels must be 'ordinal', 'interval', or a sequence."
        )
    result = list(labels)
    if len(result) != n_intervals:
        raise InvalidBucketingError("Custom label count must equal interval count.")
    return result


def missing_count(values: Iterable[Any]) -> int:
    return sum(1 for value in values if is_missing(value))


def numeric_bin_counts(
    values: Iterable[float],
    edges: Sequence[float],
    labels: Sequence[Any],
    *,
    boundary_strategy: str = "clip",
    underflow_label: Any = "Underflow",
    overflow_label: Any = "Overflow",
) -> list[dict[str, Any]]:
    count_labels = list(labels)
    if boundary_strategy == "underflow_overflow":
        count_labels = [underflow_label, *count_labels, overflow_label]
    counts = [0 for _ in count_labels]
    for value in values:
        if value < edges[0]:
            index = 0
        elif value > edges[-1]:
            index = len(counts) - 1
        else:
            offset = 1 if boundary_strategy == "underflow_overflow" else 0
            index = offset + _numeric_bin_index(value, edges)
        counts[index] += 1
    return [
        {"label": label, "count": count}
        for label, count in zip(count_labels, counts)
    ]


def label_counts(
    values: Iterable[Any],
    labels: Sequence[Any],
) -> list[dict[str, Any]]:
    counts = [0 for _ in labels]
    for value in values:
        for index, label in enumerate(labels):
            if value == label:
                counts[index] += 1
                break
    return [
        {"label": label, "count": count}
        for label, count in zip(labels, counts)
    ]


def most_frequent_label(counts: Sequence[dict[str, Any]]) -> Any:
    best = counts[0]
    for item in counts[1:]:
        if item["count"] > best["count"]:
            best = item
    return best["label"]


def _numeric_bin_index(value: float, edges: Sequence[float]) -> int:
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        is_final = index == len(edges) - 2
        if left <= value < right or (is_final and left <= value <= right):
            return index
    return len(edges) - 2


def _format_edge(edge: float) -> str:
    if isinf(edge):
        return "inf" if edge > 0 else "-inf"
    if edge.is_integer():
        return str(int(edge))
    return f"{edge:g}"
