from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite, isinf, isnan
from typing import Any

from pickbuckets.exceptions import InvalidBucketingError
from pickbuckets.runtime.apply import is_missing


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
    if not isinstance(n_bins, int) or n_bins < 1:
        raise InvalidBucketingError("n_bins must be a positive integer.")


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


def _format_edge(edge: float) -> str:
    if isinf(edge):
        return "inf" if edge > 0 else "-inf"
    if edge.is_integer():
        return str(int(edge))
    return f"{edge:g}"
