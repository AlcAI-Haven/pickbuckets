from __future__ import annotations

import sys
from collections.abc import Iterable
from math import isnan
from typing import Any

from pickbuckets.exceptions import (
    BoundaryError,
    InvalidBucketingError,
    MissingValueError,
    UnknownCategoryError,
)
from pickbuckets.rules import Rule


def apply_rule(rule: Rule, values: Iterable[Any]) -> list[Any]:
    """Apply a saved rule to plain Python values."""

    if rule.kind == "numeric":
        return [_apply_numeric(rule, value) for value in values]
    return [_apply_categorical(rule, value) for value in values]


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    # Recognise pandas NA/NaT (and similar) without importing pandas: only
    # consult the module if the caller already imported it.
    pandas = sys.modules.get("pandas")
    if pandas is not None:
        try:
            return bool(pandas.isna(value))
        except (TypeError, ValueError):
            return False
    return False


def _context(rule: Rule) -> str:
    if rule.feature_name is None:
        return ""
    return f"Feature {rule.feature_name!r}: "


def _handle_missing(rule: Rule, value: Any) -> Any:
    if rule.missing_strategy in {"separate", "most_frequent"}:
        return rule.missing_label
    if rule.missing_strategy == "propagate":
        return value
    raise MissingValueError(
        f"{_context(rule)}Missing value encountered and "
        "missing_strategy='error'."
    )


def _apply_numeric(rule: Rule, value: Any) -> Any:
    if is_missing(value):
        return _handle_missing(rule, value)

    edges = rule.edges or []
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        message = (
            f"{_context(rule)}Non-numeric value encountered during "
            f"transform: {value!r}"
        )
        raise InvalidBucketingError(message) from exc
    if isnan(number):
        return _handle_missing(rule, value)
    first = edges[0]
    last = edges[-1]

    if number < first:
        if rule.boundary_strategy == "error":
            raise BoundaryError(
                f"{_context(rule)}Value {value!r} is below the first edge {first!r}."
            )
        if rule.boundary_strategy == "underflow_overflow":
            return rule.underflow_label
        return rule.labels[0]
    if number > last:
        if rule.boundary_strategy == "error":
            raise BoundaryError(
                f"{_context(rule)}Value {value!r} is above the final edge {last!r}."
            )
        if rule.boundary_strategy == "underflow_overflow":
            return rule.overflow_label
        return rule.labels[-1]

    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        is_final = index == len(edges) - 2
        if left <= number < right or (is_final and left <= number <= right):
            return rule.labels[index]

    if rule.boundary_strategy == "error":
        raise BoundaryError(
            f"{_context(rule)}Value {value!r} did not fit any interval."
        )
    return rule.labels[-1]


def _apply_categorical(rule: Rule, value: Any) -> Any:
    if is_missing(value):
        return _handle_missing(rule, value)

    key = str(value)
    mapping = rule.category_mapping or {}
    if key in mapping:
        return mapping[key]
    if rule.unknown_category_strategy == "other":
        return rule.unknown_label
    if rule.unknown_category_strategy == "missing":
        return rule.missing_label
    if rule.unknown_category_strategy == "keep":
        return key
    raise UnknownCategoryError(f"{_context(rule)}Unknown category: {value!r}")
