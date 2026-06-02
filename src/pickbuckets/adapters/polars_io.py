"""Vectorized Polars apply path.

Rules are applied with native Polars expressions (``when/then`` chains and
``replace_strict``) — never a row-wise Python loop. The same Phase 1 rule object
drives this path and the pandas/pure-Python path, so outputs match.
"""

from __future__ import annotations

from typing import Any

from pickbuckets.exceptions import BoundaryError, PickBucketsError
from pickbuckets.rules import Rule


def _pl() -> Any:
    import polars as pl  # local import; never a core dependency

    return pl


def _label_pool(rule: Rule) -> list[Any]:
    pool: list[Any] = list(rule.labels)
    if rule.missing_strategy == "separate":
        pool.append(rule.missing_label)
    if rule.kind == "categorical":
        if rule.category_mapping:
            pool.extend(rule.category_mapping.values())
        if rule.unknown_category_strategy == "other":
            pool.append(rule.unknown_label)
    return pool


def _homogeneous(rule: Rule) -> bool:
    """True when every possible output value shares one Python type."""

    if rule.missing_strategy == "propagate":
        return False
    types = {type(value) for value in _label_pool(rule)}
    # bool is a subclass of int; treat the {int} / {str} / {float} cases as homogeneous.
    return len(types) <= 1


def _stringify(value: Any) -> Any:
    return value if value is None else str(value)


def rule_raises(rule: Rule) -> bool:
    """Whether the rule uses a strategy that must raise on bad input."""

    return (
        rule.missing_strategy == "error"
        or rule.boundary_strategy == "error"
        or rule.unknown_category_strategy == "error"
    )


def numeric_expr(rule: Rule, col: Any, *, use_object_dtype: bool) -> Any:
    pl = _pl()
    edges = rule.edges or []
    labels = rule.labels

    def lit(value: Any) -> Any:
        if use_object_dtype:
            return pl.lit(value, dtype=pl.Object)
        return pl.lit(value)

    chain = pl.when(col < edges[1]).then(lit(labels[0]))
    for index in range(1, len(labels) - 1):
        chain = chain.when(col < edges[index + 1]).then(lit(labels[index]))
    mapped = chain.otherwise(lit(labels[-1]))

    # Dtype-agnostic missing mask. Polars treats ``NaN == NaN`` as True, so the
    # IEEE ``col != col`` trick fails; cast to Float64 (non-strict) and use
    # ``is_nan`` instead, which works on a bare Expr without knowing the dtype.
    missing_mask = col.is_null() | col.cast(pl.Float64, strict=False).is_nan()

    if rule.missing_strategy == "separate":
        return pl.when(missing_mask).then(lit(rule.missing_label)).otherwise(mapped)
    if rule.missing_strategy == "propagate":
        passthrough = col.cast(pl.Object) if use_object_dtype else col
        return pl.when(missing_mask).then(passthrough).otherwise(mapped)
    return mapped  # error mode handled by pre-check


def categorical_expr(rule: Rule, col: Any, *, stringify: bool) -> Any:
    pl = _pl()
    mapping = rule.category_mapping or {}

    def out(value: Any) -> Any:
        return _stringify(value) if stringify else value

    key = col.cast(pl.Utf8)
    mapped = key.replace_strict(
        list(mapping.keys()),
        [out(value) for value in mapping.values()],
        default=out(rule.unknown_label),
    )
    if rule.missing_strategy == "separate":
        return (
            pl.when(col.is_null())
            .then(pl.lit(out(rule.missing_label)))
            .otherwise(mapped)
        )
    if rule.missing_strategy == "propagate":
        passthrough = col.cast(pl.Utf8) if stringify else col
        return pl.when(col.is_null()).then(passthrough).otherwise(mapped)
    return mapped


def rule_expr(rule: Rule, col: Any) -> Any:
    """Build a Polars expression that applies ``rule`` to a column expression."""

    stringify = not _homogeneous(rule)
    if rule.kind == "numeric":
        return numeric_expr(rule, col, use_object_dtype=stringify)
    return categorical_expr(rule, col, stringify=stringify)


def precheck_series(rule: Rule, series: Any) -> None:
    """Eager-path validation for raising strategies."""

    pl = _pl()
    if rule.missing_strategy == "error":
        has_null = series.null_count() > 0
        has_nan = bool(series.is_nan().any()) if series.dtype.is_float() else False
        if has_null or has_nan:
            from pickbuckets.exceptions import MissingValueError

            raise MissingValueError(
                "Missing value encountered and missing_strategy='error'."
            )
    if rule.kind == "numeric" and rule.boundary_strategy == "error":
        edges = rule.edges or []
        finite = series.drop_nulls()
        if finite.dtype.is_float():
            finite = finite.filter(~finite.is_nan())
        if finite.len():
            below = finite.min() < edges[0]
            above = finite.max() > edges[-1]
            if below or above:
                raise BoundaryError("Value out of range and boundary_strategy='error'.")
    if rule.kind == "categorical" and rule.unknown_category_strategy == "error":
        mapping = rule.category_mapping or {}
        known = set(mapping.keys())
        seen = series.drop_nulls().cast(pl.Utf8).unique().to_list()
        unknown = [value for value in seen if value not in known]
        if unknown:
            from pickbuckets.exceptions import UnknownCategoryError

            raise UnknownCategoryError(f"Unknown category: {unknown[0]!r}")


def apply_rule_polars(rule: Rule, series: Any) -> Any:
    """Apply ``rule`` to a Polars Series, returning a Polars Series."""

    pl = _pl()
    precheck_series(rule, series)
    frame = series.to_frame()
    name = series.name
    result = frame.select(rule_expr(rule, pl.col(name)).alias(name))
    return result.to_series(0)


def lazy_guard(rule: Rule) -> None:
    if rule_raises(rule):
        raise PickBucketsError(
            "Rules using an 'error' strategy (missing/boundary/unknown) require "
            "eager evaluation; collect the LazyFrame or use a non-raising strategy."
        )
