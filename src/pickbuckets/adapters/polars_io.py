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
    if rule.missing_strategy in {"separate", "most_frequent"}:
        pool.append(rule.missing_label)
    if rule.kind == "numeric":
        if rule.boundary_strategy == "underflow_overflow":
            pool.extend([rule.underflow_label, rule.overflow_label])
    else:
        if rule.category_mapping:
            pool.extend(rule.category_mapping.values())
        if rule.unknown_category_strategy == "other":
            pool.append(rule.unknown_label)
        if rule.unknown_category_strategy == "missing":
            pool.append(rule.missing_label)
        if rule.unknown_category_strategy == "keep":
            pool.append("")
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


def _context(rule: Rule) -> str:
    if rule.feature_name is None:
        return ""
    return f"Feature {rule.feature_name!r}: "


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
    if rule.boundary_strategy == "underflow_overflow":
        mapped = (
            pl.when(col < edges[0])
            .then(lit(rule.underflow_label))
            .when(col > edges[-1])
            .then(lit(rule.overflow_label))
            .otherwise(mapped)
        )

    # Dtype-agnostic missing mask. Polars treats ``NaN == NaN`` as True, so the
    # IEEE ``col != col`` trick fails; cast to Float64 (non-strict) and use
    # ``is_nan`` instead, which works on a bare Expr without knowing the dtype.
    missing_mask = col.is_null() | col.cast(pl.Float64, strict=False).is_nan()

    if rule.missing_strategy in {"separate", "most_frequent"}:
        return pl.when(missing_mask).then(lit(rule.missing_label)).otherwise(mapped)
    if rule.missing_strategy == "propagate":
        null_value = pl.lit(None, dtype=pl.Object) if use_object_dtype else pl.lit(None)
        nan_value = lit(float("nan"))
        return (
            pl.when(col.is_null())
            .then(null_value)
            .when(col.cast(pl.Float64, strict=False).is_nan())
            .then(nan_value)
            .otherwise(mapped)
        )
    return mapped  # error mode handled by pre-check


def categorical_expr(rule: Rule, col: Any, *, stringify: bool) -> Any:
    pl = _pl()
    mapping = rule.category_mapping or {}

    def out(value: Any) -> Any:
        return _stringify(value) if stringify else value

    key = col.cast(pl.Utf8)
    if stringify:
        if rule.unknown_category_strategy == "keep":
            raise PickBucketsError(
                "Polars cannot preserve mixed-type categorical outputs with "
                "unknown_category_strategy='keep'. Use string labels or choose "
                "'other', 'missing', or 'error'."
            )

        def lit_obj(value: Any) -> Any:
            return pl.lit(value, dtype=pl.Object)

        if rule.unknown_category_strategy == "missing":
            mapped = lit_obj(rule.missing_label)
        else:
            mapped = lit_obj(rule.unknown_label)
        for category, value in mapping.items():
            mapped = (
                pl.when(key == category)
                .then(lit_obj(value))
                .otherwise(mapped)
            )
    else:
        if rule.unknown_category_strategy == "keep":
            unknown = key
        elif rule.unknown_category_strategy == "missing":
            unknown = pl.lit(out(rule.missing_label))
        else:
            unknown = pl.lit(out(rule.unknown_label))
        mapped = key.replace_strict(
            list(mapping.keys()),
            [out(value) for value in mapping.values()],
            default=unknown,
        )
    if rule.missing_strategy in {"separate", "most_frequent"}:
        missing = (
            pl.lit(rule.missing_label, dtype=pl.Object)
            if stringify
            else pl.lit(out(rule.missing_label))
        )
        return (
            pl.when(col.is_null())
            .then(missing)
            .otherwise(mapped)
        )
    if rule.missing_strategy == "propagate":
        null_value = (
            pl.lit(None, dtype=pl.Object) if stringify else pl.lit(None)
        )
        return pl.when(col.is_null()).then(null_value).otherwise(mapped)
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
                f"{_context(rule)}Missing value encountered and "
                "missing_strategy='error'."
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
                raise BoundaryError(
                    f"{_context(rule)}Value out of range and "
                    "boundary_strategy='error'."
                )
    if rule.kind == "categorical" and rule.unknown_category_strategy == "error":
        mapping = rule.category_mapping or {}
        known = set(mapping.keys())
        seen = series.drop_nulls().cast(pl.Utf8).unique().to_list()
        unknown = [value for value in seen if value not in known]
        if unknown:
            from pickbuckets.exceptions import UnknownCategoryError

            raise UnknownCategoryError(
                f"{_context(rule)}Unknown category: {unknown[0]!r}"
            )


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
            f"{_context(rule)}Rules using an 'error' strategy "
            "(missing/boundary/unknown) require eager evaluation; collect the "
            "LazyFrame or use a non-raising strategy."
        )
