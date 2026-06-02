"""AutoBucket: column-wise bucketing with smart per-column type dispatch."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from pickbuckets.adapters.frames import (
    FrameKind,
    build_dataframe,
    column_kind,
    column_names,
    column_values,
    pandas_index,
    require_frame,
)
from pickbuckets.core._utils import (
    validate_boundary_strategy,
    validate_min_frequency,
    validate_missing_strategy,
    validate_unknown_category_strategy,
)
from pickbuckets.core.base import BaseBucket
from pickbuckets.core.categorical import RareCategoryBucket
from pickbuckets.core.equal_frequency import EqualFrequencyBucket
from pickbuckets.core.equal_width import EqualWidthBucket
from pickbuckets.exceptions import InvalidBucketingError, NotFittedError
from pickbuckets.rules import Rule
from pickbuckets.rules.schema import (
    BoundaryStrategy,
    MissingStrategy,
    UnknownCategoryStrategy,
)
from pickbuckets.runtime import apply_rule


class AutoBucket:
    """Fit a bucketing rule per column, choosing the strategy by dtype.

    Numeric columns are bucketed with equal-width or equal-frequency binning;
    string/categorical columns are grouped with rare-category folding. Explicit
    per-column ``overrides`` (an unfitted bucketer instance) always win over
    inference. Accepts pandas/Polars DataFrames, Polars LazyFrames, and plain
    ``{name: values}`` mappings, returning the same container type.
    """

    def __init__(
        self,
        *,
        numeric_strategy: str = "quantile",
        categorical_strategy: str = "rare",
        n_bins: int = 5,
        min_frequency: int | float = 0.01,
        duplicates: str = "drop",
        labels: str = "ordinal",
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
        unknown_category_strategy: UnknownCategoryStrategy = "other",
        other_label: Any = "Other",
        overrides: dict[Hashable, BaseBucket] | None = None,
        ignore_unsupported: bool = False,
    ) -> None:
        if numeric_strategy not in {"quantile", "width"}:
            raise InvalidBucketingError(
                "numeric_strategy must be 'quantile' or 'width'."
            )
        if categorical_strategy not in {"rare"}:
            raise InvalidBucketingError("categorical_strategy must be 'rare'.")
        validate_min_frequency(min_frequency)
        validate_missing_strategy(missing_strategy)
        validate_boundary_strategy(boundary_strategy)
        validate_unknown_category_strategy(unknown_category_strategy)
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.n_bins = n_bins
        self.min_frequency = min_frequency
        self.duplicates = duplicates
        self.labels = labels
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label
        self.unknown_category_strategy = unknown_category_strategy
        self.other_label = other_label
        self.overrides = overrides
        self.ignore_unsupported = ignore_unsupported

    # -- fitting ---------------------------------------------------------

    def _make_numeric(self) -> BaseBucket:
        if self.numeric_strategy == "width":
            return EqualWidthBucket(
                n_bins=self.n_bins,
                labels=self.labels,
                missing_strategy=self.missing_strategy,
                missing_label=self.missing_label,
                boundary_strategy=self.boundary_strategy,
                underflow_label=self.underflow_label,
                overflow_label=self.overflow_label,
            )
        return EqualFrequencyBucket(
            n_bins=self.n_bins,
            labels=self.labels,
            duplicates=self.duplicates,
            missing_strategy=self.missing_strategy,
            missing_label=self.missing_label,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
        )

    def _make_categorical(self) -> BaseBucket:
        return RareCategoryBucket(
            min_frequency=self.min_frequency,
            other_label=self.other_label,
            missing_strategy=self.missing_strategy,
            missing_label=self.missing_label,
            unknown_category_strategy=self.unknown_category_strategy,
        )

    def fit(self, frame: Any, y: Any = None) -> AutoBucket:
        kind = require_frame(frame)
        names = column_names(frame, kind)
        _validate_column_names(names)
        overrides = self.overrides or {}
        _validate_overrides(overrides, names)

        rules: dict[Hashable, Rule] = {}
        skipped: list[Hashable] = []
        for name in names:
            if name in overrides:
                bucketer = overrides[name]
                if not isinstance(bucketer, BaseBucket):
                    raise InvalidBucketingError(
                        f"Override for column {name!r} must be a pickbuckets "
                        "bucketer."
                    )
            else:
                col_kind = column_kind(frame, kind, name)
                if col_kind == "numeric":
                    bucketer = self._make_numeric()
                elif col_kind == "categorical":
                    bucketer = self._make_categorical()
                else:
                    if self.ignore_unsupported:
                        skipped.append(name)
                        continue
                    raise InvalidBucketingError(
                        f"Column {name!r} has an unsupported dtype; pass "
                        "ignore_unsupported=True to pass it through unchanged."
                    )
            values = column_values(frame, kind, name)
            bucketer.fit(values)
            rule = bucketer.rules_
            if rule.feature_name is None:
                rule = _with_feature_name(rule, str(name))
            rules[name] = rule

        self.rules_ = rules
        self.feature_names_in_ = list(names)
        self.n_features_in_ = len(names)
        self.skipped_columns_ = skipped
        return self

    # -- transforming ----------------------------------------------------

    def transform(self, frame: Any) -> Any:
        if not hasattr(self, "rules_"):
            raise NotFittedError("AutoBucket must be fitted before transform().")
        kind = require_frame(frame)
        names = column_names(frame, kind)
        if len(set(names)) != len(names):
            raise InvalidBucketingError("AutoBucket requires unique column names.")
        _validate_transform_columns(self.feature_names_in_, names)

        if kind in (FrameKind.POLARS, FrameKind.POLARS_LAZY):
            return self._transform_polars(frame, kind, names)

        index = pandas_index(frame) if kind is FrameKind.PANDAS else None
        transformed: dict[Hashable, list[Any]] = {}
        for name in names:
            values = column_values(frame, kind, name)
            if name in self.rules_:
                transformed[name] = apply_rule(self.rules_[name], values)
            else:
                transformed[name] = values
        return build_dataframe(kind, names, transformed, index=index)

    def _transform_polars(
        self, frame: Any, kind: FrameKind, names: list[Hashable]
    ) -> Any:
        import polars as pl

        from pickbuckets.adapters.polars_io import (
            lazy_guard,
            precheck_series,
            rule_expr,
        )

        exprs = []
        for name in names:
            if name in self.rules_:
                rule = self.rules_[name]
                if kind is FrameKind.POLARS_LAZY:
                    lazy_guard(rule)
                    exprs.append(rule_expr(rule, pl.col(name)).alias(name))
                else:
                    precheck_series(rule, frame.get_column(name))
                    exprs.append(
                        rule_expr(rule, pl.col(name), cast_numeric=True).alias(name)
                    )
        if not exprs:
            return frame
        return frame.with_columns(exprs)

    def fit_transform(self, frame: Any, y: Any = None) -> Any:
        return self.fit(frame, y).transform(frame)

    # -- inspection ------------------------------------------------------

    def get_feature_names_out(self, input_features: Any = None) -> list[Hashable]:
        if not hasattr(self, "rules_"):
            raise NotFittedError("AutoBucket must be fitted first.")
        return list(self.feature_names_in_)

    def summary(self) -> dict[Hashable, dict[str, Any]]:
        if not hasattr(self, "rules_"):
            raise NotFittedError("AutoBucket must be fitted before summary().")
        out: dict[Hashable, dict[str, Any]] = {}
        for name, rule in self.rules_.items():
            bucket = BaseBucket.__new__(BaseBucket)
            bucket.rules_ = rule
            out[name] = bucket.summary()
        return out

    def to_dict(self) -> dict[str, Any]:
        if not hasattr(self, "rules_"):
            raise NotFittedError("AutoBucket must be fitted before serialization.")
        return {
            "auto_schema_version": "1.0",
            "feature_names_in": list(self.feature_names_in_),
            "skipped_columns": list(self.skipped_columns_),
            "rules": {name: rule.to_dict() for name, rule in self.rules_.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutoBucket:
        obj = cls.__new__(cls)
        obj.rules_ = {
            name: Rule.from_dict(payload)
            for name, payload in data["rules"].items()
        }
        obj.feature_names_in_ = list(data.get("feature_names_in", obj.rules_.keys()))
        obj.n_features_in_ = len(obj.feature_names_in_)
        obj.skipped_columns_ = list(data.get("skipped_columns", []))
        return obj


def _with_feature_name(rule: Rule, name: str) -> Rule:
    payload = rule.to_dict()
    payload["feature_name"] = name
    return Rule.from_dict(payload)


def _validate_column_names(names: list[Hashable]) -> None:
    if not names:
        raise InvalidBucketingError("AutoBucket requires at least one column.")
    if len(set(names)) != len(names):
        raise InvalidBucketingError("AutoBucket requires unique column names.")


def _validate_transform_columns(
    expected: list[Hashable],
    received: list[Hashable],
) -> None:
    expected_set = set(expected)
    received_set = set(received)
    missing = [name for name in expected if name not in received_set]
    extra = [name for name in received if name not in expected_set]
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing columns: {missing!r}")
        if extra:
            details.append(f"unexpected columns: {extra!r}")
        raise InvalidBucketingError(
            "AutoBucket transform columns must match fit columns ("
            + "; ".join(details)
            + ")."
        )


def _validate_overrides(
    overrides: dict[Hashable, BaseBucket],
    names: list[Hashable],
) -> None:
    unknown = [name for name in overrides if name not in set(names)]
    if unknown:
        raise InvalidBucketingError(
            f"Overrides refer to unknown columns: {unknown!r}."
        )
