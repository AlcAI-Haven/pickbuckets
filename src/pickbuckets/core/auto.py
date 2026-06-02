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
from pickbuckets.core.base import BaseBucket
from pickbuckets.core.categorical import RareCategoryBucket
from pickbuckets.core.equal_frequency import EqualFrequencyBucket
from pickbuckets.core.equal_width import EqualWidthBucket
from pickbuckets.exceptions import InvalidBucketingError, NotFittedError
from pickbuckets.rules import Rule
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
        overrides: dict[Hashable, BaseBucket] | None = None,
        ignore_unsupported: bool = False,
    ) -> None:
        if numeric_strategy not in {"quantile", "width"}:
            raise InvalidBucketingError(
                "numeric_strategy must be 'quantile' or 'width'."
            )
        if categorical_strategy not in {"rare"}:
            raise InvalidBucketingError("categorical_strategy must be 'rare'.")
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.n_bins = n_bins
        self.min_frequency = min_frequency
        self.duplicates = duplicates
        self.labels = labels
        self.overrides = overrides
        self.ignore_unsupported = ignore_unsupported

    # -- fitting ---------------------------------------------------------

    def _make_numeric(self) -> BaseBucket:
        if self.numeric_strategy == "width":
            return EqualWidthBucket(n_bins=self.n_bins, labels=self.labels)
        return EqualFrequencyBucket(
            n_bins=self.n_bins, labels=self.labels, duplicates=self.duplicates
        )

    def _make_categorical(self) -> BaseBucket:
        return RareCategoryBucket(min_frequency=self.min_frequency)

    def fit(self, frame: Any, y: Any = None) -> AutoBucket:
        kind = require_frame(frame)
        names = column_names(frame, kind)
        overrides = self.overrides or {}

        rules: dict[Hashable, Rule] = {}
        skipped: list[Hashable] = []
        for name in names:
            if name in overrides:
                bucketer = overrides[name]
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
                else:
                    precheck_series(rule, frame.get_column(name))
                exprs.append(rule_expr(rule, pl.col(name)).alias(name))
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
