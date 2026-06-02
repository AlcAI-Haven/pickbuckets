"""scikit-learn compatible transformers.

These wrap the dependency-free core bucketers as ``BaseEstimator`` /
``TransformerMixin`` estimators so they drop into ``sklearn.pipeline.Pipeline``.
Each estimator applies its strategy column-wise to a 2-D ``X`` and returns a
NumPy array of integer bin codes (what downstream estimators expect). The
fitted, portable :class:`pickbuckets.Rule` objects are exposed on ``rules_``.

Importing this module requires scikit-learn; ``import pickbuckets`` does not
import it, so the core stays dependency-light.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from pickbuckets.core._utils import (
    validate_boundary_strategy,
    validate_min_frequency,
    validate_missing_strategy,
    validate_unknown_category_strategy,
)
from pickbuckets.core.categorical import RareCategoryBucket as _CoreRare
from pickbuckets.core.custom_edges import CustomBoundaryBucket as _CoreCustom
from pickbuckets.core.equal_frequency import EqualFrequencyBucket as _CoreEqFreq
from pickbuckets.core.equal_width import EqualWidthBucket as _CoreEqWidth
from pickbuckets.exceptions import InvalidBucketingError, NotFittedError
from pickbuckets.rules import Rule
from pickbuckets.rules.schema import (
    BoundaryStrategy,
    MissingStrategy,
    UnknownCategoryStrategy,
)
from pickbuckets.runtime import apply_rule


def _as_columns(X: Any) -> tuple[list[list[Any]], list[str]]:
    """Return ``(columns, feature_names)`` from a DataFrame or 2-D array."""

    columns_attr = getattr(X, "columns", None)
    if columns_attr is not None:  # pandas DataFrame
        names = [str(name) for name in columns_attr]
        return [X[name].tolist() for name in columns_attr], names

    array = np.asarray(X, dtype=object)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    n_features = array.shape[1]
    columns = [array[:, index].tolist() for index in range(n_features)]
    names = [f"x{index}" for index in range(n_features)]
    return columns, names


def _ordinal_classes(rule: Rule) -> list[Any]:
    classes: list[Any] = list(rule.labels)
    if (
        rule.missing_strategy in {"separate", "most_frequent"}
        and rule.missing_label not in classes
    ):
        classes.append(rule.missing_label)
    if rule.boundary_strategy == "underflow_overflow":
        for label in (rule.underflow_label, rule.overflow_label):
            if label not in classes:
                classes.append(label)
    if (
        rule.kind == "categorical"
        and rule.unknown_category_strategy == "other"
        and rule.unknown_label not in classes
    ):
        classes.append(rule.unknown_label)
    if (
        rule.kind == "categorical"
        and rule.unknown_category_strategy == "missing"
        and rule.missing_label not in classes
    ):
        classes.append(rule.missing_label)
    return classes


def _encode(rule: Rule, values: list[Any]) -> list[int]:
    code = {label: index for index, label in enumerate(_ordinal_classes(rule))}
    applied = apply_rule(rule, values)
    return [code.get(label, len(code)) for label in applied]


def _validate_feature_names(expected: Any, received: list[str]) -> None:
    expected_list = [str(name) for name in expected]
    if received != expected_list:
        raise ValueError(
            f"Feature names must match fit order; expected {expected_list!r}, "
            f"got {received!r}."
        )


def _validate_sklearn_unknown_strategy(strategy: UnknownCategoryStrategy) -> None:
    validate_unknown_category_strategy(strategy)
    if strategy == "keep":
        raise InvalidBucketingError(
            "unknown_category_strategy='keep' is not supported by sklearn "
            "adapters because they must return stable integer code arrays."
        )


class _BaseSklearnBucket(TransformerMixin, BaseEstimator):
    """Column-wise sklearn wrapper around a core bucketer factory."""

    def _make_core(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X: Any, y: Any = None) -> _BaseSklearnBucket:
        columns, names = _as_columns(X)
        rules: list[Rule] = []
        for name, column in zip(names, columns):
            core = self._make_core()
            core.fit(column)
            rule = core.rules_
            if rule.feature_name is None:
                payload = rule.to_dict()
                payload["feature_name"] = name
                rule = Rule.from_dict(payload)
            rules.append(rule)
        self.rules_ = rules
        self.n_features_in_ = len(names)
        self.feature_names_in_ = np.asarray(names, dtype=object)
        return self

    def transform(self, X: Any) -> Any:
        if not hasattr(self, "rules_"):
            raise NotFittedError("This estimator is not fitted yet.")
        columns, names = _as_columns(X)
        if len(columns) != self.n_features_in_:
            raise ValueError(
                f"X has {len(columns)} features, expected {self.n_features_in_}."
            )
        _validate_feature_names(self.feature_names_in_, names)
        encoded = [_encode(rule, column) for rule, column in zip(self.rules_, columns)]
        return np.asarray(encoded, dtype=float).T

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        if not hasattr(self, "rules_"):
            raise NotFittedError("This estimator is not fitted yet.")
        if input_features is not None:
            names = [str(name) for name in input_features]
            _validate_feature_names(self.feature_names_in_, names)
            return np.asarray(names, dtype=object)
        return np.asarray(
            [str(rule.feature_name) for rule in self.rules_], dtype=object
        )


class EqualWidthBucket(_BaseSklearnBucket):
    def __init__(
        self,
        n_bins: int = 5,
        *,
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
    ) -> None:
        validate_missing_strategy(missing_strategy)
        validate_boundary_strategy(boundary_strategy)
        self.n_bins = n_bins
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label

    def _make_core(self) -> Any:
        return _CoreEqWidth(
            n_bins=self.n_bins,
            labels="ordinal",
            missing_strategy=self.missing_strategy,
            missing_label=self.missing_label,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
        )


class EqualFrequencyBucket(_BaseSklearnBucket):
    def __init__(
        self,
        n_bins: int = 5,
        duplicates: str = "drop",
        *,
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
    ) -> None:
        validate_missing_strategy(missing_strategy)
        validate_boundary_strategy(boundary_strategy)
        self.n_bins = n_bins
        self.duplicates = duplicates
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label

    def _make_core(self) -> Any:
        return _CoreEqFreq(
            n_bins=self.n_bins,
            labels="ordinal",
            duplicates=self.duplicates,
            missing_strategy=self.missing_strategy,
            missing_label=self.missing_label,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
        )


class CustomBoundaryBucket(_BaseSklearnBucket):
    def __init__(
        self,
        edges: Any,
        *,
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
    ) -> None:
        validate_missing_strategy(missing_strategy)
        validate_boundary_strategy(boundary_strategy)
        self.edges = edges
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label

    def _make_core(self) -> Any:
        return _CoreCustom(
            edges=self.edges,
            labels="ordinal",
            missing_strategy=self.missing_strategy,
            missing_label=self.missing_label,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
        )


class RareCategoryBucket(_BaseSklearnBucket):
    def __init__(
        self,
        min_frequency: int | float = 0.01,
        *,
        other_label: Any = "Other",
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        unknown_category_strategy: UnknownCategoryStrategy = "other",
    ) -> None:
        validate_min_frequency(min_frequency)
        validate_missing_strategy(missing_strategy)
        _validate_sklearn_unknown_strategy(unknown_category_strategy)
        self.min_frequency = min_frequency
        self.other_label = other_label
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.unknown_category_strategy = unknown_category_strategy

    def _make_core(self) -> Any:
        return _CoreRare(
            min_frequency=self.min_frequency,
            other_label=self.other_label,
            missing_strategy=self.missing_strategy,
            missing_label=self.missing_label,
            unknown_category_strategy=self.unknown_category_strategy,
        )


class AutoBucket(TransformerMixin, BaseEstimator):
    """Per-column type dispatch as an sklearn transformer (integer codes out)."""

    def __init__(
        self,
        *,
        numeric_strategy: str = "quantile",
        n_bins: int = 5,
        min_frequency: int | float = 0.01,
        duplicates: str = "drop",
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
        unknown_category_strategy: UnknownCategoryStrategy = "other",
        other_label: Any = "Other",
    ) -> None:
        if numeric_strategy not in {"quantile", "width"}:
            raise InvalidBucketingError(
                "numeric_strategy must be 'quantile' or 'width'."
            )
        validate_missing_strategy(missing_strategy)
        validate_boundary_strategy(boundary_strategy)
        _validate_sklearn_unknown_strategy(unknown_category_strategy)
        validate_min_frequency(min_frequency)
        self.numeric_strategy = numeric_strategy
        self.n_bins = n_bins
        self.min_frequency = min_frequency
        self.duplicates = duplicates
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label
        self.unknown_category_strategy = unknown_category_strategy
        self.other_label = other_label

    def _is_numeric(self, column: list[Any]) -> bool:
        from pickbuckets.runtime.apply import is_missing

        seen = [value for value in column if not is_missing(value)]
        if not seen:
            return False
        if all(isinstance(value, bool) for value in seen):
            return False
        return all(isinstance(value, (int, float)) for value in seen)

    def fit(self, X: Any, y: Any = None) -> AutoBucket:
        columns, names = _as_columns(X)
        rules: list[Rule] = []
        for name, column in zip(names, columns):
            if self._is_numeric(column):
                if self.numeric_strategy == "width":
                    core: Any = _CoreEqWidth(
                        n_bins=self.n_bins,
                        labels="ordinal",
                        missing_strategy=self.missing_strategy,
                        missing_label=self.missing_label,
                        boundary_strategy=self.boundary_strategy,
                        underflow_label=self.underflow_label,
                        overflow_label=self.overflow_label,
                    )
                else:
                    core = _CoreEqFreq(
                        n_bins=self.n_bins,
                        labels="ordinal",
                        duplicates=self.duplicates,
                        missing_strategy=self.missing_strategy,
                        missing_label=self.missing_label,
                        boundary_strategy=self.boundary_strategy,
                        underflow_label=self.underflow_label,
                        overflow_label=self.overflow_label,
                    )
            else:
                core = _CoreRare(
                    min_frequency=self.min_frequency,
                    other_label=self.other_label,
                    missing_strategy=self.missing_strategy,
                    missing_label=self.missing_label,
                    unknown_category_strategy=self.unknown_category_strategy,
                )
            core.fit(column)
            rule = core.rules_
            payload = rule.to_dict()
            payload["feature_name"] = name
            rules.append(Rule.from_dict(payload))
        self.rules_ = rules
        self.n_features_in_ = len(names)
        self.feature_names_in_ = np.asarray(names, dtype=object)
        return self

    def transform(self, X: Any) -> Any:
        if not hasattr(self, "rules_"):
            raise NotFittedError("This estimator is not fitted yet.")
        columns, names = _as_columns(X)
        if len(columns) != self.n_features_in_:
            raise ValueError(
                f"X has {len(columns)} features, expected {self.n_features_in_}."
            )
        _validate_feature_names(self.feature_names_in_, names)
        encoded = [_encode(rule, column) for rule, column in zip(self.rules_, columns)]
        return np.asarray(encoded, dtype=float).T

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        if not hasattr(self, "rules_"):
            raise NotFittedError("This estimator is not fitted yet.")
        if input_features is not None:
            names = [str(name) for name in input_features]
            _validate_feature_names(self.feature_names_in_, names)
            return np.asarray(names, dtype=object)
        return np.asarray(
            [str(rule.feature_name) for rule in self.rules_], dtype=object
        )


__all__ = [
    "AutoBucket",
    "CustomBoundaryBucket",
    "EqualFrequencyBucket",
    "EqualWidthBucket",
    "RareCategoryBucket",
]
