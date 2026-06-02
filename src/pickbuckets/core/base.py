from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pickbuckets.adapters.detect import Container, adapt_1d, detect_container
from pickbuckets.exceptions import NotFittedError
from pickbuckets.rules import Rule
from pickbuckets.runtime import apply_rule


class BaseBucket:
    """Shared behaviour for single-column bucketers.

    Subclasses implement :meth:`fit` and assign the learned :attr:`rules_`.
    Transform accepts plain iterables, NumPy arrays, and pandas/Polars Series;
    the output container matches the input. Polars Series use the vectorized
    expression path. The optional libraries are only touched when the caller
    actually passes one of their objects, so ``import pickbuckets`` stays light.
    """

    rules_: Rule

    def fit(self, values: Iterable[Any]) -> BaseBucket:
        raise NotImplementedError

    def transform(self, values: Iterable[Any]) -> Any:
        if not hasattr(self, "rules_"):
            raise NotFittedError("This bucketer must be fitted before transform().")

        kind = detect_container(values)
        if kind is Container.POLARS_SERIES:
            from pickbuckets.adapters.polars_io import apply_rule_polars

            return apply_rule_polars(self.rules_, values)

        plain, rebuild = adapt_1d(values)
        return rebuild(apply_rule(self.rules_, plain))

    def fit_transform(self, values: Iterable[Any]) -> Any:
        self.fit(values)
        return self.transform(values)

    @property
    def n_features_in_(self) -> int:
        if not hasattr(self, "rules_"):
            raise NotFittedError("This bucketer is not fitted yet.")
        return 1

    @property
    def feature_names_in_(self) -> list[Any] | None:
        if not hasattr(self, "rules_"):
            raise NotFittedError("This bucketer is not fitted yet.")
        name = self.rules_.feature_name
        return [name] if name is not None else None

    def summary(self) -> dict[str, Any]:
        """Human-readable view of the fitted rule and active policies."""

        if not hasattr(self, "rules_"):
            raise NotFittedError("This bucketer must be fitted before summary().")
        rule = self.rules_
        info: dict[str, Any] = {
            "feature_name": rule.feature_name,
            "kind": rule.kind,
            "labels": list(rule.labels),
            "missing_strategy": rule.missing_strategy,
            "fit_stats": dict(rule.fit_stats),
        }
        if rule.kind == "numeric":
            info["n_bins"] = len(rule.labels)
            info["edges"] = list(rule.edges or [])
            info["boundary_strategy"] = rule.boundary_strategy
        else:
            info["n_categories"] = len(rule.category_mapping or {})
            info["unknown_category_strategy"] = rule.unknown_category_strategy
        return info

    def to_dict(self) -> dict[str, Any]:
        if not hasattr(self, "rules_"):
            raise NotFittedError("This bucketer must be fitted before serialization.")
        return self.rules_.to_dict()

    def to_json(self) -> str:
        if not hasattr(self, "rules_"):
            raise NotFittedError("This bucketer must be fitted before serialization.")
        return self.rules_.to_json()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseBucket:
        obj = cls.__new__(cls)
        obj.rules_ = Rule.from_dict(data)
        return obj

    @classmethod
    def from_json(cls, payload: str) -> BaseBucket:
        obj = cls.__new__(cls)
        obj.rules_ = Rule.from_json(payload)
        return obj
