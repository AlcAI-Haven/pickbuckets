from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pickbuckets.exceptions import NotFittedError
from pickbuckets.rules import Rule
from pickbuckets.runtime import apply_rule


class BaseBucket:
    rules_: Rule

    def transform(self, values: Iterable[Any]) -> list[Any]:
        if not hasattr(self, "rules_"):
            raise NotFittedError("This bucketer must be fitted before transform().")
        return apply_rule(self.rules_, values)

    def fit_transform(self, values: Iterable[Any]) -> list[Any]:
        self.fit(values)
        return self.transform(values)

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

