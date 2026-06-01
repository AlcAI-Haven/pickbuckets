from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pickbuckets._version import __version__
from pickbuckets.exceptions import RuleSchemaError

RuleKind = Literal["numeric", "categorical"]
ClosedSide = Literal["left"]
MissingStrategy = Literal["separate", "error", "propagate"]
BoundaryStrategy = Literal["clip", "error"]
UnknownCategoryStrategy = Literal["other", "error"]


@dataclass(frozen=True)
class Rule:
    """Unified serializable model for numeric and categorical bucket rules."""

    kind: RuleKind
    labels: list[Any]
    feature_name: str | None = None
    edges: list[float] | None = None
    category_mapping: dict[str, Any] | None = None
    closed: ClosedSide = "left"
    missing_strategy: MissingStrategy = "separate"
    missing_label: Any = "Missing"
    boundary_strategy: BoundaryStrategy = "clip"
    unknown_category_strategy: UnknownCategoryStrategy = "other"
    unknown_label: Any = "Other"
    fit_stats: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    package_version: str = __version__

    SUPPORTED_SCHEMA_MAJOR: ClassVar[str] = "1"

    def __post_init__(self) -> None:
        if self.kind not in {"numeric", "categorical"}:
            raise RuleSchemaError(f"Unsupported rule kind: {self.kind!r}")
        if self.kind == "numeric":
            if self.edges is None or len(self.edges) < 2:
                raise RuleSchemaError("Numeric rules require at least two edges.")
            if len(self.labels) != len(self.edges) - 1:
                raise RuleSchemaError("Numeric label count must equal interval count.")
        if self.kind == "categorical":
            if self.category_mapping is None:
                raise RuleSchemaError("Categorical rules require a category mapping.")

    def __repr__(self) -> str:
        name = f" feature={self.feature_name!r}" if self.feature_name else ""
        if self.kind == "numeric":
            return (
                f"Rule(kind='numeric'{name}, bins={len(self.labels)}, "
                f"schema_version={self.schema_version!r})"
            )
        return (
            f"Rule(kind='categorical'{name}, categories="
            f"{len(self.category_mapping or {})}, "
            f"schema_version={self.schema_version!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "kind": self.kind,
            "feature_name": self.feature_name,
            "edges": self.edges,
            "category_mapping": self.category_mapping,
            "labels": self.labels,
            "closed": self.closed,
            "missing_strategy": self.missing_strategy,
            "missing_label": self.missing_label,
            "boundary_strategy": self.boundary_strategy,
            "unknown_category_strategy": self.unknown_category_strategy,
            "unknown_label": self.unknown_label,
            "fit_stats": self.fit_stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rule:
        schema_version = str(data.get("schema_version", ""))
        major = schema_version.split(".", maxsplit=1)[0]
        if major != cls.SUPPORTED_SCHEMA_MAJOR:
            raise RuleSchemaError(
                f"Unsupported rule schema version {schema_version!r}; "
                f"expected major version {cls.SUPPORTED_SCHEMA_MAJOR}."
            )
        return cls(
            kind=data["kind"],
            feature_name=data.get("feature_name"),
            edges=data.get("edges"),
            category_mapping=data.get("category_mapping"),
            labels=data["labels"],
            closed=data.get("closed", "left"),
            missing_strategy=data.get("missing_strategy", "separate"),
            missing_label=data.get("missing_label", "Missing"),
            boundary_strategy=data.get("boundary_strategy", "clip"),
            unknown_category_strategy=data.get("unknown_category_strategy", "other"),
            unknown_label=data.get("unknown_label", "Other"),
            fit_stats=data.get("fit_stats", {}),
            schema_version=schema_version,
            package_version=data.get("package_version", __version__),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Rule:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise RuleSchemaError("Rule JSON payload must decode to an object.")
        return cls.from_dict(data)
