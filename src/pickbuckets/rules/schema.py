from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import isinf, isnan
from typing import Any, ClassVar, Literal

from pickbuckets._version import __version__
from pickbuckets.exceptions import RuleSchemaError

RuleKind = Literal["numeric", "categorical"]
ClosedSide = Literal["left"]
MissingStrategy = Literal["separate", "most_frequent", "error", "propagate"]
BoundaryStrategy = Literal["clip", "underflow_overflow", "error"]
UnknownCategoryStrategy = Literal["other", "missing", "error", "keep"]


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
    underflow_label: Any = "Underflow"
    overflow_label: Any = "Overflow"
    unknown_category_strategy: UnknownCategoryStrategy = "other"
    unknown_label: Any = "Other"
    fit_stats: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.1"
    package_version: str = __version__

    CURRENT_SCHEMA_VERSION: ClassVar[str] = "1.1"
    SUPPORTED_SCHEMA_MAJOR: ClassVar[str] = "1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, str)
            or self.kind not in {"numeric", "categorical"}
        ):
            raise RuleSchemaError(f"Unsupported rule kind: {self.kind!r}")
        if not isinstance(self.closed, str) or self.closed != "left":
            raise RuleSchemaError(f"Unsupported closed-side policy: {self.closed!r}")
        if (
            not isinstance(self.missing_strategy, str)
            or self.missing_strategy
            not in {"separate", "most_frequent", "error", "propagate"}
        ):
            raise RuleSchemaError(
                f"Unsupported missing strategy: {self.missing_strategy!r}"
            )
        if (
            not isinstance(self.boundary_strategy, str)
            or self.boundary_strategy not in {"clip", "underflow_overflow", "error"}
        ):
            raise RuleSchemaError(
                f"Unsupported boundary strategy: {self.boundary_strategy!r}"
            )
        if (
            not isinstance(self.unknown_category_strategy, str)
            or self.unknown_category_strategy
            not in {"other", "missing", "error", "keep"}
        ):
            raise RuleSchemaError(
                "Unsupported unknown-category strategy: "
                f"{self.unknown_category_strategy!r}"
            )
        if self.kind == "numeric":
            edges = _normalize_edges(self.edges)
            object.__setattr__(self, "edges", edges)
            if len(self.labels) != len(edges) - 1:
                raise RuleSchemaError("Numeric label count must equal interval count.")
        if self.kind == "categorical":
            if self.category_mapping is None:
                raise RuleSchemaError("Categorical rules require a category mapping.")
            mapping = _normalize_category_mapping(self.category_mapping)
            object.__setattr__(self, "category_mapping", mapping)
            if not self.labels:
                raise RuleSchemaError("Categorical rules require at least one label.")
            for category, label in mapping.items():
                if label not in self.labels:
                    raise RuleSchemaError(
                        f"Categorical mapping for {category!r} uses label "
                        f"{label!r}, which is not present in labels."
                    )
            if (
                self.unknown_category_strategy == "other"
                and self.unknown_label not in self.labels
            ):
                raise RuleSchemaError(
                    "unknown_label must be present in labels when "
                    "unknown_category_strategy='other'."
                )
        possible_missing_labels = list(self.labels)
        if self.kind == "numeric" and self.boundary_strategy == "underflow_overflow":
            possible_missing_labels.extend(
                [self.underflow_label, self.overflow_label]
            )
        if (
            self.missing_strategy == "most_frequent"
            and self.missing_label not in possible_missing_labels
        ):
            raise RuleSchemaError(
                "missing_strategy='most_frequent' requires missing_label to be one "
                "of the fitted labels."
            )

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
            "edges": _encode_edges(self.edges),
            "category_mapping": self.category_mapping,
            "labels": self.labels,
            "closed": self.closed,
            "missing_strategy": self.missing_strategy,
            "missing_label": self.missing_label,
            "boundary_strategy": self.boundary_strategy,
            "underflow_label": self.underflow_label,
            "overflow_label": self.overflow_label,
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
            underflow_label=data.get("underflow_label", "Underflow"),
            overflow_label=data.get("overflow_label", "Overflow"),
            unknown_category_strategy=data.get("unknown_category_strategy", "other"),
            unknown_label=data.get("unknown_label", "Other"),
            fit_stats=data.get("fit_stats", {}),
            schema_version=cls.CURRENT_SCHEMA_VERSION,
            package_version=data.get("package_version", __version__),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Rule:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise RuleSchemaError("Rule JSON payload must decode to an object.")
        return cls.from_dict(data)


def _normalize_edges(edges: list[Any] | None) -> list[float]:
    if edges is None or len(edges) < 2:
        raise RuleSchemaError("Numeric rules require at least two edges.")

    normalized: list[float] = []
    for edge in edges:
        try:
            number = _decode_edge(edge)
        except (TypeError, ValueError) as exc:
            raise RuleSchemaError(f"Invalid numeric edge: {edge!r}") from exc
        if isnan(number):
            raise RuleSchemaError("Numeric edges cannot be NaN.")
        normalized.append(number)

    if len(normalized) == 2:
        if normalized[0] > normalized[1]:
            raise RuleSchemaError("Numeric edges must be sorted.")
    elif any(left >= right for left, right in zip(normalized, normalized[1:])):
        raise RuleSchemaError("Numeric edges must be sorted and unique.")

    return normalized


def _normalize_category_mapping(mapping: dict[Any, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        text = str(key)
        if text in normalized and normalized[text] != value:
            raise RuleSchemaError(
                f"Category key {text!r} is duplicated after string conversion."
            )
        normalized[text] = value
    return normalized


def _decode_edge(edge: Any) -> float:
    if edge == "inf":
        return float("inf")
    if edge == "-inf":
        return float("-inf")
    return float(edge)


def _encode_edges(edges: list[float] | None) -> list[Any] | None:
    if edges is None:
        return None
    result: list[Any] = []
    for edge in edges:
        if isinf(edge):
            result.append("inf" if edge > 0 else "-inf")
        else:
            result.append(edge)
    return result
