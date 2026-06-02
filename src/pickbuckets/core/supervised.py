from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import ceil, isfinite, log
from typing import Any

from pickbuckets.core._utils import (
    make_labels,
    missing_count,
    most_frequent_label,
    numeric_bin_counts,
    validate_boundary_strategy,
    validate_missing_strategy,
    validate_n_bins,
)
from pickbuckets.core.base import BaseBucket
from pickbuckets.exceptions import InvalidBucketingError
from pickbuckets.rules import Rule
from pickbuckets.rules.schema import BoundaryStrategy, MissingStrategy
from pickbuckets.runtime.apply import is_missing


@dataclass
class _Bin:
    left: float
    right: float
    values: list[float]
    events: list[int]


class SupervisedBucket(BaseBucket):
    """Base class for single-feature target-aware bucketers."""

    iv_: float
    target_summary_: dict[str, Any]

    def fit(
        self, values: Iterable[Any], y: Iterable[Any] | None = None
    ) -> SupervisedBucket:
        if y is None:
            raise InvalidBucketingError(
                "Supervised bucketers require target values: call fit(X, y)."
            )
        raise NotImplementedError

    def fit_transform(
        self, values: Iterable[Any], y: Iterable[Any] | None = None
    ) -> Any:
        self.fit(values, y)
        return self.transform(values)

    def summary_table(self) -> list[dict[str, Any]]:
        if not hasattr(self, "rules_"):
            from pickbuckets.exceptions import NotFittedError

            raise NotFittedError("This bucketer must be fitted before summary_table().")
        summary = self.rules_.fit_stats.get("bin_summary", [])
        return [dict(row) for row in summary]

    def iv_summary(self) -> dict[str, Any]:
        if not hasattr(self, "rules_"):
            from pickbuckets.exceptions import NotFittedError

            raise NotFittedError("This bucketer must be fitted before iv_summary().")
        return {
            "feature_name": self.rules_.feature_name,
            "iv": self.rules_.fit_stats.get("iv"),
            "algorithm": self.rules_.fit_stats.get("algorithm"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupervisedBucket:
        obj = cls.__new__(cls)
        obj.rules_ = Rule.from_dict(data)
        _restore_supervised_attributes(obj)
        return obj

    @classmethod
    def from_json(cls, payload: str) -> SupervisedBucket:
        obj = cls.__new__(cls)
        obj.rules_ = Rule.from_json(payload)
        _restore_supervised_attributes(obj)
        return obj


class DecisionTreeBucket(SupervisedBucket):
    """Learn numeric bin edges from a shallow sklearn decision tree."""

    def __init__(
        self,
        max_leaf_nodes: int = 5,
        *,
        min_samples_leaf: int | float = 1,
        criterion: str = "gini",
        labels: str | Sequence[Any] = "ordinal",
        feature_name: str | None = None,
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
        random_state: int | None = 0,
    ) -> None:
        _validate_max_leaf_nodes(max_leaf_nodes)
        _validate_min_samples_leaf(min_samples_leaf)
        if criterion not in {"gini", "entropy", "log_loss"}:
            raise InvalidBucketingError(
                "criterion must be 'gini', 'entropy', or 'log_loss'."
            )
        validate_missing_strategy(missing_strategy)
        validate_boundary_strategy(boundary_strategy)
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.labels = labels
        self.feature_name = feature_name
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label
        self.random_state = random_state

    def fit(
        self, values: Iterable[Any], y: Iterable[Any] | None = None
    ) -> DecisionTreeBucket:
        raw_values, clean, targets = _validated_pairs(values, y)
        try:
            from sklearn.tree import DecisionTreeClassifier
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise InvalidBucketingError(
                "DecisionTreeBucket requires scikit-learn; install "
                "pickbuckets[sklearn]."
            ) from exc

        tree = DecisionTreeClassifier(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state,
        )
        tree.fit([[value] for value in clean], targets)
        tree_ = tree.tree_
        thresholds = sorted(
            {
                float(tree_.threshold[index])
                for index in range(tree_.node_count)
                if tree_.children_left[index] != tree_.children_right[index]
            }
        )
        minimum = min(clean)
        maximum = max(clean)
        internal = [
            threshold for threshold in thresholds if minimum < threshold < maximum
        ]
        edges = [minimum, *internal, maximum]
        labels = make_labels(edges, self.labels)
        bin_counts = numeric_bin_counts(
            clean,
            edges,
            labels,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
        )
        missing_label = _resolved_missing_label(
            self.missing_strategy, self.missing_label, bin_counts
        )
        self.rules_ = Rule(
            kind="numeric",
            feature_name=self.feature_name,
            edges=edges,
            labels=labels,
            missing_strategy=self.missing_strategy,
            missing_label=missing_label,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
            fit_stats={
                "algorithm": "decision_tree",
                "n_observations": len(clean),
                "n_missing": missing_count(raw_values),
                "max_leaf_nodes": self.max_leaf_nodes,
                "min_samples_leaf": self.min_samples_leaf,
                "criterion": self.criterion,
                "tree_thresholds": internal,
                "bin_counts": bin_counts,
            },
        )
        return self


class WoEBucket(SupervisedBucket):
    """Binary-target Weight of Evidence and Information Value bucketer."""

    def __init__(
        self,
        n_bins: int = 5,
        *,
        strategy: str = "quantile",
        output: str = "label",
        min_bin_size: int | float = 1,
        monotonic: str | None = None,
        event_label: Any = 1,
        smoothing: float = 0.5,
        labels: str | Sequence[Any] = "ordinal",
        feature_name: str | None = None,
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
    ) -> None:
        validate_n_bins(n_bins)
        if strategy not in {"quantile", "width"}:
            raise InvalidBucketingError("strategy must be 'quantile' or 'width'.")
        if output not in {"label", "woe"}:
            raise InvalidBucketingError("output must be 'label' or 'woe'.")
        _validate_min_bin_size(min_bin_size)
        _validate_monotonic(monotonic)
        _validate_smoothing(smoothing)
        validate_missing_strategy(missing_strategy)
        validate_boundary_strategy(boundary_strategy)
        self.n_bins = n_bins
        self.strategy = strategy
        self.output = output
        self.min_bin_size = min_bin_size
        self.monotonic = monotonic
        self.event_label = event_label
        self.smoothing = float(smoothing)
        self.labels = labels
        self.feature_name = feature_name
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label

    def fit(self, values: Iterable[Any], y: Iterable[Any] | None = None) -> WoEBucket:
        raw_values, clean, targets = _validated_pairs(values, y)
        events = _binary_events(targets, self.event_label)
        min_bin_count = _min_bin_count(self.min_bin_size, len(clean))
        bins = _make_initial_bins(clean, events, self.n_bins, self.strategy)
        size_merge_count = _merge_min_size_bins(bins, min_bin_count)
        monotonic_direction = _resolve_monotonic_direction(
            bins, self.monotonic, self.smoothing
        )
        monotonic_merge_count = _merge_monotonic_bins(
            bins, monotonic_direction, self.smoothing
        )
        edges = [bins[0].left, *[item.right for item in bins]]
        display_labels = make_labels(edges, self.labels)
        summary, iv = _supervised_summary(
            clean, events, edges, display_labels, self.smoothing
        )
        output_labels = (
            [row["woe"] for row in summary] if self.output == "woe" else display_labels
        )
        bin_counts = numeric_bin_counts(
            clean,
            edges,
            output_labels,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
        )
        missing_label = _resolved_missing_label(
            self.missing_strategy, self.missing_label, bin_counts
        )
        self.iv_ = iv
        self.target_summary_ = {
            "event_label": self.event_label,
            "event_count": sum(events),
            "non_event_count": len(events) - sum(events),
        }
        self.rules_ = Rule(
            kind="numeric",
            feature_name=self.feature_name,
            edges=edges,
            labels=output_labels,
            missing_strategy=self.missing_strategy,
            missing_label=missing_label,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
            fit_stats={
                "algorithm": "woe",
                "n_observations": len(clean),
                "n_missing": missing_count(raw_values),
                "requested_bins": self.n_bins,
                "actual_bins": len(edges) - 1,
                "strategy": self.strategy,
                "output": self.output,
                "min_bin_size": self.min_bin_size,
                "min_bin_count": min_bin_count,
                "monotonic": self.monotonic,
                "monotonic_direction": monotonic_direction,
                "merge_count": size_merge_count + monotonic_merge_count,
                "min_bin_size_merge_count": size_merge_count,
                "monotonic_merge_count": monotonic_merge_count,
                "event_label": self.event_label,
                "smoothing": self.smoothing,
                "iv": iv,
                "bin_summary": summary,
                "bin_counts": bin_counts,
            },
        )
        return self


class ChiMergeBucket(SupervisedBucket):
    """Merge adjacent intervals using binary-target Chi-Square similarity."""

    def __init__(
        self,
        max_bins: int = 5,
        *,
        initial_bins: int = 20,
        min_chi2: float | None = None,
        min_bin_size: int = 1,
        event_label: Any = 1,
        smoothing: float = 0.5,
        labels: str | Sequence[Any] = "ordinal",
        feature_name: str | None = None,
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
    ) -> None:
        validate_n_bins(max_bins)
        validate_n_bins(initial_bins)
        if initial_bins < max_bins:
            raise InvalidBucketingError("initial_bins must be >= max_bins.")
        _validate_min_chi2(min_chi2)
        if (
            isinstance(min_bin_size, bool)
            or not isinstance(min_bin_size, int)
            or min_bin_size < 1
        ):
            raise InvalidBucketingError("min_bin_size must be a positive integer.")
        _validate_smoothing(smoothing)
        validate_missing_strategy(missing_strategy)
        validate_boundary_strategy(boundary_strategy)
        self.max_bins = max_bins
        self.initial_bins = initial_bins
        self.min_chi2 = min_chi2
        self.min_bin_size = min_bin_size
        self.event_label = event_label
        self.smoothing = float(smoothing)
        self.labels = labels
        self.feature_name = feature_name
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label

    def fit(
        self, values: Iterable[Any], y: Iterable[Any] | None = None
    ) -> ChiMergeBucket:
        raw_values, clean, targets = _validated_pairs(values, y)
        events = _binary_events(targets, self.event_label)
        bins = _make_initial_bins(clean, events, self.initial_bins)
        merge_count = _merge_chi_bins(
            bins,
            max_bins=self.max_bins,
            min_chi2=self.min_chi2,
            min_bin_size=self.min_bin_size,
        )
        edges = [bins[0].left, *[item.right for item in bins]]
        labels = make_labels(edges, self.labels)
        summary, iv = _supervised_summary(
            clean, events, edges, labels, self.smoothing
        )
        bin_counts = numeric_bin_counts(
            clean,
            edges,
            labels,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
        )
        missing_label = _resolved_missing_label(
            self.missing_strategy, self.missing_label, bin_counts
        )
        self.iv_ = iv
        self.target_summary_ = {
            "event_label": self.event_label,
            "event_count": sum(events),
            "non_event_count": len(events) - sum(events),
        }
        self.rules_ = Rule(
            kind="numeric",
            feature_name=self.feature_name,
            edges=edges,
            labels=labels,
            missing_strategy=self.missing_strategy,
            missing_label=missing_label,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
            fit_stats={
                "algorithm": "chimerge",
                "n_observations": len(clean),
                "n_missing": missing_count(raw_values),
                "initial_bins": self.initial_bins,
                "actual_bins": len(edges) - 1,
                "max_bins": self.max_bins,
                "min_chi2": self.min_chi2,
                "min_bin_size": self.min_bin_size,
                "merge_count": merge_count,
                "event_label": self.event_label,
                "iv": iv,
                "bin_summary": summary,
                "bin_counts": bin_counts,
            },
        )
        return self


class ExternalSplitBucket(BaseBucket):
    """Import externally learned numeric splits into a portable rule.

    Splits are internal thresholds, such as those exposed by OptBinning. The
    fitted rule uses open-ended numeric edges so it can be applied without the
    external solver dependency.
    """

    def __init__(
        self,
        splits: Sequence[float],
        labels: str | Sequence[Any] = "ordinal",
        *,
        feature_name: str | None = None,
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        boundary_strategy: BoundaryStrategy = "clip",
        underflow_label: Any = "Underflow",
        overflow_label: Any = "Overflow",
    ) -> None:
        validate_missing_strategy(missing_strategy)
        validate_boundary_strategy(boundary_strategy)
        self.splits = _validate_splits(splits)
        self.labels = labels
        self.feature_name = feature_name
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.boundary_strategy = boundary_strategy
        self.underflow_label = underflow_label
        self.overflow_label = overflow_label

    def fit(self, values: Iterable[Any] | None = None) -> ExternalSplitBucket:
        edges = [float("-inf"), *self.splits, float("inf")]
        labels = make_labels(edges, self.labels)
        fit_stats: dict[str, Any] = {
            "algorithm": "external_splits",
            "source": "external",
            "splits": list(self.splits),
        }
        missing_label = self.missing_label
        if values is not None:
            raw = list(values)
            clean = _numeric_values_with_missing(raw)
            bin_counts = numeric_bin_counts(
                clean,
                edges,
                labels,
                boundary_strategy=self.boundary_strategy,
                underflow_label=self.underflow_label,
                overflow_label=self.overflow_label,
            )
            fit_stats["n_observations"] = len(clean)
            fit_stats["n_missing"] = missing_count(raw)
            fit_stats["bin_counts"] = bin_counts
            if self.missing_strategy == "most_frequent":
                missing_label = most_frequent_label(bin_counts)
        elif self.missing_strategy == "most_frequent":
            raise InvalidBucketingError(
                "ExternalSplitBucket.fit(values) requires data when "
                "missing_strategy='most_frequent'."
            )

        self.rules_ = Rule(
            kind="numeric",
            feature_name=self.feature_name,
            edges=edges,
            labels=labels,
            missing_strategy=self.missing_strategy,
            missing_label=missing_label,
            boundary_strategy=self.boundary_strategy,
            underflow_label=self.underflow_label,
            overflow_label=self.overflow_label,
            fit_stats=fit_stats,
        )
        return self


def _validated_pairs(
    values: Iterable[Any], y: Iterable[Any] | None
) -> tuple[list[Any], list[float], list[Any]]:
    if y is None:
        raise InvalidBucketingError(
            "Supervised bucketers require target values: call fit(X, y)."
        )
    _validate_target_alignment(values, y)
    raw_values = list(values)
    targets = list(y)
    if len(raw_values) != len(targets):
        raise InvalidBucketingError(
            "X and y must have the same length for supervised bucketing."
        )

    clean_values: list[float] = []
    clean_targets: list[Any] = []
    for value, target in zip(raw_values, targets):
        if is_missing(target):
            raise InvalidBucketingError(
                "Supervised targets cannot contain missing values."
            )
        if is_missing(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidBucketingError(
                f"Non-numeric value encountered: {value!r}"
            ) from exc
        if not isfinite(number):
            raise InvalidBucketingError(f"Non-finite value encountered: {value!r}")
        clean_values.append(number)
        clean_targets.append(target)

    if not clean_values:
        raise InvalidBucketingError(
            "At least one non-missing numeric value is required."
        )
    return raw_values, clean_values, clean_targets


def _numeric_values_with_missing(values: Sequence[Any]) -> list[float]:
    clean: list[float] = []
    for value in values:
        if is_missing(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidBucketingError(
                f"Non-numeric value encountered: {value!r}"
            ) from exc
        if not isfinite(number):
            raise InvalidBucketingError(f"Non-finite value encountered: {value!r}")
        clean.append(number)
    if not clean:
        raise InvalidBucketingError(
            "At least one non-missing numeric value is required."
        )
    return clean


def _binary_events(targets: Sequence[Any], event_label: Any) -> list[int]:
    classes: list[Any] = []
    for target in targets:
        if target not in classes:
            classes.append(target)
    if len(classes) != 2:
        raise InvalidBucketingError(
            "WoE and ChiMerge bucketers require exactly two target classes."
        )
    if event_label not in classes:
        raise InvalidBucketingError("event_label must be present in y.")
    return [1 if target == event_label else 0 for target in targets]


def _initial_edges(values: Sequence[float], n_bins: int, strategy: str) -> list[float]:
    sorted_values = sorted(values)
    minimum = sorted_values[0]
    maximum = sorted_values[-1]
    if minimum == maximum:
        return [minimum, maximum]
    if strategy == "width":
        width = (maximum - minimum) / n_bins
        edges = [minimum + width * index for index in range(n_bins + 1)]
        edges[-1] = maximum
        return edges
    edges = [_quantile(sorted_values, index / n_bins) for index in range(n_bins + 1)]
    return _dedupe_edges(edges)


def _supervised_summary(
    values: Sequence[float],
    events: Sequence[int],
    edges: Sequence[float],
    labels: Sequence[Any],
    smoothing: float,
) -> tuple[list[dict[str, Any]], float]:
    n_bins = len(labels)
    rows: list[dict[str, Any]] = []
    assigned_values: list[list[float]] = [[] for _ in labels]
    event_counts = [0 for _ in labels]
    total_events = sum(events)
    total_non_events = len(events) - total_events
    if total_events == 0 or total_non_events == 0:
        raise InvalidBucketingError(
            "Both target classes must be present after filtering."
        )

    for value, event in zip(values, events):
        index = _numeric_bin_index(value, edges)
        assigned_values[index].append(value)
        event_counts[index] += event

    iv = 0.0
    for index, label in enumerate(labels):
        count = len(assigned_values[index])
        event_count = event_counts[index]
        non_event_count = count - event_count
        event_dist = (event_count + smoothing) / (
            total_events + smoothing * n_bins
        )
        non_event_dist = (non_event_count + smoothing) / (
            total_non_events + smoothing * n_bins
        )
        woe = log(event_dist / non_event_dist)
        iv_contribution = (event_dist - non_event_dist) * woe
        iv += iv_contribution
        row_values = assigned_values[index]
        rows.append(
            {
                "label": label,
                "count": count,
                "event_count": event_count,
                "non_event_count": non_event_count,
                "target_rate": event_count / count if count else 0.0,
                "min": min(row_values) if row_values else None,
                "max": max(row_values) if row_values else None,
                "woe": woe,
                "iv": iv_contribution,
            }
        )
    return rows, iv


def _make_initial_bins(
    values: Sequence[float],
    events: Sequence[int],
    initial_bins: int,
    strategy: str = "quantile",
) -> list[_Bin]:
    edges = _initial_edges(values, initial_bins, strategy)
    bins = [_Bin(left, right, [], []) for left, right in zip(edges, edges[1:])]
    for value, event in zip(values, events):
        index = _numeric_bin_index(value, edges)
        bins[index].values.append(value)
        bins[index].events.append(event)
    return [item for item in bins if item.values] or [
        _Bin(min(values), max(values), list(values), list(events))
    ]


def _merge_min_size_bins(bins: list[_Bin], min_bin_count: int) -> int:
    merge_count = 0
    while len(bins) > 1 and any(len(item.values) < min_bin_count for item in bins):
        stats = [_chi_square(left, right) for left, right in zip(bins, bins[1:])]
        pair_index = _best_merge_index(bins, stats, min_bin_count)
        bins[pair_index] = _merge_two_bins(bins[pair_index], bins[pair_index + 1])
        del bins[pair_index + 1]
        merge_count += 1
    if any(len(item.values) < min_bin_count for item in bins):
        raise InvalidBucketingError(
            "min_bin_size is larger than the number of fitted observations."
        )
    return merge_count


def _resolve_monotonic_direction(
    bins: Sequence[_Bin], monotonic: str | None, smoothing: float
) -> str | None:
    if monotonic is None or len(bins) < 2:
        return None
    if monotonic in {"ascending", "descending"}:
        return monotonic

    values = _bin_woes(bins, smoothing)
    ascending_violations = _monotonic_violations(values, "ascending")
    descending_violations = _monotonic_violations(values, "descending")
    if len(ascending_violations) < len(descending_violations):
        return "ascending"
    if len(descending_violations) < len(ascending_violations):
        return "descending"
    return "ascending" if values[-1] >= values[0] else "descending"


def _merge_monotonic_bins(
    bins: list[_Bin], direction: str | None, smoothing: float
) -> int:
    if direction is None:
        return 0

    merge_count = 0
    while len(bins) > 1:
        woes = _bin_woes(bins, smoothing)
        violations = _monotonic_violations(woes, direction)
        if not violations:
            break
        pair_index = min(
            violations,
            key=lambda index: (abs(woes[index] - woes[index + 1]), index),
        )
        bins[pair_index] = _merge_two_bins(bins[pair_index], bins[pair_index + 1])
        del bins[pair_index + 1]
        merge_count += 1
    return merge_count


def _bin_woes(bins: Sequence[_Bin], smoothing: float) -> list[float]:
    total_events = sum(sum(item.events) for item in bins)
    total_non_events = sum(len(item.events) - sum(item.events) for item in bins)
    n_bins = len(bins)
    return [
        _bin_woe(item, total_events, total_non_events, n_bins, smoothing)
        for item in bins
    ]


def _bin_woe(
    item: _Bin,
    total_events: int,
    total_non_events: int,
    n_bins: int,
    smoothing: float,
) -> float:
    event_count = sum(item.events)
    non_event_count = len(item.events) - event_count
    event_dist = (event_count + smoothing) / (total_events + smoothing * n_bins)
    non_event_dist = (non_event_count + smoothing) / (
        total_non_events + smoothing * n_bins
    )
    return log(event_dist / non_event_dist)


def _monotonic_violations(values: Sequence[float], direction: str) -> list[int]:
    if direction == "ascending":
        return [
            index
            for index, (left, right) in enumerate(zip(values, values[1:]))
            if left > right
        ]
    return [
        index
        for index, (left, right) in enumerate(zip(values, values[1:]))
        if left < right
    ]


def _merge_chi_bins(
    bins: list[_Bin],
    *,
    max_bins: int,
    min_chi2: float | None,
    min_bin_size: int,
) -> int:
    merge_count = 0
    while len(bins) > 1:
        stats = [_chi_square(left, right) for left, right in zip(bins, bins[1:])]
        min_stat = min(stats)
        under_size = any(len(item.values) < min_bin_size for item in bins)
        should_merge = len(bins) > max_bins or under_size
        if min_chi2 is not None and min_stat < min_chi2:
            should_merge = True
        if not should_merge:
            break
        pair_index = _best_merge_index(bins, stats, min_bin_size)
        bins[pair_index] = _merge_two_bins(bins[pair_index], bins[pair_index + 1])
        del bins[pair_index + 1]
        merge_count += 1
    return merge_count


def _best_merge_index(
    bins: Sequence[_Bin], stats: Sequence[float], min_bin_size: int
) -> int:
    undersized = [
        index for index, item in enumerate(bins) if len(item.values) < min_bin_size
    ]
    if not undersized:
        return stats.index(min(stats))

    candidates: set[int] = set()
    for index in undersized:
        if index > 0:
            candidates.add(index - 1)
        if index < len(bins) - 1:
            candidates.add(index)
    return min(candidates, key=lambda candidate: stats[candidate])


def _merge_two_bins(left: _Bin, right: _Bin) -> _Bin:
    return _Bin(
        left=left.left,
        right=right.right,
        values=[*left.values, *right.values],
        events=[*left.events, *right.events],
    )


def _chi_square(left: _Bin, right: _Bin) -> float:
    table = [
        [sum(left.events), len(left.events) - sum(left.events)],
        [sum(right.events), len(right.events) - sum(right.events)],
    ]
    row_totals = [sum(row) for row in table]
    column_totals = [
        table[0][0] + table[1][0],
        table[0][1] + table[1][1],
    ]
    total = sum(row_totals)
    if total == 0:
        return 0.0
    statistic = 0.0
    for row_index, row in enumerate(table):
        for column_index, observed in enumerate(row):
            expected = row_totals[row_index] * column_totals[column_index] / total
            if expected:
                statistic += (observed - expected) ** 2 / expected
    return statistic


def _resolved_missing_label(
    strategy: MissingStrategy, missing_label: Any, bin_counts: Sequence[dict[str, Any]]
) -> Any:
    if strategy == "most_frequent":
        return most_frequent_label(bin_counts)
    return missing_label


def _numeric_bin_index(value: float, edges: Sequence[float]) -> int:
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        is_final = index == len(edges) - 2
        if left <= value < right or (is_final and left <= value <= right):
            return index
    return len(edges) - 2


def _quantile(values: Sequence[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _dedupe_edges(edges: Sequence[float]) -> list[float]:
    result: list[float] = []
    for edge in edges:
        if not result or edge != result[-1]:
            result.append(edge)
    if len(result) == 1:
        return [result[0], result[0]]
    return result


def _validate_min_samples_leaf(value: int | float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidBucketingError(
            "min_samples_leaf must be a positive integer or float in (0, 0.5]."
        )
    if isinstance(value, int) and value < 1:
        raise InvalidBucketingError("min_samples_leaf must be positive.")
    if isinstance(value, float) and not 0 < value <= 0.5:
        raise InvalidBucketingError("Float min_samples_leaf must be in (0, 0.5].")


def _validate_max_leaf_nodes(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise InvalidBucketingError("max_leaf_nodes must be an integer >= 2.")


def _validate_smoothing(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or value <= 0
    ):
        raise InvalidBucketingError("smoothing must be a positive finite number.")


def _validate_min_bin_size(value: int | float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or value <= 0
    ):
        raise InvalidBucketingError(
            "min_bin_size must be a positive count or ratio."
        )
    if isinstance(value, float) and value > 1:
        raise InvalidBucketingError(
            "Float min_bin_size must be in the range (0, 1]."
        )


def _min_bin_count(min_bin_size: int | float, n_observations: int) -> int:
    count = (
        int(ceil(min_bin_size * n_observations))
        if isinstance(min_bin_size, float)
        else min_bin_size
    )
    if count > n_observations:
        raise InvalidBucketingError(
            "min_bin_size is larger than the number of fitted observations."
        )
    return max(1, count)


def _validate_monotonic(value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str) or value not in {"ascending", "descending", "auto"}:
        raise InvalidBucketingError(
            "monotonic must be None, 'ascending', 'descending', or 'auto'."
        )


def _validate_min_chi2(value: float | None) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or value < 0
    ):
        raise InvalidBucketingError("min_chi2 must be a non-negative finite number.")


def _validate_target_alignment(values: Iterable[Any], y: Iterable[Any]) -> None:
    left_index = _non_callable_index(values)
    right_index = _non_callable_index(y)
    if left_index is None or right_index is None:
        return

    equals = getattr(left_index, "equals", None)
    if callable(equals):
        try:
            aligned = bool(equals(right_index))
        except (TypeError, ValueError):
            aligned = False
    else:
        try:
            aligned = list(left_index) == list(right_index)
        except TypeError:
            aligned = left_index == right_index
    if not aligned:
        raise InvalidBucketingError(
            "X and y indexes must be aligned for supervised bucketing."
        )


def _non_callable_index(obj: Any) -> Any | None:
    index = getattr(obj, "index", None)
    if index is None or callable(index):
        return None
    return index


def _restore_supervised_attributes(obj: SupervisedBucket) -> None:
    stats = obj.rules_.fit_stats
    if "iv" in stats:
        obj.iv_ = stats["iv"]
    summary = stats.get("bin_summary")
    if isinstance(summary, list) and "event_label" in stats:
        event_count = sum(row.get("event_count", 0) for row in summary)
        non_event_count = sum(row.get("non_event_count", 0) for row in summary)
        obj.target_summary_ = {
            "event_label": stats["event_label"],
            "event_count": event_count,
            "non_event_count": non_event_count,
        }


def _validate_splits(splits: Sequence[float]) -> list[float]:
    if isinstance(splits, (str, bytes)):
        raise InvalidBucketingError("External splits must be a sequence of numbers.")
    converted: list[float] = []
    for split in splits:
        try:
            number = float(split)
        except (TypeError, ValueError) as exc:
            raise InvalidBucketingError(f"Invalid split value: {split!r}") from exc
        if not isfinite(number):
            raise InvalidBucketingError("External splits must be finite numbers.")
        converted.append(number)
    if any(left >= right for left, right in zip(converted, converted[1:])):
        raise InvalidBucketingError("External splits must be sorted and unique.")
    return converted
