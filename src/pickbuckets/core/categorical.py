from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from pickbuckets.core._utils import (
    label_counts,
    missing_count,
    most_frequent_label,
    validate_min_frequency,
    validate_missing_strategy,
    validate_unknown_category_strategy,
)
from pickbuckets.core.base import BaseBucket
from pickbuckets.exceptions import InvalidBucketingError
from pickbuckets.rules import Rule
from pickbuckets.rules.schema import MissingStrategy, UnknownCategoryStrategy
from pickbuckets.runtime.apply import is_missing


class RareCategoryBucket(BaseBucket):
    def __init__(
        self,
        min_frequency: int | float = 0.01,
        *,
        other_label: Any = "Other",
        feature_name: str | None = None,
        missing_strategy: MissingStrategy = "separate",
        missing_label: Any = "Missing",
        unknown_category_strategy: UnknownCategoryStrategy = "other",
    ) -> None:
        validate_min_frequency(min_frequency)
        validate_missing_strategy(missing_strategy)
        validate_unknown_category_strategy(unknown_category_strategy)
        self.min_frequency = min_frequency
        self.other_label = other_label
        self.feature_name = feature_name
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.unknown_category_strategy = unknown_category_strategy

    def fit(self, values: Iterable[Any]) -> RareCategoryBucket:
        raw = list(values)
        seen = [value for value in raw if not is_missing(value)]
        if not seen:
            raise InvalidBucketingError(
                "At least one non-missing category is required."
            )

        counts = Counter(str(value) for value in seen)
        threshold = (
            self.min_frequency * len(seen)
            if isinstance(self.min_frequency, float)
            else self.min_frequency
        )
        mapping = {
            category: category if count >= threshold else self.other_label
            for category, count in sorted(counts.items())
        }
        frequent = [
            category for category, count in sorted(counts.items()) if count >= threshold
        ]
        labels = list(frequent)
        if self.other_label not in labels:
            labels.append(self.other_label)
        mapped_seen = [mapping[str(value)] for value in seen]
        output_counts = label_counts(mapped_seen, labels)
        missing_label = self.missing_label
        if self.missing_strategy == "most_frequent":
            missing_label = most_frequent_label(output_counts)

        self.rules_ = Rule(
            kind="categorical",
            feature_name=self.feature_name,
            category_mapping=mapping,
            labels=labels,
            missing_strategy=self.missing_strategy,
            missing_label=missing_label,
            unknown_category_strategy=self.unknown_category_strategy,
            unknown_label=self.other_label,
            fit_stats={
                "algorithm": "rare_category",
                "n_observations": len(seen),
                "n_missing": missing_count(raw),
                "n_categories": len(counts),
                "n_frequent": len(frequent),
                "min_frequency": self.min_frequency,
                "category_counts": dict(sorted(counts.items())),
                "output_counts": output_counts,
            },
        )
        return self
