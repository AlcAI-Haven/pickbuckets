from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

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
        if not isinstance(min_frequency, (int, float)) or min_frequency <= 0:
            raise InvalidBucketingError(
                "min_frequency must be a positive count or ratio."
            )
        if isinstance(min_frequency, float) and min_frequency > 1:
            raise InvalidBucketingError(
                "Float min_frequency must be in the range (0, 1]."
            )
        self.min_frequency = min_frequency
        self.other_label = other_label
        self.feature_name = feature_name
        self.missing_strategy = missing_strategy
        self.missing_label = missing_label
        self.unknown_category_strategy = unknown_category_strategy

    def fit(self, values: Iterable[Any]) -> RareCategoryBucket:
        seen = [value for value in values if not is_missing(value)]
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
            category for category, label in mapping.items() if label != self.other_label
        ]
        labels = frequent + [self.other_label]

        self.rules_ = Rule(
            kind="categorical",
            feature_name=self.feature_name,
            category_mapping=mapping,
            labels=labels,
            missing_strategy=self.missing_strategy,
            missing_label=self.missing_label,
            unknown_category_strategy=self.unknown_category_strategy,
            unknown_label=self.other_label,
            fit_stats={
                "algorithm": "rare_category",
                "n_observations": len(seen),
                "n_categories": len(counts),
                "n_frequent": len(frequent),
                "min_frequency": self.min_frequency,
            },
        )
        return self
