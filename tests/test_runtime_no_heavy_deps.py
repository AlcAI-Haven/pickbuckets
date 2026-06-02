import sys

import pytest

from pickbuckets import EqualWidthBucket, InvalidBucketingError, MissingValueError, Rule
from pickbuckets.runtime import apply_rule


def test_base_import_does_not_import_heavy_dependencies():
    assert "pandas" not in sys.modules
    assert "polars" not in sys.modules
    assert "sklearn" not in sys.modules


def test_saved_rule_applies_with_plain_python_runtime():
    rule = Rule(kind="numeric", edges=[0, 10, 20], labels=["low", "high"])

    assert apply_rule(rule, [0, 10, 20, None]) == ["low", "high", "high", "Missing"]


def test_numeric_runtime_handles_string_nan_as_missing():
    rule = Rule(kind="numeric", edges=[0, 10, 20], labels=["low", "high"])

    assert apply_rule(rule, ["nan"]) == ["Missing"]


def test_numeric_runtime_rejects_malformed_values():
    rule = Rule(kind="numeric", edges=[0, 10, 20], labels=["low", "high"])

    with pytest.raises(InvalidBucketingError):
        apply_rule(rule, ["not-a-number"])


def test_missing_strategy_error_uses_typed_exception():
    rule = Rule(
        kind="numeric",
        edges=[0, 10, 20],
        labels=["low", "high"],
        missing_strategy="error",
    )

    with pytest.raises(MissingValueError):
        apply_rule(rule, [None])


def test_bucket_json_restores_without_training_data():
    bucket = EqualWidthBucket(n_bins=2).fit([0, 10])
    restored = EqualWidthBucket.from_json(bucket.to_json())

    assert restored.transform([1, 9]) == [0, 1]
