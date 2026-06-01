import pytest

from pickbuckets import (
    BoundaryError,
    CustomBoundaryBucket,
    EqualFrequencyBucket,
    EqualWidthBucket,
    InvalidBucketingError,
    NotFittedError,
)


def test_equal_width_uses_left_closed_right_open_and_final_closed():
    bucket = EqualWidthBucket(n_bins=2, labels="ordinal").fit([0, 10])

    assert bucket.transform([0, 4.99, 5, 10]) == [0, 0, 1, 1]


def test_equal_width_interval_labels_and_round_trip():
    bucket = EqualWidthBucket(n_bins=2, labels="interval").fit([0, 10])
    restored = EqualWidthBucket.from_json(bucket.to_json())

    assert restored.transform([0, 5, 10]) == bucket.transform([0, 5, 10])
    assert bucket.to_dict()["schema_version"] == "1.0"


def test_equal_width_rejects_empty_input():
    with pytest.raises(InvalidBucketingError):
        EqualWidthBucket(n_bins=3).fit([])


def test_transform_before_fit_raises_clear_error():
    with pytest.raises(NotFittedError):
        EqualWidthBucket(n_bins=3).transform([1, 2, 3])


def test_equal_frequency_balances_simple_values():
    bucket = EqualFrequencyBucket(n_bins=4).fit([1, 2, 3, 4, 5, 6, 7, 8])

    assert len(bucket.rules_.edges) == 5
    assert bucket.transform([1, 3, 5, 8]) == [0, 1, 2, 3]


def test_equal_frequency_duplicate_edges_raise_by_default():
    with pytest.raises(InvalidBucketingError):
        EqualFrequencyBucket(n_bins=4).fit([1, 1, 1, 1])


def test_equal_frequency_duplicate_edges_can_drop():
    bucket = EqualFrequencyBucket(n_bins=4, duplicates="drop").fit([1, 1, 1, 1, 2, 3])

    assert bucket.rules_.fit_stats["actual_bins"] < 4


def test_custom_boundaries_validate_labels():
    with pytest.raises(InvalidBucketingError):
        CustomBoundaryBucket(edges=[0, 10, 20], labels=["too_few"])


def test_custom_boundaries_can_error_on_out_of_range():
    bucket = CustomBoundaryBucket(
        edges=[0, 10, 20],
        labels=["low", "high"],
        boundary_strategy="error",
    ).fit()

    with pytest.raises(BoundaryError):
        bucket.transform([-1])


def test_custom_boundary_round_trip():
    bucket = CustomBoundaryBucket(edges=[0, 10, 20], labels="interval").fit([1, 19])
    restored = CustomBoundaryBucket.from_dict(bucket.to_dict())

    assert restored.transform([0, 10, 20]) == bucket.transform([0, 10, 20])
