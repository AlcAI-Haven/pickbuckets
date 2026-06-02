import pytest

from pickbuckets import (
    BoundaryError,
    CustomBoundaryBucket,
    EqualFrequencyBucket,
    EqualWidthBucket,
    InvalidBucketingError,
    NotFittedError,
)


@pytest.mark.parametrize("bucket_cls", [EqualWidthBucket, EqualFrequencyBucket])
@pytest.mark.parametrize("n_bins", [0, -1, True, 1.5, "3"])
def test_numeric_bucketers_reject_invalid_n_bins(bucket_cls, n_bins):
    with pytest.raises(InvalidBucketingError):
        bucket_cls(n_bins=n_bins)


def test_equal_width_uses_left_closed_right_open_and_final_closed():
    bucket = EqualWidthBucket(n_bins=2, labels="ordinal").fit([0, 10])

    assert bucket.transform([0, 4.99, 5, 10]) == [0, 0, 1, 1]


def test_equal_width_handles_constant_input_as_single_stable_bin():
    bucket = EqualWidthBucket(n_bins=3).fit([7, 7, 7])

    assert bucket.rules_.edges == [7.0, 7.0]
    assert bucket.transform([6, 7, 8]) == [0, 0, 0]


def test_equal_width_interval_labels_and_round_trip():
    bucket = EqualWidthBucket(n_bins=2, labels="interval").fit([0, 10])
    restored = EqualWidthBucket.from_json(bucket.to_json())

    assert restored.transform([0, 5, 10]) == bucket.transform([0, 5, 10])
    assert bucket.to_dict()["schema_version"].startswith("1.")


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


def test_equal_frequency_round_trip():
    bucket = EqualFrequencyBucket(n_bins=4).fit([1, 2, 3, 4, 5, 6, 7, 8])
    restored = EqualFrequencyBucket.from_dict(bucket.to_dict())

    assert restored.transform([1, 3, 5, 8]) == bucket.transform([1, 3, 5, 8])


def test_equal_frequency_duplicate_edges_raise_by_default():
    with pytest.raises(InvalidBucketingError):
        EqualFrequencyBucket(n_bins=4).fit([1, 1, 1, 1])


def test_equal_frequency_rejects_invalid_duplicates_policy():
    with pytest.raises(InvalidBucketingError):
        EqualFrequencyBucket(n_bins=4, duplicates=["drop"])


def test_equal_frequency_duplicate_edges_can_drop():
    bucket = EqualFrequencyBucket(n_bins=4, duplicates="drop").fit([1, 1, 1, 1, 2, 3])

    assert bucket.rules_.fit_stats["actual_bins"] < 4


def test_equal_frequency_duplicate_edges_drop_to_single_constant_bin():
    bucket = EqualFrequencyBucket(n_bins=4, duplicates="drop").fit([1, 1, 1, 1])

    assert bucket.rules_.edges == [1.0, 1.0]
    assert bucket.transform([0, 1, 2]) == [0, 0, 0]


def test_custom_boundaries_validate_labels():
    with pytest.raises(InvalidBucketingError):
        CustomBoundaryBucket(edges=[0, 10, 20], labels=["too_few"])


def test_custom_boundaries_reject_nan_edges():
    with pytest.raises(InvalidBucketingError):
        CustomBoundaryBucket(edges=[0, float("nan"), 20])


def test_custom_boundaries_support_open_ended_edges():
    bucket = CustomBoundaryBucket(
        edges=[float("-inf"), 0, float("inf")],
        labels=["negative", "non_negative"],
    ).fit()

    assert bucket.transform([-100, -0.1, 0, 100]) == [
        "negative",
        "negative",
        "non_negative",
        "non_negative",
    ]


def test_custom_boundaries_can_error_on_out_of_range():
    bucket = CustomBoundaryBucket(
        edges=[0, 10, 20],
        labels=["low", "high"],
        boundary_strategy="error",
    ).fit()

    with pytest.raises(BoundaryError):
        bucket.transform([-1])


def test_custom_boundaries_validate_fit_values_when_erroring_on_boundaries():
    with pytest.raises(InvalidBucketingError, match="outside custom edges"):
        CustomBoundaryBucket(
            edges=[0, 10],
            boundary_strategy="error",
        ).fit([-1, 5])


def test_custom_boundary_round_trip():
    bucket = CustomBoundaryBucket(edges=[0, 10, 20], labels="interval").fit([1, 19])
    restored = CustomBoundaryBucket.from_dict(bucket.to_dict())

    assert restored.transform([0, 10, 20]) == bucket.transform([0, 10, 20])


def test_generated_numeric_edges_are_monotonic():
    buckets = [
        EqualWidthBucket(n_bins=4).fit([-10, 0, 10, 20]),
        EqualWidthBucket(n_bins=4).fit([5, 5, 5]),
        EqualFrequencyBucket(n_bins=4).fit([1, 2, 3, 4, 5, 6, 7, 8]),
        CustomBoundaryBucket(edges=[float("-inf"), 0, 10, float("inf")]).fit(),
    ]

    for bucket in buckets:
        edges = bucket.rules_.edges
        assert edges is not None
        assert all(left <= right for left, right in zip(edges, edges[1:]))


def test_numeric_bucketers_accept_numpy_arrays_when_available():
    np = pytest.importorskip("numpy")

    bucket = EqualWidthBucket(n_bins=2).fit(np.array([0, 5, 10]))

    assert bucket.transform(np.array([0, 5, 10])) == [0, 1, 1]
