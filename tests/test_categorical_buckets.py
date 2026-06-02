import pytest

from pickbuckets import InvalidBucketingError, RareCategoryBucket, UnknownCategoryError


def test_rare_category_groups_rare_values_and_missing_separately():
    bucket = RareCategoryBucket(min_frequency=2).fit(["a", "a", "b", None])

    assert bucket.transform(["a", "b", "new", None]) == [
        "a",
        "Other",
        "Other",
        "Missing",
    ]


def test_rare_category_supports_ratio_threshold():
    bucket = RareCategoryBucket(min_frequency=0.5).fit(["a", "a", "b", "c"])

    assert bucket.transform(["a", "b", "c"]) == ["a", "Other", "Other"]


def test_rare_category_rejects_bool_min_frequency():
    with pytest.raises(InvalidBucketingError):
        RareCategoryBucket(min_frequency=True)


def test_rare_category_supports_integer_and_mixed_object_values():
    bucket = RareCategoryBucket(min_frequency=2).fit([1, 1, "x", "y", "y"])

    assert bucket.transform([1, "x", "y", 2]) == ["1", "Other", "y", "Other"]


def test_rare_category_counts_frequent_category_that_matches_other_label():
    bucket = RareCategoryBucket(
        min_frequency=2,
        other_label="Other",
    ).fit(["Other", "Other", "A"])

    assert bucket.rules_.labels == ["Other"]
    assert bucket.summary()["fit_stats"]["n_frequent"] == 1
    assert bucket.summary()["output_counts"] == [{"label": "Other", "count": 3}]


def test_rare_category_can_error_for_unknown_categories():
    bucket = RareCategoryBucket(
        min_frequency=1,
        unknown_category_strategy="error",
    ).fit(["a"])

    with pytest.raises(UnknownCategoryError):
        bucket.transform(["z"])


def test_rare_category_round_trip():
    bucket = RareCategoryBucket(min_frequency=2).fit(["a", "a", "b"])
    restored = RareCategoryBucket.from_json(bucket.to_json())

    assert restored.transform(["a", "b", "z"]) == bucket.transform(["a", "b", "z"])


def test_rare_category_rejects_all_missing():
    with pytest.raises(InvalidBucketingError):
        RareCategoryBucket().fit([None])
