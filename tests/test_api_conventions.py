import pytest

from pickbuckets import (
    AutoBucket,
    EqualWidthBucket,
    InvalidBucketingError,
    NotFittedError,
    RareCategoryBucket,
)


def test_learned_attributes_appear_only_after_fit():
    bucket = EqualWidthBucket(n_bins=3)
    with pytest.raises(NotFittedError):
        _ = bucket.n_features_in_
    bucket.fit([0, 5, 10])
    assert bucket.n_features_in_ == 1


def test_feature_names_in_reflects_feature_name():
    bucket = EqualWidthBucket(n_bins=2, feature_name="age").fit([0, 10])
    assert bucket.feature_names_in_ == ["age"]
    assert EqualWidthBucket(n_bins=2).fit([0, 10]).feature_names_in_ is None


def test_summary_reports_numeric_rule():
    summary = EqualWidthBucket(n_bins=2, labels="interval").fit([0, 10]).summary()
    assert summary["kind"] == "numeric"
    assert summary["n_bins"] == 2
    assert summary["boundary_strategy"] == "clip"
    assert summary["edges"][0] == 0


def test_summary_reports_categorical_rule():
    summary = RareCategoryBucket(min_frequency=2).fit(["a", "a", "b"]).summary()
    assert summary["kind"] == "categorical"
    assert summary["n_categories"] == 2


def test_autobucket_dispatches_on_plain_dict_frame():
    frame = {"n": [1, 2, 3, 4, 5, 6], "c": ["x", "x", "x", "y", "z", "z"]}
    auto = AutoBucket(n_bins=3, min_frequency=2).fit(frame)
    assert auto.summary()["n"]["kind"] == "numeric"
    assert auto.summary()["c"]["kind"] == "categorical"
    assert auto.n_features_in_ == 2
    out = auto.transform(frame)
    assert set(out) == {"n", "c"}
    assert len(out["n"]) == 6


def test_autobucket_round_trips_through_dict():
    frame = {"n": [1, 2, 3, 4, 5, 6], "c": ["x", "x", "x", "y", "z", "z"]}
    auto = AutoBucket(n_bins=3, min_frequency=2).fit(frame)
    restored = AutoBucket.from_dict(auto.to_dict())
    assert restored.transform(frame) == auto.transform(frame)
    assert restored.feature_names_in_ == auto.feature_names_in_


def test_autobucket_override_takes_precedence():
    frame = {"score": [0, 10, 20, 30, 40, 50]}
    override = EqualWidthBucket(n_bins=2, labels="interval")
    auto = AutoBucket(overrides={"score": override}).fit(frame)
    assert auto.rules_["score"].fit_stats["algorithm"] == "equal_width"
    assert auto.rules_["score"].feature_name == "score"


def test_autobucket_rejects_unknown_override_column():
    with pytest.raises(InvalidBucketingError, match="unknown columns"):
        AutoBucket(overrides={"typo": EqualWidthBucket(n_bins=2)}).fit(
            {"score": [0, 1]}
        )


def test_autobucket_rejects_non_bucketer_override():
    with pytest.raises(InvalidBucketingError, match="must be a pickbuckets bucketer"):
        AutoBucket(overrides={"score": object()}).fit({"score": [0, 1]})


def test_autobucket_rejects_empty_and_ragged_mapping_frames():
    with pytest.raises(InvalidBucketingError, match="at least one column"):
        AutoBucket().fit({})
    with pytest.raises(InvalidBucketingError, match="same length"):
        AutoBucket().fit({"a": [1, 2], "b": [1]})


def test_autobucket_rejects_invalid_min_frequency_early():
    with pytest.raises(InvalidBucketingError):
        AutoBucket(min_frequency=True)


def test_autobucket_transform_requires_fit_columns():
    auto = AutoBucket(numeric_strategy="width").fit({"a": [1, 2], "b": [3, 4]})

    with pytest.raises(InvalidBucketingError, match="missing columns"):
        auto.transform({"a": [1, 2]})
    with pytest.raises(InvalidBucketingError, match="unexpected columns"):
        auto.transform({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
