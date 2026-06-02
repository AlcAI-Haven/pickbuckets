import pytest

from pickbuckets import (
    AutoBucket,
    BoundaryError,
    CustomBoundaryBucket,
    EqualWidthBucket,
    InvalidBucketingError,
    PickBucketsError,
    RareCategoryBucket,
    Rule,
    RuleSchemaError,
    UnknownCategoryError,
)
from pickbuckets.runtime import apply_rule


def test_missing_strategy_most_frequent_numeric_round_trips():
    bucket = EqualWidthBucket(n_bins=2, missing_strategy="most_frequent").fit(
        [0, 1, 2, 9, 10, None]
    )

    assert bucket.rules_.missing_label == 0
    assert bucket.transform([None, 10]) == [0, 1]
    assert bucket.summary()["bin_counts"] == [
        {"label": 0, "count": 3},
        {"label": 1, "count": 2},
    ]

    restored = EqualWidthBucket.from_json(bucket.to_json())
    assert restored.transform([None, 10]) == [0, 1]


def test_missing_strategy_most_frequent_categorical_is_not_rare_bucket():
    bucket = RareCategoryBucket(
        min_frequency=2,
        missing_strategy="most_frequent",
    ).fit(["a", "a", "a", "b", "c", None])

    assert bucket.rules_.missing_label == "a"
    assert bucket.transform([None, "b"]) == ["a", "Other"]


def test_most_frequent_requires_fit_data_for_custom_boundaries():
    bucket = CustomBoundaryBucket(
        edges=[0, 10],
        missing_strategy="most_frequent",
    )

    with pytest.raises(InvalidBucketingError, match="requires data"):
        bucket.fit()


def test_boundary_strategy_underflow_overflow_round_trips():
    bucket = EqualWidthBucket(
        n_bins=2,
        boundary_strategy="underflow_overflow",
        underflow_label="Below",
        overflow_label="Above",
    ).fit([0, 10])

    assert bucket.transform([-1, 0, 10, 11]) == ["Below", 0, 1, "Above"]
    assert bucket.summary()["underflow_label"] == "Below"

    restored = Rule.from_dict(bucket.to_dict())
    assert apply_rule(restored, [-1, 11]) == ["Below", "Above"]


def test_open_ended_edges_do_not_emit_underflow_or_overflow():
    bucket = CustomBoundaryBucket(
        edges=[float("-inf"), 0, float("inf")],
        labels=["negative", "non_negative"],
        boundary_strategy="underflow_overflow",
    ).fit()

    assert bucket.transform([-100, 100]) == ["negative", "non_negative"]


def test_custom_boundary_most_frequent_can_learn_underflow_label():
    bucket = CustomBoundaryBucket(
        edges=[0, 10],
        missing_strategy="most_frequent",
        boundary_strategy="underflow_overflow",
        underflow_label="Below",
        overflow_label="Above",
    ).fit([-3, -2, 5])

    assert bucket.rules_.missing_label == "Below"
    assert bucket.transform([None, -1, 5]) == ["Below", "Below", 0]


def test_unknown_category_can_map_to_missing_bucket():
    bucket = RareCategoryBucket(
        min_frequency=2,
        unknown_category_strategy="missing",
    ).fit(["a", "a", "b"])

    assert bucket.transform(["z", None]) == ["Missing", "Missing"]


def test_unknown_category_can_map_to_most_frequent_missing_label():
    bucket = RareCategoryBucket(
        min_frequency=2,
        missing_strategy="most_frequent",
        unknown_category_strategy="missing",
    ).fit(["a", "a", "b"])

    assert bucket.transform(["z", None]) == ["a", "a"]


def test_unknown_category_can_be_kept_as_string():
    bucket = RareCategoryBucket(
        min_frequency=2,
        unknown_category_strategy="keep",
    ).fit(["a", "a", "b"])

    assert bucket.transform(["z", 10]) == ["z", "10"]


def test_autobucket_errors_include_failing_column_name():
    frame = {"score": [0, 10]}
    auto = AutoBucket(
        overrides={
            "score": EqualWidthBucket(n_bins=2, boundary_strategy="error"),
        }
    ).fit(frame)

    with pytest.raises(BoundaryError, match="score"):
        auto.transform({"score": [-1]})


def test_invalid_policy_values_fail_early():
    with pytest.raises(InvalidBucketingError):
        EqualWidthBucket(boundary_strategy="bad")
    with pytest.raises(InvalidBucketingError):
        EqualWidthBucket(missing_strategy=["separate"])
    with pytest.raises(InvalidBucketingError):
        RareCategoryBucket(unknown_category_strategy="bad")
    with pytest.raises(InvalidBucketingError):
        AutoBucket(missing_strategy="bad")


def test_rule_validates_most_frequent_missing_replacement():
    with pytest.raises(RuleSchemaError, match="most_frequent"):
        Rule(
            kind="numeric",
            edges=[0, 10],
            labels=["low"],
            missing_strategy="most_frequent",
        )


def test_pandas_series_uses_most_frequent_policy():
    pd = pytest.importorskip("pandas")
    series = pd.Series([None, 10], index=["a", "b"], name="score")
    bucket = EqualWidthBucket(n_bins=2, missing_strategy="most_frequent").fit(
        [0, 1, 2, 9, 10]
    )

    out = bucket.transform(series)

    assert out.tolist() == [0, 1]
    assert list(out.index) == ["a", "b"]
    assert out.name == "score"


def test_numpy_array_uses_most_frequent_policy():
    np = pytest.importorskip("numpy")
    bucket = EqualWidthBucket(n_bins=2, missing_strategy="most_frequent").fit(
        [0, 1, 2, 9, 10]
    )

    assert bucket.transform(np.array([None, 10], dtype=object)) == [0, 1]


def test_polars_series_matches_runtime_for_phase3_policies():
    pl = pytest.importorskip("polars")
    bucket = EqualWidthBucket(
        n_bins=2,
        missing_strategy="most_frequent",
        boundary_strategy="underflow_overflow",
        underflow_label="Below",
        overflow_label="Above",
    ).fit([0, 1, 2, 9, 10])
    values = [-1, 0, None, 10, 11]

    out = bucket.transform(pl.Series("score", values))

    assert out.to_list() == bucket.transform(values)


def test_polars_numeric_missing_propagate_matches_runtime():
    pl = pytest.importorskip("polars")
    bucket = EqualWidthBucket(n_bins=2, missing_strategy="propagate").fit([0, 10])

    out = bucket.transform(
        pl.Series("score", [0, None, float("nan"), 10], strict=False)
    )

    assert out[0] == 0
    assert out[1] is None
    assert out[2] != out[2]  # NaN propagates.
    assert out[3] == 1


def test_polars_categorical_unknown_keep_matches_runtime():
    pl = pytest.importorskip("polars")
    bucket = RareCategoryBucket(
        min_frequency=2,
        unknown_category_strategy="keep",
    ).fit(["a", "a", "b"])
    values = ["a", "z", None]

    out = bucket.transform(pl.Series("category", values))

    assert out.to_list() == bucket.transform(values)


def test_polars_categorical_missing_propagate_matches_runtime():
    pl = pytest.importorskip("polars")
    bucket = RareCategoryBucket(
        min_frequency=2,
        missing_strategy="propagate",
    ).fit(["a", "a", "b"])
    values = ["a", None, "z"]

    out = bucket.transform(pl.Series("category", values))

    assert out.to_list() == bucket.transform(values)


@pytest.mark.parametrize(
    "bucket",
    [
        RareCategoryBucket(
            min_frequency=2,
            other_label=999,
            unknown_category_strategy="other",
        ).fit(["a", "b"]),
        RareCategoryBucket(
            min_frequency=2,
            missing_label=0,
            unknown_category_strategy="missing",
        ).fit(["a", "b"]),
    ],
)
def test_polars_categorical_mixed_type_outputs_match_runtime(bucket):
    pl = pytest.importorskip("polars")
    values = ["a", "z", None]

    out = bucket.transform(pl.Series("category", values))

    assert out.to_list() == bucket.transform(values)


def test_polars_rejects_mixed_type_unknown_keep_outputs():
    pl = pytest.importorskip("polars")
    bucket = RareCategoryBucket(
        min_frequency=2,
        other_label=999,
        unknown_category_strategy="keep",
    ).fit(["a", "b"])

    with pytest.raises(PickBucketsError, match="mixed-type"):
        bucket.transform(pl.Series("category", ["a", "z", None]))


def test_polars_eager_errors_include_column_name():
    pl = pytest.importorskip("polars")
    auto = AutoBucket(
        overrides={
            "city": RareCategoryBucket(
                min_frequency=1,
                unknown_category_strategy="error",
            ),
        }
    ).fit(pl.DataFrame({"city": ["A"]}))

    with pytest.raises(UnknownCategoryError, match="city"):
        auto.transform(pl.DataFrame({"city": ["B"]}))

def test_polars_numeric_malformed_value_raises_typed_error():
    pl = pytest.importorskip("polars")
    bucket = EqualWidthBucket(n_bins=3, labels="ordinal").fit([0, 1, 2, 3, 4, 5])

    # A numeric rule applied to a Utf8 Series containing a non-numeric value
    # must raise the typed, pure-Python-equivalent error rather than leaking a
    # Polars ComputeError.
    with pytest.raises(InvalidBucketingError, match="oops"):
        bucket.transform(pl.Series("age", ["1.0", "oops", "3.0"]))


def test_polars_numeric_valid_string_column_matches_runtime():
    pl = pytest.importorskip("polars")
    bucket = EqualWidthBucket(n_bins=3, labels="ordinal").fit([0, 1, 2, 3, 4, 5])
    values = ["0", "5", "10"]

    out = bucket.transform(pl.Series("age", values))

    assert out.to_list() == bucket.transform(values)


def test_autobucket_polars_malformed_numeric_names_column():
    pl = pytest.importorskip("polars")
    auto = AutoBucket().fit(pl.DataFrame({"age": [1.0, 2.0, 3.0, 4.0, 5.0]}))

    with pytest.raises(InvalidBucketingError, match="'age'"):
        auto.transform(pl.DataFrame({"age": ["1.0", "bad", "3.0"]}))
