import pytest

pd = pytest.importorskip("pandas")

from pickbuckets import (  # noqa: E402
    AutoBucket,
    EqualWidthBucket,
    InvalidBucketingError,
)


def test_transforming_series_returns_series_with_index_and_name():
    series = pd.Series([0, 5, 10], index=[100, 200, 300], name="age")
    bucket = EqualWidthBucket(n_bins=2).fit([0, 10])
    out = bucket.transform(series)
    assert isinstance(out, pd.Series)
    assert list(out.index) == [100, 200, 300]
    assert out.name == "age"
    assert out.tolist() == [0, 1, 1]


def test_dataframe_in_dataframe_out_preserves_index_and_columns():
    df = pd.DataFrame(
        {"age": [10, 20, 30, 40, 50, 60], "city": ["A", "A", "A", "B", "C", "C"]},
        index=["r0", "r1", "r2", "r3", "r4", "r5"],
    )
    auto = AutoBucket(n_bins=3, min_frequency=2).fit(df)
    out = auto.transform(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["age", "city"]
    assert list(out.index) == list(df.index)


def test_dataframe_with_non_string_columns_is_preserved():
    df = pd.DataFrame({0: [10, 20, 30, 40], 1: ["A", "A", "B", "A"]})
    out = AutoBucket(n_bins=2, min_frequency=2).fit_transform(df)
    assert list(out.columns) == [0, 1]


def test_nullable_integer_dtype_treated_as_numeric_with_missing():
    series = pd.Series([1, 2, None, 8], dtype="Int64", name="x")
    bucket = EqualWidthBucket(n_bins=2).fit([1, 8])
    out = bucket.transform(series)
    assert out.iloc[2] == "Missing"
    assert out.iloc[0] == 0


def test_pandas_categorical_dtype_is_grouped():
    series = pd.Series(
        pd.Categorical(["a", "a", "a", "b", "c", "c"]), name="cat"
    )
    df = series.to_frame()
    auto = AutoBucket(min_frequency=2).fit(df)
    assert auto.summary()["cat"]["kind"] == "categorical"


def test_unsupported_datetime_column_requires_ignore_flag():
    df = pd.DataFrame({"when": pd.to_datetime(["2020-01-01", "2021-01-01"])})
    with pytest.raises(InvalidBucketingError):
        AutoBucket().fit(df)
    auto = AutoBucket(ignore_unsupported=True).fit(df)
    assert auto.skipped_columns_ == ["when"]
    out = auto.transform(df)
    assert list(out["when"]) == list(df["when"])


def test_duplicate_pandas_columns_are_rejected():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"])

    with pytest.raises(InvalidBucketingError, match="unique column names"):
        AutoBucket().fit(df)


def test_duplicate_pandas_columns_are_rejected_on_transform():
    fit_df = pd.DataFrame({"x": [1, 2]})
    transform_df = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"])
    auto = AutoBucket(numeric_strategy="width").fit(fit_df)

    with pytest.raises(InvalidBucketingError, match="unique column names"):
        auto.transform(transform_df)
