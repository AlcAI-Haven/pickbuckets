import pytest

pl = pytest.importorskip("polars")

from pickbuckets import (  # noqa: E402
    AutoBucket,
    BoundaryError,
    EqualWidthBucket,
    RareCategoryBucket,
)


def test_transforming_series_returns_series_with_name():
    series = pl.Series("age", [0, 5, 10])
    out = EqualWidthBucket(n_bins=2, labels="interval").fit([0, 10]).transform(series)
    assert isinstance(out, pl.Series)
    assert out.name == "age"
    assert out.to_list() == ["[0, 5)", "[5, 10]", "[5, 10]"]


def test_polars_and_pandas_paths_match_for_homogeneous_labels():
    pd = pytest.importorskip("pandas")
    bucket = EqualWidthBucket(n_bins=3, labels="interval").fit([0, 30])
    values = [0, 5, 10, 20, 30]
    pl_out = bucket.transform(pl.Series("x", values)).to_list()
    pd_out = bucket.transform(pd.Series(values)).tolist()
    assert pl_out == pd_out


def test_polars_numeric_missing_handles_null_and_nan():
    bucket = EqualWidthBucket(n_bins=2, labels="interval").fit([0, 10])
    out = bucket.transform(pl.Series("x", [0.0, 5.0, None, float("nan")]))
    assert out.to_list() == ["[0, 5)", "[5, 10]", "Missing", "Missing"]


def test_polars_numeric_ordinal_labels_match_python_runtime():
    bucket = EqualWidthBucket(n_bins=2).fit([0, 10])
    values = [0, 5, None]
    assert bucket.transform(pl.Series("x", values)).to_list() == bucket.transform(
        values
    )


def test_polars_categorical_uses_vectorized_replace():
    bucket = RareCategoryBucket(min_frequency=2).fit(
        ["FR", "FR", "FR", "US", "DE", "DE"]
    )
    out = bucket.transform(pl.Series("c", ["FR", "US", "CA", None]))
    assert out.to_list() == ["FR", "Other", "Other", "Missing"]


def test_polars_dataframe_preserves_schema_order():
    df = pl.DataFrame(
        {"age": [10, 20, 30, 40, 50, 60], "city": ["A", "A", "A", "B", "C", "C"]}
    )
    auto = AutoBucket(n_bins=3, min_frequency=2, labels="interval").fit(df)
    out = auto.transform(df)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["age", "city"]
    assert out.height == df.height


def test_polars_dataframe_eager_path_preserves_error_strategies():
    df = pl.DataFrame({"score": [0, 10]})
    auto = AutoBucket(
        overrides={
            "score": EqualWidthBucket(n_bins=2, boundary_strategy="error"),
        }
    ).fit(df)
    with pytest.raises(BoundaryError):
        auto.transform(pl.DataFrame({"score": [-1]}))


def test_polars_lazyframe_composes_and_collects():
    lf = pl.DataFrame(
        {"age": [10, 20, 30, 40, 50, 60], "city": ["A", "A", "A", "B", "C", "C"]}
    ).lazy()
    auto = AutoBucket(n_bins=3, min_frequency=2, labels="interval").fit(lf)
    out = auto.transform(lf)
    assert isinstance(out, pl.LazyFrame)
    collected = out.collect()
    assert collected.columns == ["age", "city"]


def test_eager_and_lazy_polars_match():
    df = pl.DataFrame({"age": [10, 20, 30, 40, 50, 60]})
    auto = AutoBucket(n_bins=3, labels="interval").fit(df)
    eager = auto.transform(df)
    lazy = auto.transform(df.lazy()).collect()
    assert eager.to_dict(as_series=False) == lazy.to_dict(as_series=False)


def test_same_rule_object_drives_both_paths():
    pd = pytest.importorskip("pandas")
    bucket = RareCategoryBucket(min_frequency=2).fit(["a", "a", "a", "b", "c", "c"])
    rule_dict = bucket.to_dict()
    pl_out = bucket.transform(pl.Series("c", ["a", "b", "z"])).to_list()
    pd_out = bucket.transform(pd.Series(["a", "b", "z"])).tolist()
    assert pl_out == pd_out
    assert bucket.to_dict() == rule_dict  # transform does not mutate the rule
