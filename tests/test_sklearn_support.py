import pytest

pytest.importorskip("sklearn")

import numpy as np  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import pickbuckets.sklearn as pbsk  # noqa: E402
from pickbuckets import InvalidBucketingError, Rule  # noqa: E402


def test_equal_width_works_inside_pipeline():
    X = np.array([[float(v)] for v in range(8)])
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    pipe = Pipeline(
        [("bucket", pbsk.EqualWidthBucket(n_bins=4)), ("model", LogisticRegression())]
    )
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (8,)


def test_get_params_and_set_params_round_trip():
    estimator = pbsk.AutoBucket(n_bins=4, min_frequency=3)
    params = estimator.get_params()
    assert params["n_bins"] == 4
    estimator.set_params(n_bins=7)
    assert estimator.get_params()["n_bins"] == 7
    cloned = clone(estimator)
    assert cloned.get_params()["n_bins"] == 7


def test_transform_returns_integer_codes_array():
    X = np.array([[float(v)] for v in range(8)])
    transformer = pbsk.EqualFrequencyBucket(n_bins=4).fit(X)
    out = transformer.transform(X)
    assert out.shape == (8, 1)
    assert set(np.unique(out)).issubset({0.0, 1.0, 2.0, 3.0})


def test_feature_names_out_stable():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    transformer = pbsk.EqualWidthBucket(n_bins=2).fit(X)
    assert transformer.get_feature_names_out().tolist() == ["a", "b"]


def test_transform_rejects_reordered_dataframe_features():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    transformer = pbsk.EqualWidthBucket(n_bins=2).fit(X)

    with pytest.raises(ValueError, match="Feature names must match"):
        transformer.transform(X[["b", "a"]])


def test_get_feature_names_out_validates_input_features():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    transformer = pbsk.EqualWidthBucket(n_bins=2).fit(X)

    with pytest.raises(ValueError, match="Feature names must match"):
        transformer.get_feature_names_out(["b", "a"])


def test_fitted_rules_are_portable_rule_objects():
    X = np.array([[float(v)] for v in range(8)])
    transformer = pbsk.EqualWidthBucket(n_bins=4).fit(X)
    assert all(isinstance(rule, Rule) for rule in transformer.rules_)
    # rule survives a serialization round trip
    restored = Rule.from_dict(transformer.rules_[0].to_dict())
    assert restored.edges == transformer.rules_[0].edges


def test_autobucket_handles_mixed_object_columns():
    X = np.array([[1, "A"], [2, "A"], [3, "B"], [4, "A"]], dtype=object)
    out = pbsk.AutoBucket(n_bins=2, min_frequency=2).fit_transform(X)
    assert out.shape == (4, 2)


def test_sklearn_autobucket_rejects_reordered_dataframe_features():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["A", "A", "B", "A"]})
    transformer = pbsk.AutoBucket(n_bins=2, min_frequency=2).fit(X)

    with pytest.raises(ValueError, match="Feature names must match"):
        transformer.transform(X[["b", "a"]])


def test_autobucket_rejects_feature_count_mismatch():
    X = np.array([[1, "A"], [2, "A"], [3, "B"], [4, "A"]], dtype=object)
    transformer = pbsk.AutoBucket(n_bins=2, min_frequency=2).fit(X)
    with pytest.raises(ValueError):
        transformer.transform(np.array([[1, "A", 10]], dtype=object))


def test_autobucket_rejects_unknown_numeric_strategy():
    with pytest.raises(InvalidBucketingError):
        pbsk.AutoBucket(numeric_strategy="typo")


def test_sklearn_adapters_validate_policy_parameters_early():
    with pytest.raises(InvalidBucketingError):
        pbsk.EqualWidthBucket(boundary_strategy="typo")
    with pytest.raises(InvalidBucketingError):
        pbsk.RareCategoryBucket(unknown_category_strategy="typo")
    with pytest.raises(InvalidBucketingError):
        pbsk.RareCategoryBucket(min_frequency=True)
    with pytest.raises(InvalidBucketingError):
        pbsk.AutoBucket(min_frequency=True)


def test_sklearn_adapters_reject_unknown_keep_strategy():
    with pytest.raises(InvalidBucketingError, match="stable integer code arrays"):
        pbsk.RareCategoryBucket(unknown_category_strategy="keep")
    with pytest.raises(InvalidBucketingError, match="stable integer code arrays"):
        pbsk.AutoBucket(unknown_category_strategy="keep")
