import math

import pytest

from pickbuckets import (
    ChiMergeBucket,
    DecisionTreeBucket,
    ExternalSplitBucket,
    InvalidBucketingError,
    Rule,
    WoEBucket,
)


def test_supervised_fit_requires_target_values():
    with pytest.raises(InvalidBucketingError, match="fit\\(X, y\\)"):
        WoEBucket(n_bins=2).fit([0, 1, 2])


def test_supervised_fit_validates_target_length():
    with pytest.raises(InvalidBucketingError, match="same length"):
        WoEBucket(n_bins=2).fit([0, 1, 2], [0, 1])


def test_supervised_fit_validates_pandas_target_alignment():
    pd = pytest.importorskip("pandas")

    values = pd.Series([0, 1, 2, 3], index=["a", "b", "c", "d"])
    y = pd.Series([0, 0, 1, 1], index=["d", "c", "b", "a"])

    with pytest.raises(InvalidBucketingError, match="indexes must be aligned"):
        WoEBucket(n_bins=2).fit(values, y)


def test_supervised_fit_accepts_matching_pandas_target_alignment():
    pd = pytest.importorskip("pandas")

    values = pd.Series([0, 1, 2, 3], index=["a", "b", "c", "d"])
    y = pd.Series([0, 0, 1, 1], index=["a", "b", "c", "d"])

    assert WoEBucket(n_bins=2).fit(values, y).transform([0, 3]) == [0, 1]


def test_decision_tree_rejects_invalid_max_leaf_nodes_early():
    with pytest.raises(InvalidBucketingError, match="max_leaf_nodes"):
        DecisionTreeBucket(max_leaf_nodes=1)


def test_decision_tree_bucket_extracts_tree_thresholds():
    pytest.importorskip("sklearn")

    values = list(range(10))
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    bucket = DecisionTreeBucket(max_leaf_nodes=2, min_samples_leaf=2).fit(values, y)

    assert bucket.rules_.edges == pytest.approx([0.0, 4.5, 9.0])
    assert bucket.transform([0, 4, 5, 9]) == [0, 0, 1, 1]
    restored = DecisionTreeBucket.from_dict(bucket.to_dict())
    assert restored.transform([0, 5, 9]) == [0, 1, 1]
    assert Rule.from_dict(bucket.to_dict()).fit_stats["algorithm"] == "decision_tree"


def test_woe_bucket_computes_finite_woe_and_hand_checked_iv():
    bucket = WoEBucket(n_bins=2, smoothing=0.5).fit(
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    )

    rows = bucket.summary_table()
    assert all(math.isfinite(row["woe"]) for row in rows)
    expected_iv = 2 * ((1 / 6 - 5 / 6) * math.log((1 / 6) / (5 / 6)))
    assert bucket.iv_ == pytest.approx(expected_iv)
    assert bucket.iv_summary()["iv"] == pytest.approx(expected_iv)
    assert rows[0]["event_count"] == 0
    assert rows[0]["non_event_count"] == 2
    assert "targets" not in bucket.rules_.fit_stats


@pytest.mark.parametrize("smoothing", [0, -0.1, True, float("inf")])
def test_woe_bucket_rejects_invalid_smoothing(smoothing):
    with pytest.raises(InvalidBucketingError, match="smoothing"):
        WoEBucket(n_bins=2, smoothing=smoothing)


@pytest.mark.parametrize("min_bin_size", [0, -0.1, True, 1.5, float("inf")])
def test_woe_bucket_rejects_invalid_min_bin_size(min_bin_size):
    with pytest.raises(InvalidBucketingError, match="min_bin_size"):
        WoEBucket(n_bins=2, min_bin_size=min_bin_size)


@pytest.mark.parametrize("monotonic", ["up", True])
def test_woe_bucket_rejects_invalid_monotonic_policy(monotonic):
    with pytest.raises(InvalidBucketingError, match="monotonic"):
        WoEBucket(n_bins=2, monotonic=monotonic)


def test_woe_bucket_min_bin_size_merges_adjacent_bins():
    bucket = WoEBucket(n_bins=4, min_bin_size=3, smoothing=0.5).fit(
        list(range(8)),
        [0, 0, 0, 1, 1, 1, 1, 1],
    )

    rows = bucket.summary_table()
    assert all(row["count"] >= 3 for row in rows)
    assert bucket.rules_.fit_stats["actual_bins"] == len(rows)
    assert bucket.rules_.fit_stats["actual_bins"] < 4
    assert bucket.rules_.fit_stats["min_bin_count"] == 3
    assert bucket.rules_.fit_stats["min_bin_size_merge_count"] > 0


def test_woe_bucket_monotonic_constraint_merges_violations():
    bucket = WoEBucket(n_bins=4, monotonic="ascending", smoothing=0.5).fit(
        list(range(8)),
        [0, 0, 1, 1, 0, 0, 1, 1],
    )

    woes = [row["woe"] for row in bucket.summary_table()]
    assert woes == sorted(woes)
    assert bucket.rules_.fit_stats["monotonic_direction"] == "ascending"
    assert bucket.rules_.fit_stats["monotonic_merge_count"] > 0
    assert bucket.rules_.fit_stats["actual_bins"] < 4


def test_woe_bucket_auto_monotonic_records_resolved_direction():
    bucket = WoEBucket(n_bins=4, monotonic="auto", smoothing=0.5).fit(
        list(range(8)),
        [1, 1, 0, 0, 1, 1, 0, 0],
    )

    woes = [row["woe"] for row in bucket.summary_table()]
    assert woes == sorted(woes, reverse=True)
    assert bucket.rules_.fit_stats["monotonic_direction"] == "descending"


def test_woe_bucket_can_transform_to_woe_values_and_round_trip():
    bucket = WoEBucket(n_bins=2, output="woe", smoothing=0.5).fit(
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    )
    restored = WoEBucket.from_json(bucket.to_json())

    assert bucket.transform([0, 3]) == restored.transform([0, 3])
    assert restored.iv_ == pytest.approx(bucket.iv_)
    assert restored.target_summary_ == bucket.target_summary_
    assert bucket.transform([0, 3]) == [
        bucket.rules_.labels[0],
        bucket.rules_.labels[1],
    ]


def test_woe_bucket_requires_binary_targets():
    with pytest.raises(InvalidBucketingError, match="exactly two"):
        WoEBucket(n_bins=2).fit([0, 1, 2], [0, 1, 2])


def test_chimerge_respects_max_bins_and_is_order_stable():
    values = [0, 1, 2, 3, 4, 5, 6, 7]
    y = [0, 0, 0, 1, 1, 1, 1, 1]
    sorted_bucket = ChiMergeBucket(max_bins=3, initial_bins=6).fit(values, y)
    shuffled_bucket = ChiMergeBucket(max_bins=3, initial_bins=6).fit(
        [4, 0, 7, 2, 6, 1, 5, 3],
        [1, 0, 1, 0, 1, 0, 1, 1],
    )

    assert sorted_bucket.rules_.fit_stats["actual_bins"] <= 3
    assert sorted_bucket.rules_.edges == shuffled_bucket.rules_.edges
    assert len(sorted_bucket.summary_table()) == sorted_bucket.rules_.fit_stats[
        "actual_bins"
    ]


def test_chimerge_rejects_bool_numeric_controls():
    with pytest.raises(InvalidBucketingError, match="min_bin_size"):
        ChiMergeBucket(min_bin_size=True)
    with pytest.raises(InvalidBucketingError, match="min_chi2"):
        ChiMergeBucket(min_chi2=True)
    with pytest.raises(InvalidBucketingError, match="smoothing"):
        ChiMergeBucket(smoothing=False)


def test_external_split_bucket_imports_portable_thresholds():
    bucket = ExternalSplitBucket([10, 20], labels=["low", "mid", "high"]).fit()
    restored = ExternalSplitBucket.from_dict(bucket.to_dict())

    assert bucket.rules_.edges == [float("-inf"), 10.0, 20.0, float("inf")]
    assert restored.transform([5, 10, 25]) == ["low", "mid", "high"]
    assert bucket.rules_.fit_stats["algorithm"] == "external_splits"


def test_external_split_bucket_validates_splits():
    with pytest.raises(InvalidBucketingError, match="sorted and unique"):
        ExternalSplitBucket([2, 1])
    with pytest.raises(InvalidBucketingError, match="sequence of numbers"):
        ExternalSplitBucket("10")
