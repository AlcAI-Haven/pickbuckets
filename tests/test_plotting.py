from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

from pickbuckets import EqualWidthBucket, RareCategoryBucket, WoEBucket
from pickbuckets.exceptions import InvalidBucketingError, NotFittedError
from pickbuckets.plotting import plot_bucket_counts, plot_target_rate, plot_woe


def test_plot_bucket_counts_numeric_returns_axes() -> None:
    bucket = EqualWidthBucket(n_bins=4).fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax = plot_bucket_counts(bucket)
    assert ax.__class__.__name__.endswith("Axes")
    assert len(ax.patches) == 4


def test_plot_bucket_counts_categorical() -> None:
    bucket = RareCategoryBucket(min_frequency=2).fit(["a", "a", "b", "b", "c"])
    ax = plot_bucket_counts(bucket)
    assert len(ax.patches) >= 1


def test_plot_accepts_rule_directly() -> None:
    bucket = EqualWidthBucket(n_bins=3).fit([0, 5, 10])
    ax = plot_bucket_counts(bucket.rules_)
    assert ax.__class__.__name__.endswith("Axes")


def test_plot_uses_supplied_axes() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    bucket = EqualWidthBucket(n_bins=3).fit([0, 5, 10])
    returned = plot_bucket_counts(bucket, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_user_can_customize_after_creation() -> None:
    bucket = EqualWidthBucket(n_bins=3).fit([0, 5, 10])
    ax = plot_bucket_counts(bucket)
    ax.set_title("custom")
    assert ax.get_title() == "custom"


def test_target_rate_and_woe_plots() -> None:
    bucket = WoEBucket(n_bins=3, output="woe").fit(
        [10, 20, 30, 40, 50, 60], [0, 0, 0, 1, 1, 1]
    )
    rate_ax = plot_target_rate(bucket)
    assert rate_ax.get_ylabel() == "target rate"

    woe_ax = plot_woe(bucket)
    assert woe_ax.get_ylabel() == "WoE"
    assert len(woe_ax.patches) == len(bucket.summary_table())


def test_target_rate_requires_supervised_rule() -> None:
    bucket = EqualWidthBucket(n_bins=3).fit([0, 5, 10])
    with pytest.raises(InvalidBucketingError):
        plot_target_rate(bucket)


def test_plot_requires_fitted_object() -> None:
    with pytest.raises(NotFittedError):
        plot_bucket_counts(object())
