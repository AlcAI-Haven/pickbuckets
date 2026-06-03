from __future__ import annotations

import random

import pytest

import pickbuckets
from pickbuckets import EqualFrequencyBucket
from pickbuckets.exceptions import InvalidBucketingError, NotFittedError
from pickbuckets.experimental import (
    StreamingEqualFrequencyBucket,
    StreamingHistogram,
)
from pickbuckets.rules import Rule
from pickbuckets.runtime import apply_rule


def test_streaming_is_not_exported_from_top_level() -> None:
    assert not hasattr(pickbuckets, "StreamingEqualFrequencyBucket")


def test_partial_fit_processes_chunks() -> None:
    bucket = StreamingEqualFrequencyBucket(n_bins=4)
    bucket.partial_fit([1, 2, 3])
    bucket.partial_fit([4, 5, 6, 7, 8])
    bucket.finalize()
    assert bucket.rules_.kind == "numeric"
    assert len(bucket.rules_.labels) == 4
    assert bucket.rules_.fit_stats["n_observations"] == 8
    assert bucket.rules_.fit_stats["approximate"] is True


def test_transform_auto_finalizes() -> None:
    bucket = StreamingEqualFrequencyBucket(n_bins=2)
    bucket.partial_fit(range(100))
    out = bucket.transform([0, 99])
    assert out == [0, 1]


def test_edges_are_sorted_and_monotonic() -> None:
    bucket = StreamingEqualFrequencyBucket(n_bins=5, max_centroids=64)
    bucket.fit(range(1000))
    edges = bucket.rules_.edges or []
    assert edges == sorted(edges)
    assert edges[0] == 0.0
    assert edges[-1] == 999.0


def test_approximation_close_to_exact() -> None:
    rng = random.Random(7)
    data = [rng.gauss(0.0, 1.0) for _ in range(20_000)]

    exact = EqualFrequencyBucket(n_bins=10, duplicates="drop").fit(data)
    streaming = StreamingEqualFrequencyBucket(
        n_bins=10, max_centroids=256, duplicates="drop"
    )
    for start in range(0, len(data), 1000):
        streaming.partial_fit(data[start : start + 1000])
    streaming.finalize()

    exact_edges = exact.rules_.edges or []
    approx_edges = streaming.rules_.edges or []
    assert len(exact_edges) == len(approx_edges)
    spread = max(data) - min(data)
    for exact_edge, approx_edge in zip(exact_edges, approx_edges):
        assert abs(exact_edge - approx_edge) < 0.05 * spread


def test_exact_when_centroids_exceed_distinct_values() -> None:
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    streaming = StreamingEqualFrequencyBucket(
        n_bins=4, max_centroids=256, duplicates="drop"
    ).fit(data)
    exact = EqualFrequencyBucket(n_bins=4, duplicates="drop").fit(data)
    assert streaming.rules_.edges == exact.rules_.edges


def test_serialization_round_trip() -> None:
    bucket = StreamingEqualFrequencyBucket(n_bins=3).fit(range(30))
    payload = bucket.to_json()
    rule = Rule.from_json(payload)
    assert apply_rule(rule, [0, 15, 29]) == bucket.transform([0, 15, 29])


def test_missing_values_counted_not_binned() -> None:
    bucket = StreamingEqualFrequencyBucket(n_bins=2).fit([1, 2, None, 3, 4])
    assert bucket.rules_.fit_stats["n_missing"] == 1
    assert bucket.rules_.fit_stats["n_observations"] == 4
    assert bucket.transform([None]) == ["Missing"]


def test_empty_input_raises() -> None:
    bucket = StreamingEqualFrequencyBucket(n_bins=2)
    with pytest.raises(InvalidBucketingError):
        bucket.fit([])


def test_transform_before_fit_raises() -> None:
    bucket = StreamingEqualFrequencyBucket(n_bins=2)
    with pytest.raises(NotFittedError):
        bucket.transform([1, 2, 3])


def test_non_numeric_raises() -> None:
    bucket = StreamingEqualFrequencyBucket(n_bins=2)
    with pytest.raises(InvalidBucketingError):
        bucket.fit(["a", "b"])


def test_invalid_max_centroids() -> None:
    with pytest.raises(InvalidBucketingError):
        StreamingEqualFrequencyBucket(n_bins=2, max_centroids=1)


def test_histogram_merge_matches_single_stream() -> None:
    rng = random.Random(11)
    data = [rng.gauss(0.0, 1.0) for _ in range(5000)]

    single = StreamingHistogram(max_centroids=128)
    for value in data:
        single.update(value)

    left = StreamingHistogram(max_centroids=128)
    right = StreamingHistogram(max_centroids=128)
    for value in data[: len(data) // 2]:
        left.update(value)
    for value in data[len(data) // 2 :]:
        right.update(value)
    left.merge(right)

    assert left.total == single.total
    spread = max(data) - min(data)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert abs(left.quantile(q) - single.quantile(q)) < 0.1 * spread
