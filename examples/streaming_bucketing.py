"""Fit approximate equal-frequency edges from data fed in chunks.

The streaming bucketer keeps a bounded sketch, so the data never has to fit in
memory at once. Fitting is approximate; the resulting rule is an ordinary
portable rule that applies through the normal runtime.
"""

import random

from pickbuckets.experimental import StreamingEqualFrequencyBucket


def chunks(n: int, size: int, seed: int = 0):
    rng = random.Random(seed)
    produced = 0
    while produced < n:
        take = min(size, n - produced)
        yield [rng.gauss(0.0, 1.0) for _ in range(take)]
        produced += take


def main() -> None:
    bucket = StreamingEqualFrequencyBucket(n_bins=5, max_centroids=256)
    for chunk in chunks(n=1_000_000, size=50_000):
        bucket.partial_fit(chunk)
    bucket.finalize()

    print("Approximate edges:")
    print(bucket.rules_.edges)
    print()
    print("Applied via the normal runtime:")
    print(bucket.transform([-2.0, 0.0, 2.0]))
    print()
    print("Portable rule (truncated):")
    print(bucket.to_json()[:200], "...")


if __name__ == "__main__":
    main()
