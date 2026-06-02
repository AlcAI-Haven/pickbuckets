"""Use AutoBucket on a dependency-free dict-of-lists frame."""

from pickbuckets import AutoBucket, EqualWidthBucket


def main() -> None:
    frame = {
        "age": [18, 25, 34, 52, 70, 88],
        "country": ["FR", "FR", "US", "DE", "FR", "ES"],
        "is_trial": [True, False, False, True, False, False],
    }

    auto = AutoBucket(
        n_bins=3,
        min_frequency=2,
        labels="interval",
        overrides={"age": EqualWidthBucket(n_bins=3, labels="interval")},
    ).fit(frame)

    print("Transformed frame:")
    print(auto.transform(frame))
    print()
    print("Column summaries:")
    print(auto.summary())


if __name__ == "__main__":
    main()
