"""Apply AutoBucket inside a Polars lazy query."""

try:
    import polars as pl
except ModuleNotFoundError:
    print('Install Polars support with: python -m pip install "pickbuckets[polars]"')
    raise SystemExit(1) from None

from pickbuckets import AutoBucket


def main() -> None:
    df = pl.DataFrame(
        {
            "score": [0.10, 0.40, 0.55, 0.90, None],
            "plan": ["free", "free", "pro", "team", "free"],
        }
    )

    auto = AutoBucket(n_bins=3, min_frequency=2, labels="interval").fit(df)

    lazy_result = (
        auto.transform(df.lazy())
        .filter(pl.col("plan") != "Other")
        .with_columns(pl.lit("kept").alias("status"))
    )

    print("Lazy result:")
    print(lazy_result.collect().to_dict(as_series=False))


if __name__ == "__main__":
    main()
