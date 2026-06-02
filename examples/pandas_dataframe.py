"""Bucket a pandas DataFrame while preserving index and columns."""

try:
    import pandas as pd
except ModuleNotFoundError:
    print('Install pandas support with: python -m pip install "pickbuckets[pandas]"')
    raise SystemExit(1) from None

from pickbuckets import AutoBucket


def main() -> None:
    df = pd.DataFrame(
        {
            "age": [22, 31, None, 47, 64],
            "segment": pd.Categorical(["free", "pro", "free", "team", "free"]),
            "signed_up": pd.to_datetime(
                ["2024-01-03", "2024-02-10", "2024-02-18", "2024-03-05", "2024-03-19"]
            ),
        },
        index=["u1", "u2", "u3", "u4", "u5"],
    )
    df["age"] = df["age"].astype("Int64")

    auto = AutoBucket(n_bins=3, min_frequency=2, ignore_unsupported=True).fit(df)
    transformed = auto.transform(df)

    print("Original:")
    print(df)
    print()
    print("Transformed:")
    print(transformed)
    print()
    print("Skipped columns:", auto.skipped_columns_)


if __name__ == "__main__":
    main()
