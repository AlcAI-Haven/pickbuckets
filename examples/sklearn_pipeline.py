"""Use pickbuckets inside a scikit-learn Pipeline."""

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
except ModuleNotFoundError:
    print('Install sklearn support with: python -m pip install "pickbuckets[sklearn]"')
    raise SystemExit(1) from None

import pickbuckets.sklearn as pbsk


def main() -> None:
    X = np.array([[float(value)] for value in range(10)])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    pipeline = Pipeline(
        [
            ("bucket", pbsk.EqualWidthBucket(n_bins=4)),
            ("model", LogisticRegression()),
        ]
    )
    pipeline.fit(X, y)

    print("Feature names:")
    print(pipeline.named_steps["bucket"].get_feature_names_out())
    print()
    print("Predictions:")
    print(pipeline.predict([[1.5], [6.5], [9.0]]))


if __name__ == "__main__":
    main()
