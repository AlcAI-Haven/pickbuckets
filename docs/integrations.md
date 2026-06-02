# Phase 2 Integrations

This document describes the optional integration surface shipped in
`pickbuckets` v0.2. The core package remains dependency-free; adapters import
pandas, Polars, NumPy, or scikit-learn only when the caller uses objects from
those libraries or imports the sklearn adapter module explicitly.

## Dependency Extras

```bash
python -m pip install "pickbuckets[pandas]"
python -m pip install "pickbuckets[polars]"  # Python 3.10+
python -m pip install "pickbuckets[sklearn]"
python -m pip install "pickbuckets[all]"
```

Development and CI use:

```bash
python -m pip install -e ".[dev,all]"
```

## pandas

Single-column bucketers accept `pandas.Series` and return a `Series` with the
same index and name.

`AutoBucket` accepts `pandas.DataFrame`, fits one rule per supported column, and
returns a `DataFrame` with the original index and column order. Numeric dtypes
are bucketed with the configured numeric strategy; string, object, and
categorical dtypes are bucketed with rare-category folding.

Unsupported columns raise `InvalidBucketingError` by default. Pass
`ignore_unsupported=True` to leave unsupported columns unchanged on transform.
Transform input must contain the same fitted columns; missing or unexpected
columns raise `InvalidBucketingError`.

## Polars

Polars integration requires Python 3.10 or newer. Single-column bucketers accept
`polars.Series` and return a `Series` with the same name.

`AutoBucket` accepts `polars.DataFrame` and `polars.LazyFrame`. Transform uses
native Polars expressions, so it composes inside an eager frame update or lazy
query plan without a pandas conversion.

Rules using `missing_strategy="error"`, `boundary_strategy="error"`, or
`unknown_category_strategy="error"` require eager evaluation. This keeps the
typed exception behavior aligned with the pure-Python runtime.

`unknown_category_strategy="keep"` is supported when outputs remain string
typed. Mixed-type outputs with `keep` are rejected because Polars cannot
preserve arbitrary kept unknown strings and non-string labels in a native
expression.

## scikit-learn

The sklearn adapters live in `pickbuckets.sklearn`:

```python
import pickbuckets.sklearn as pbsk
```

They implement sklearn-style `fit`, `transform`, `fit_transform`,
`get_feature_names_out`, `get_params`, and `set_params`. Transforms return
integer code arrays, which are suitable for downstream estimators in
`sklearn.pipeline.Pipeline`.

Feature names must match the fit-time order on transform. The sklearn adapters
reject `unknown_category_strategy="keep"` because they return stable integer
code arrays rather than arbitrary new string labels.

Each fitted sklearn adapter exposes its portable `Rule` objects on `rules_`, so
rules can still be serialized and applied outside sklearn.

## AutoBucket

`AutoBucket` supports:

- `dict[str, list]`
- `pandas.DataFrame`
- `polars.DataFrame`
- `polars.LazyFrame`

It records:

- `rules_`: per-column `Rule` objects
- `feature_names_in_`: input column order
- `n_features_in_`: input column count
- `skipped_columns_`: unsupported columns passed through because
  `ignore_unsupported=True`

Use `overrides={"column": bucketer}` to force a specific bucketer for a column.
Overrides are fitted like automatically selected bucketers and receive the
column name in the exported rule. Unknown override columns and non-bucketer
override values raise `InvalidBucketingError`.

## CI Coverage

The GitHub Actions workflow runs on Python 3.9 through 3.13 and installs
`.[dev,all]`. Python 3.9 exercises the dependency-free core plus integrations
whose upstream packages still support 3.9; Polars tests run on Python 3.10+.
The workflow also runs ruff, mypy, pytest, and package build checks.

A separate no-heavy-dependencies job installs only the base package and pytest,
then verifies that importing `pickbuckets` does not import pandas, Polars, or
sklearn.
