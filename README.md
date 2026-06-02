# pickbuckets

Portable bucketing rules for Python.

`pickbuckets` turns raw numerical and categorical values into human-readable,
versioned rules that can be serialized, reviewed, diffed, and applied in a
plain-Python runtime. The core package has no runtime dependencies; pandas,
Polars, and scikit-learn support live behind optional extras.

```python
from pickbuckets import EqualWidthBucket

bucket = EqualWidthBucket(n_bins=3, labels="interval")
bucket.fit([1, 2, 3, 4, 5, 6])

print(bucket.transform([1, 2.5, 6]))
print(bucket.to_json())
```

## Status

The v0.2 developer-experience release is implemented:

- one unified `Rule` model for numeric and categorical buckets
- readable dict/JSON serialization with schema and package versions
- dependency-free rule application for services and jobs
- equal-width, equal-frequency, custom-boundary, and rare-category bucketers
- `AutoBucket` for column-wise smart type dispatch
- pandas `Series`/`DataFrame` support with index and column preservation
- first-class Polars `Series`, `DataFrame`, and `LazyFrame` support
- scikit-learn-compatible transformers in `pickbuckets.sklearn`
- typed exceptions and configurable missing, boundary, and unknown-category
  policies
- tests, lint, typing, build checks, and a no-heavy-dependencies CI gate

## Install

For the dependency-free core:

```bash
python -m pip install pickbuckets
```

For optional integrations:

```bash
python -m pip install "pickbuckets[pandas]"
python -m pip install "pickbuckets[polars]"
python -m pip install "pickbuckets[sklearn]"
python -m pip install "pickbuckets[all]"
```

For development:

```bash
python -m pip install -e ".[dev,all]"
```

## Current API

```python
from pickbuckets import (
    AutoBucket,
    CustomBoundaryBucket,
    EqualFrequencyBucket,
    EqualWidthBucket,
    RareCategoryBucket,
    Rule,
)
```

Every single-column bucketer follows the same shape:

```python
bucket.fit(values)
bucket.transform(new_values)
bucket.summary()
bucket.to_dict()
bucket.to_json()
```

`transform()` uses only the saved `rules_` object; it does not need the original
training data.

## Numeric Buckets

### Equal-width

```python
from pickbuckets import EqualWidthBucket

bucket = EqualWidthBucket(n_bins=4, labels="ordinal")
bucket.fit([0, 10, 20, 30, 40])

assert bucket.transform([5, 15, 40]) == [0, 1, 3]
```

Intervals are left-closed and right-open, except the final interval, which is
closed on both sides so the maximum fitted value maps into the last bucket.

### Equal-frequency

```python
from pickbuckets import EqualFrequencyBucket

bucket = EqualFrequencyBucket(n_bins=4, duplicates="drop")
bucket.fit([1, 1, 2, 3, 5, 8, 13, 21])

print(bucket.rules_.edges)
print(bucket.transform([1, 4, 21]))
```

Repeated values can produce duplicate quantile edges. By default this raises an
error. Use `duplicates="drop"` to keep only the unique intervals that can be
formed.

### Custom boundaries

```python
from pickbuckets import CustomBoundaryBucket

bucket = CustomBoundaryBucket(
    edges=[0, 18, 35, 60, 100],
    labels=["child", "young_adult", "adult", "senior"],
    boundary_strategy="error",
).fit()

assert bucket.transform([18, 35, 60]) == ["young_adult", "adult", "senior"]
```

Custom edges must be sorted and unique. Open-ended boundaries are supported:

```python
bucket = CustomBoundaryBucket(
    edges=[float("-inf"), 0, float("inf")],
    labels=["negative", "non_negative"],
).fit()
```

When exported to JSON, infinite edges are encoded as the strings `"-inf"` and
`"inf"` so the file remains portable JSON.

## Categorical Buckets

`RareCategoryBucket` keeps frequent categories and maps rare or unseen
categories to a fallback label by default.

```python
from pickbuckets import RareCategoryBucket

bucket = RareCategoryBucket(min_frequency=2, other_label="Other")
bucket.fit(["FR", "FR", "US", "DE"])

assert bucket.transform(["FR", "US", "CA", None]) == [
    "FR",
    "Other",
    "Other",
    "Missing",
]
```

`min_frequency` accepts either an absolute count (`2`) or a ratio (`0.05`).
Category keys are stored as strings in the exported rule so JSON payloads remain
portable.

## AutoBucket

`AutoBucket` fits one rule per column and chooses the strategy from each column
dtype. Numeric columns use equal-frequency binning by default; categorical and
string columns use rare-category folding. Explicit per-column overrides win over
automatic inference.

```python
from pickbuckets import AutoBucket, EqualWidthBucket

frame = {
    "age": [18, 25, 34, 52, 70],
    "country": ["FR", "FR", "US", "DE", "DE"],
}

auto = AutoBucket(
    n_bins=3,
    min_frequency=2,
    overrides={"age": EqualWidthBucket(n_bins=3, labels="interval")},
).fit(frame)

print(auto.transform(frame))
print(auto.summary())
```

Plain `dict[str, list]`, pandas `DataFrame`, Polars `DataFrame`, and Polars
`LazyFrame` inputs return the same container type on transform.

## pandas

Install with `pickbuckets[pandas]`. Single-column bucketers preserve pandas
`Series` index and name, and `AutoBucket` preserves `DataFrame` columns and
index.

```python
import pandas as pd
from pickbuckets import AutoBucket

df = pd.DataFrame(
    {"age": [10, 20, 30, 40], "city": ["A", "A", "B", "C"]},
    index=["r0", "r1", "r2", "r3"],
)

out = AutoBucket(n_bins=2, min_frequency=2).fit_transform(df)
assert list(out.index) == list(df.index)
assert list(out.columns) == ["age", "city"]
```

Nullable numeric dtypes, object/string columns, and categorical dtypes are
handled by the same saved rule model as the dependency-free runtime.

## Polars

Install with `pickbuckets[polars]`. Polars support uses native expressions for
transform, not row-wise Python loops. Eager frames and lazy queries are both
supported for non-raising rules.

```python
import polars as pl
from pickbuckets import AutoBucket

df = pl.DataFrame(
    {"age": [10, 20, 30, 40], "city": ["A", "A", "B", "C"]}
)

auto = AutoBucket(n_bins=2, min_frequency=2, labels="interval").fit(df)
eager = auto.transform(df)
lazy = auto.transform(df.lazy()).collect()

assert eager.to_dict(as_series=False) == lazy.to_dict(as_series=False)
```

Rules configured with an `error` strategy for missing values, numeric
boundaries, or unknown categories require eager evaluation so the library can
raise the same typed exceptions as the pure-Python runtime.

## scikit-learn

Install with `pickbuckets[sklearn]`. The sklearn adapters live in
`pickbuckets.sklearn` so importing `pickbuckets` stays dependency-free.

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import pickbuckets.sklearn as pbsk

X = np.array([[float(v)] for v in range(8)])
y = [0, 0, 0, 0, 1, 1, 1, 1]

pipe = Pipeline(
    [("bucket", pbsk.EqualWidthBucket(n_bins=4)), ("model", LogisticRegression())]
)
pipe.fit(X, y)
```

The adapters expose `get_params`, `set_params`, `fit_transform`,
`get_feature_names_out`, and fitted portable `Rule` objects on `rules_`.
Transforms return integer code arrays suitable for downstream estimators.

See [docs/integrations.md](docs/integrations.md) for the phase 2 integration
details and CI coverage notes.

## Policies And Errors

Policies are stored on the unified rule and survive serialization.

Missing values:

- `missing_strategy="separate"` maps missing values to `missing_label`
  (default: `"Missing"`).
- `missing_strategy="propagate"` returns the original missing value.
- `missing_strategy="error"` raises `MissingValueError`.

Numeric boundaries:

- `boundary_strategy="clip"` maps values below/above the fitted range to the
  first/last label.
- `boundary_strategy="error"` raises `BoundaryError`.

Unknown categories:

- `unknown_category_strategy="other"` maps unseen categories to `other_label`.
- `unknown_category_strategy="error"` raises `UnknownCategoryError`.

Malformed numeric transform values raise `InvalidBucketingError`.

## Rule Portability

Every fitted bucketer exposes a `rules_` object:

```python
from pickbuckets import EqualWidthBucket

bucket = EqualWidthBucket(n_bins=2).fit([0, 10])
payload = bucket.to_json()

restored = EqualWidthBucket.from_json(payload)
assert restored.transform([1, 9]) == bucket.transform([1, 9])
```

You can also apply a saved rule without constructing the original bucketer:

```python
from pickbuckets import Rule
from pickbuckets.runtime import apply_rule

rule = Rule.from_json(payload)
assert apply_rule(rule, [0, 5, 10, None]) == [0, 1, 1, "Missing"]
```

Rules include:

- `schema_version`
- `package_version`
- `kind`
- numeric `edges` or categorical `category_mapping`
- output `labels`
- active missing, boundary, and unknown-category policies
- `fit_stats`

Loading a rule with an unsupported major schema version or an unknown policy
value raises `RuleSchemaError`.

## Development

Run the full local check set before opening a PR:

```bash
ruff check .
mypy src/pickbuckets
pytest
python -m build
```

Useful project principles:

- Keep the base import dependency-free.
- Put integrations behind optional extras.
- Fit once, transform from saved rules.
- Add serialization round-trip tests for every new bucketer.
- Prefer clear typed exceptions over implicit coercion.
