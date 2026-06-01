# pickbuckets

Portable bucketing rules for Python.

`pickbuckets` turns raw numerical and categorical values into human-readable,
versioned rules that can be serialized, reviewed, diffed, and applied in a
plain-Python runtime. The core package has no runtime dependencies.

```python
from pickbuckets import EqualWidthBucket

bucket = EqualWidthBucket(n_bins=3, labels="interval")
bucket.fit([1, 2, 3, 4, 5, 6])

print(bucket.transform([1, 2.5, 6]))
print(bucket.to_json())
```

## Status

The v0.1 core is implemented:

- one unified `Rule` model for numeric and categorical buckets
- readable dict/JSON serialization with schema and package versions
- dependency-free rule application for services and jobs
- equal-width, equal-frequency, custom-boundary, and rare-category bucketers
- typed exceptions and configurable missing, boundary, and unknown-category
  policies
- tests, lint, typing, and a no-heavy-dependencies CI gate

The roadmap lives in [pickbuckets-roadmap](pickbuckets-roadmap/README.md).
The next milestone is Phase 2: scikit-learn, pandas, and first-class Polars
integrations around the stable rule engine.

## Install

For development:

```bash
python -m pip install -e ".[dev]"
```

For optional integrations planned after the core:

```bash
python -m pip install "pickbuckets[pandas]"
python -m pip install "pickbuckets[polars]"
python -m pip install "pickbuckets[sklearn]"
```

Those extras are declared for the roadmap, but integration modules are not part
of the current v0.1 core yet.

## Current API

```python
from pickbuckets import (
    CustomBoundaryBucket,
    EqualFrequencyBucket,
    EqualWidthBucket,
    RareCategoryBucket,
    Rule,
)
```

Every bucketer follows the same shape:

```python
bucket.fit(values)
bucket.transform(new_values)
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
```

Useful project principles:

- Keep the base import dependency-free.
- Put integrations behind optional extras.
- Fit once, transform from saved rules.
- Add serialization round-trip tests for every new bucketer.
- Prefer clear typed exceptions over implicit coercion.
