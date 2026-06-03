# Usage Guide

Worked examples for every `pickbuckets` bucketer and policy. For a terse symbol
lookup see [api-reference.md](api-reference.md); for the optional dataframe and
scikit-learn adapters see [integrations.md](integrations.md).

## Contents

- [The common bucketer shape](#the-common-bucketer-shape)
- [Numeric buckets](#numeric-buckets)
- [Categorical buckets](#categorical-buckets)
- [AutoBucket](#autobucket)
- [Supervised buckets](#supervised-buckets)
- [Policies and errors](#policies-and-errors)
- [Rule portability](#rule-portability)
- [Streaming (experimental)](#streaming-experimental)
- [Plotting](#plotting)

## The common bucketer shape

Every single-column bucketer follows the same shape:

```python
bucket.fit(values)          # supervised bucketers: fit(values, y)
bucket.transform(new_values)
bucket.summary()
bucket.to_dict()
bucket.to_json()
```

`transform()` uses only the saved `rules_` object; it does not need the original
training data.

## Numeric buckets

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

## Categorical buckets

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
`LazyFrame` inputs return the same container type on transform. Transform input
must contain the same column names fitted by the bucketer; missing or unexpected
columns raise `InvalidBucketingError`. See [integrations.md](integrations.md) for
the dataframe adapters.

## Supervised buckets

Supervised bucketers require `fit(values, y)` and never require `y` during
`transform()`. Target data is stored only as aggregate fit statistics such as
counts, target rates, WoE, and IV.

```python
from pickbuckets import WoEBucket

bucket = WoEBucket(
    n_bins=3,
    output="woe",
    min_bin_size=0.1,
    monotonic="auto",
    smoothing=0.5,
)
bucket.fit([10, 20, 30, 40, 50, 60], [0, 0, 0, 1, 1, 1])

print(bucket.transform([15, 45]))
print(bucket.iv_summary())
print(bucket.summary_table())
```

`DecisionTreeBucket` learns numeric edges from a shallow scikit-learn decision
tree, so it requires `pickbuckets[sklearn]`. `ChiMergeBucket` and `WoEBucket`
are dependency-free. `WoEBucket` supports count- or ratio-based `min_bin_size`
constraints and optional monotonic WoE constraints with `monotonic="ascending"`,
`"descending"`, or `"auto"`. `ExternalSplitBucket` imports externally learned
split points, for example from OptBinning, into a normal portable numeric rule.

## Policies and errors

Policies are stored on the unified rule and survive serialization.

Missing values:

- `missing_strategy="separate"` maps missing values to `missing_label`
  (default: `"Missing"`).
- `missing_strategy="most_frequent"` maps missing values to the most frequent
  fitted output label. The learned replacement is stored on the rule as
  `missing_label`.
- `missing_strategy="propagate"` returns the original missing value.
- `missing_strategy="error"` raises `MissingValueError`.

Numeric boundaries:

- `boundary_strategy="clip"` maps values below/above the fitted range to the
  first/last label.
- `boundary_strategy="underflow_overflow"` maps values below/above the fitted
  range to `underflow_label` and `overflow_label`.
- `boundary_strategy="error"` raises `BoundaryError`.

Unknown categories:

- `unknown_category_strategy="other"` maps unseen categories to `other_label`.
- `unknown_category_strategy="missing"` maps unseen categories to
  `missing_label`.
- `unknown_category_strategy="keep"` returns unseen categories as strings.
- `unknown_category_strategy="error"` raises `UnknownCategoryError`.

Malformed numeric transform values raise `InvalidBucketingError`. Data-related
errors include the fitted feature name when one is available.

## Rule portability

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
- underflow and overflow labels
- `fit_stats`

Loading a rule with an unsupported major schema version or an unknown policy
value raises `RuleSchemaError`.

## Streaming (experimental)

`pickbuckets.experimental` holds online approximations, kept deliberately
separate from the exact bucketers. `StreamingEqualFrequencyBucket` fits
approximate quantile edges from data that does not fit in memory, then
materializes a normal portable `Rule`.

```python
from pickbuckets.experimental import StreamingEqualFrequencyBucket

bucket = StreamingEqualFrequencyBucket(n_bins=10, max_centroids=256)
for chunk in chunks_of_data:        # data need not fit in memory
    bucket.partial_fit(chunk)
bucket.finalize()                   # build a standard portable Rule

bucket.transform([1.0, 2.0])        # applies via the normal runtime
print(bucket.to_json())
```

Only fitting is approximate. Error is bounded by `max_centroids` and the edges
match the exact bucketer once it reaches the number of distinct values. APIs in
`experimental` may change between minor releases.

## Plotting

Install with `pickbuckets[plot]`. The helpers in `pickbuckets.plotting` return a
matplotlib `Axes` so you can customize and decide when to display.

```python
from pickbuckets import WoEBucket
from pickbuckets.plotting import plot_bucket_counts, plot_target_rate, plot_woe

bucket = WoEBucket(n_bins=4).fit([10, 20, 30, 40, 50, 60], [0, 0, 0, 1, 1, 1])

ax = plot_woe(bucket)
ax.set_title("WoE by score band")
ax.figure.savefig("woe.png")
```

`plot_bucket_counts` works for any fitted bucketer or `Rule`; `plot_target_rate`
and `plot_woe` require a supervised bucketer. Plotting is never required for the
core package.
