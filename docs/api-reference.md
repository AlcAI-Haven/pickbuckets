# API Reference

`pickbuckets` turns raw values into portable, versioned bucketing rules. The
headline is portability: **fit with the full stack, apply with none of it.**

## Portability first: export → apply without the stack

Fit a bucketer anywhere (with pandas, Polars, or scikit-learn installed),
serialize the rule, and apply it in a lightweight service using only the
standard library.

```python
# --- training environment (has the full stack) ---
from pickbuckets import EqualFrequencyBucket

bucket = EqualFrequencyBucket(n_bins=4, duplicates="drop").fit(training_values)
payload = bucket.to_json()  # store this string anywhere

# --- serving environment (stdlib only, no pandas/Polars/sklearn) ---
from pickbuckets import Rule
from pickbuckets.runtime import apply_rule

rule = Rule.from_json(payload)
codes = apply_rule(rule, [0, 5, 10, None])
```

`apply_rule` is pure Python (optionally NumPy). The same JSON applies
identically in pandas, Polars, scikit-learn pipelines, and the runtime.

## Core bucketers

All single-column bucketers share the same shape:

```python
bucket.fit(values)          # supervised bucketers: fit(values, y)
bucket.transform(values)    # uses only the saved rule
bucket.summary()
bucket.to_dict() / bucket.to_json()
Bucket.from_dict(d) / Bucket.from_json(s)
```

| Class | Module | Notes |
|---|---|---|
| `EqualWidthBucket` | `pickbuckets` | Min/max range split into equal-width bins. |
| `EqualFrequencyBucket` | `pickbuckets` | Quantile edges; `duplicates="raise"\|"drop"`. |
| `CustomBoundaryBucket` | `pickbuckets` | Manual, validated edges (supports `±inf`). |
| `RareCategoryBucket` | `pickbuckets` | Folds rare/unseen categories to a fallback. |
| `AutoBucket` | `pickbuckets` | One rule per column, dtype-driven dispatch. |

## Supervised bucketers

`fit(X, y)` is required; `transform` never needs `y`. Only aggregate fit
statistics are stored on the rule — never raw targets.

| Class | Module | Notes |
|---|---|---|
| `DecisionTreeBucket` | `pickbuckets` | Edges from a shallow sklearn tree (needs `[sklearn]`). |
| `WoEBucket` | `pickbuckets` | WoE/IV, smoothing, min-bin-size, monotonic constraints. |
| `ChiMergeBucket` | `pickbuckets` | Chi-square adjacent-bin merging. |
| `ExternalSplitBucket` | `pickbuckets` | Import external splits (e.g. OptBinning) into a portable rule. |

Supervised extras: `summary_table()`, `iv_summary()`.

## The `Rule` model

`pickbuckets.Rule` is the single serializable model for numeric and categorical
buckets. Key fields: `kind`, `edges` or `category_mapping`, `labels`, the active
missing/boundary/unknown-category policies, `fit_stats`, `schema_version`, and
`package_version`. Loading an unsupported major schema version or an unknown
policy raises `RuleSchemaError`.

## Runtime

`pickbuckets.runtime.apply_rule(rule, values) -> list` applies a saved rule with
the standard library only. This path is exercised by a CI job with pandas,
Polars, and scikit-learn uninstalled.

## Adapters (optional extras)

- pandas (`pickbuckets[pandas]`): `Series`/`DataFrame` index and column
  preservation.
- Polars (`pickbuckets[polars]`, Python 3.10+): native vectorized expressions
  for `Series`/`DataFrame`/`LazyFrame`.
- scikit-learn (`pickbuckets[sklearn]`): transformers in `pickbuckets.sklearn`
  with `get_feature_names_out`, `get_params`/`set_params`, and `rules_`.

## Plotting (optional extra)

`pickbuckets.plotting` (install `pickbuckets[plot]`) returns matplotlib `Axes`:

- `plot_bucket_counts(bucket_or_rule, ax=None, **bar_kwargs)`
- `plot_target_rate(bucket_or_rule, ax=None, **plot_kwargs)` (supervised)
- `plot_woe(bucket_or_rule, ax=None, **bar_kwargs)` (supervised)

Plotting is never required for the core package and never forces display.

## Experimental: streaming (unstable API)

`pickbuckets.experimental` holds online approximations kept separate from the
exact bucketers. APIs here may change between minor releases.

```python
from pickbuckets.experimental import StreamingEqualFrequencyBucket

bucket = StreamingEqualFrequencyBucket(n_bins=10, max_centroids=256)
for chunk in chunks_of_data:           # data need not fit in memory
    bucket.partial_fit(chunk)
bucket.finalize()                       # materialise a standard portable Rule

bucket.transform([1.0, 2.0])            # applies via the normal runtime
bucket.to_json()
```

Fitting is approximate (a bounded streaming-histogram quantile sketch);
approximation error shrinks as `max_centroids` grows and is exact once it
reaches the number of distinct values. Transform and serialization are exact —
the produced `Rule` is an ordinary numeric rule.

## Exceptions

`PickBucketsError` is the base. Subtypes: `NotFittedError`,
`InvalidBucketingError`, `MissingValueError`, `BoundaryError`,
`UnknownCategoryError`, `RuleSchemaError`.

## Benchmarks

`benchmarks/run_benchmarks.py` is a reproducible suite (fixed seed) covering
fit/transform across sizes, the runtime versus adapter paths, and
high-cardinality categorical grouping. It also reports peak heap memory
(`tracemalloc`) and includes a bounded-memory check for large streaming inputs.
Use `--json` to compare runs before and after a change. See
`benchmarks/README.md`.
