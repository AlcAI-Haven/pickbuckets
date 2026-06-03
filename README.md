# pickbuckets

[![Python 3.9-3.13](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/pickbuckets/)
[![Rule schema 1.x](https://img.shields.io/badge/rule%20schema-1.x-blueviolet)](docs/api-reference.md)
[![pandas >=1.5](https://img.shields.io/badge/pandas-%3E%3D1.5-150458)](docs/integrations.md) [![Polars >=1.40](https://img.shields.io/badge/polars-%3E%3D1.40%20%28py3.10%2B%29-CD792C)](docs/integrations.md) [![scikit-learn >=1.2](https://img.shields.io/badge/scikit--learn-%3E%3D1.2-F7931E)](docs/integrations.md)

**Bucketing rules you can serialize, inspect, diff, and run anywhere.**

`pickbuckets` turns raw numerical and categorical values into human-readable,
versioned bucketing rules. Fit a rule with the full stack (pandas, Polars,
scikit-learn), serialize it to JSON, and apply it in a plain-Python service with
**no runtime dependencies**. The same rule produces identical results in
training, batch scoring, online inference, and monitoring.

## Portability in one example

Fit where the data lives; apply where the data doesn't.

```python
# --- training environment (full stack available) ---
from pickbuckets import EqualFrequencyBucket

bucket = EqualFrequencyBucket(n_bins=4, duplicates="drop").fit(training_values)
payload = bucket.to_json()          # store this string anywhere
```

```python
# --- serving environment (standard library only) ---
from pickbuckets import Rule
from pickbuckets.runtime import apply_rule

rule = Rule.from_json(payload)
codes = apply_rule(rule, [0, 5, 10, None])   # no pandas / Polars / sklearn
```

`apply_rule` is pure Python (optionally NumPy) and is verified by a CI job with
pandas, Polars, and scikit-learn uninstalled.

## Install

```bash
python -m pip install pickbuckets                 # dependency-free core
python -m pip install "pickbuckets[pandas]"
python -m pip install "pickbuckets[polars]"       # Python 3.10+
python -m pip install "pickbuckets[sklearn]"
python -m pip install "pickbuckets[plot]"
python -m pip install "pickbuckets[all]"
python -m pip install -e ".[dev,all]"             # development
```

## Bucketers

Every bucketer fits to a single unified [`Rule`](docs/api-reference.md) and shares
the same `fit` / `transform` / `summary` / `to_json` shape. `transform()` uses
only the saved rule — never the training data.

| Bucketer | Kind | What it does | Needs |
|---|---|---|---|
| `EqualWidthBucket` | numeric | Equal-width bins from min/max | core |
| `EqualFrequencyBucket` | numeric | Quantile (equal-frequency) bins | core |
| `CustomBoundaryBucket` | numeric | Manual, validated edges (supports `±inf`) | core |
| `RareCategoryBucket` | categorical | Fold rare/unseen categories to a fallback | core |
| `AutoBucket` | mixed | One rule per column, dtype-driven dispatch | core |
| `WoEBucket` | supervised | WoE/IV with monotonic + min-bin-size constraints | core |
| `ChiMergeBucket` | supervised | Chi-square adjacent-bin merging | core |
| `DecisionTreeBucket` | supervised | Edges from a shallow decision tree | `[sklearn]` |
| `ExternalSplitBucket` | numeric | Import external splits (e.g. OptBinning) | core |
| `StreamingEqualFrequencyBucket` | numeric, experimental | Online/approximate quantile bins for out-of-core data | core |

Plotting helpers (`pickbuckets.plotting`, behind `[plot]`) return matplotlib
`Axes` for bucket counts, target rate, and WoE.

```python
from pickbuckets import AutoBucket, EqualWidthBucket

frame = {"age": [18, 25, 34, 52, 70], "country": ["FR", "FR", "US", "DE", "DE"]}

auto = AutoBucket(
    n_bins=3,
    min_frequency=2,
    overrides={"age": EqualWidthBucket(n_bins=3, labels="interval")},
).fit(frame)
print(auto.transform(frame))
```

Configurable, serializable policies cover missing values
(`separate` / `most_frequent` / `propagate` / `error`), numeric boundaries
(`clip` / `underflow_overflow` / `error`), and unknown categories
(`other` / `missing` / `keep` / `error`), each raising a clear typed exception.

## Documentation

- [Usage guide](docs/usage-guide.md) — worked examples for every bucketer and policy
- [API reference](docs/api-reference.md) — symbols, the `Rule` model, runtime, exceptions
- [Integrations](docs/integrations.md) — pandas, Polars, and scikit-learn adapters
- [Rule gallery](docs/rule-gallery.md) — realistic portable JSON rules across ML use cases
- [Benchmarks](benchmarks/README.md) — reproducible speed and memory suite
- [Changelog](CHANGELOG.md)

## When to use it

Use `pickbuckets` when ML buckets must be stable, reviewable, and identical
across training, batch scoring, online inference, and monitoring — tabular
preprocessing, feature-store transforms, score banding, drift slices, and
governed models such as credit scoring, fraud, churn, pricing, and insurance
risk. See the [rule gallery](docs/rule-gallery.md) for examples.

## Development

```bash
ruff check .
mypy src/pickbuckets
pytest
python -m build
```

Contributions follow a few principles — keep the core import dependency-free,
put integrations behind extras, fit once and transform from saved rules. See
[CONTRIBUTING.md](CONTRIBUTING.md).
