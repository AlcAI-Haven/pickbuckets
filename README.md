# pickbuckets

Portable bucketing rules for Python.

`pickbuckets` turns raw numerical and categorical values into human-readable,
versioned rules that can be serialized, reviewed, diffed, and applied in a
plain-Python runtime.

```python
from pickbuckets import EqualWidthBucket

bucket = EqualWidthBucket(n_bins=3, labels="interval")
bucket.fit([1, 2, 3, 4, 5, 6])

print(bucket.transform([1, 2.5, 6]))
print(bucket.to_json())
```

## Why this exists

Many binning tools are powerful inside a training stack. `pickbuckets` focuses
on the part developers need when models and features leave notebooks:

- one unified rule model for numeric and categorical buckets
- readable JSON serialization
- dependency-free rule application for services and jobs
- optional integrations for pandas, Polars, scikit-learn, plotting, and YAML

## Install for development

```bash
python -m pip install -e ".[dev]"
pytest
```

The base package has no runtime dependencies.

## Current API

```python
from pickbuckets import (
    CustomBoundaryBucket,
    EqualFrequencyBucket,
    EqualWidthBucket,
    RareCategoryBucket,
)
```

### Equal-width numeric buckets

```python
bucket = EqualWidthBucket(n_bins=4, labels="ordinal")
bucket.fit([0, 10, 20, 30, 40])
bucket.transform([5, 15, 40])
```

### Equal-frequency numeric buckets

```python
bucket = EqualFrequencyBucket(n_bins=4, duplicates="drop")
bucket.fit([1, 1, 2, 3, 5, 8, 13, 21])
bucket.transform([1, 4, 21])
```

### Custom numeric boundaries

```python
bucket = CustomBoundaryBucket(
    edges=[0, 18, 35, 60, 100],
    labels=["child", "young_adult", "adult", "senior"],
    boundary_strategy="error",
)
bucket.fit([20, 40, 70])
bucket.transform([18, 35, 60])
```

### Rare category grouping

```python
bucket = RareCategoryBucket(min_frequency=2, other_label="Other")
bucket.fit(["a", "a", "b", "c"])
bucket.transform(["a", "b", "new"])
```

## Rule portability

Every fitted bucketer exposes a `rules_` object:

```python
rules = bucket.to_dict()
restored = EqualWidthBucket.from_dict(rules)
assert restored.transform([5, 15]) == bucket.transform([5, 15])
```

Rules include an explicit schema version and package version so exported
artifacts can be inspected safely before loading.

## Project status

This is an early project scaffold for the v0.1 core.
