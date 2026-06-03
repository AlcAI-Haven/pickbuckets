# Changelog

All notable changes to `pickbuckets` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-06-03

Phase 5 — production and performance.

### Added

- Experimental streaming bucketing under `pickbuckets.experimental`:
  `StreamingEqualFrequencyBucket` with chunked `.partial_fit()` and a bounded
  `StreamingHistogram` quantile sketch. Fitting is approximate; the produced
  rule applies through the same pure-Python runtime as the exact bucketers.
  Approximation error is bounded by `max_centroids` and converges to exact
  edges as it grows.
- Optional matplotlib plotting helpers under `pickbuckets.plotting`
  (`plot_bucket_counts`, `plot_target_rate`, `plot_woe`), behind the
  `pickbuckets[plot]` extra. Each helper returns a matplotlib `Axes` for further
  customisation and never forces display.
- Reproducible benchmark suite in `benchmarks/run_benchmarks.py` covering
  fit/transform across sizes, the runtime apply path versus adapter paths, and
  high-cardinality categorical grouping, with `--json` output for before/after
  comparisons. Each benchmark also reports peak heap memory via `tracemalloc`,
  including a bounded-memory check for large streaming inputs (`--no-memory`
  skips the memory pass).
- API reference documentation in `docs/api-reference.md`, leading with the
  export → apply-without-the-stack portability demo.

### Changed

- `mypy` now recognises matplotlib as an optional, stub-free dependency.

### Notes

- The core package remains dependency-free; streaming is pure-Python and the
  no-heavy-deps CI gate is unchanged.

## [0.4.0] - 2026-06

Phase 4 — advanced and supervised binning.

### Added

- `DecisionTreeBucket`, `WoEBucket` (with WoE/IV, smoothing, minimum-bin-size,
  and monotonic constraints), `ChiMergeBucket`, and `ExternalSplitBucket`.
- Supervised summaries (`summary_table()`, `iv_summary()`) that expose target
  rates and counts without leaking raw targets into serialized rules.

## [0.3.0] - 2026

Phase 3 — real-world robustness.

### Added

- Missing-value strategies (`separate`, `most_frequent`, `propagate`, `error`).
- Numeric boundary strategies (`clip`, `underflow_overflow`, `error`).
- Unseen-category strategies (`other`, `missing`, `keep`, `error`).
- Typed exceptions and `.summary()` rule inspection.

## [0.2.0] - 2026

Phase 2 — developer experience and integrations.

### Added

- Public bucketer classes and scikit-learn-compatible adapters.
- pandas `Series`/`DataFrame` support and first-class Polars
  `Series`/`DataFrame`/`LazyFrame` support.
- `AutoBucket` smart type dispatch.

## [0.1.0] - 2026

Phase 1 — core MVP and portable rule foundation.

### Added

- Unified, versioned `Rule` model with dict/JSON serialization.
- Pure-Python rule runtime and a no-heavy-deps CI gate.
- Equal-width, equal-frequency, custom-boundary, and rare-category bucketers.

[0.5.0]: https://github.com/AlcAI-Haven/pickbuckets/releases/tag/v0.5.0
[0.4.0]: https://github.com/AlcAI-Haven/pickbuckets/releases/tag/v0.4.0
[0.3.0]: https://github.com/AlcAI-Haven/pickbuckets/releases/tag/v0.3.0
[0.2.0]: https://github.com/AlcAI-Haven/pickbuckets/releases/tag/v0.2.0
[0.1.0]: https://github.com/AlcAI-Haven/pickbuckets/releases/tag/v0.1.0
