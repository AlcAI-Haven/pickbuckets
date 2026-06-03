# Benchmarks

Reproducible micro-benchmarks for `pickbuckets` fit, transform, and the
pure-Python runtime apply path.

## Running

```bash
# Full suite (small, medium, large)
python benchmarks/run_benchmarks.py

# Quick smoke run (small + medium only)
python benchmarks/run_benchmarks.py --quick

# Machine-readable output for before/after comparisons
python benchmarks/run_benchmarks.py --json > before.json
# ...make a change...
python benchmarks/run_benchmarks.py --json > after.json
```

## What is measured

- `equal_width.fit` / `equal_width.transform`
- `equal_frequency.fit`
- `runtime.apply_rule` — the dependency-free apply path
- `streaming.partial_fit[bounded_memory]` — the streaming sketch, included as a
  **memory check**: its bounded state keeps peak memory roughly flat as the
  input grows
- `rare_category.fit[high_cardinality]` — categorical grouping with many keys
- `pandas.transform` / `polars.transform` — adapter paths (skipped if the
  library is not installed, so the core path benchmarks with no heavy deps)

Each row reports both speed (`seconds`, `rows/s`) and memory (`peak KB`,
`B/row`).

## Memory usage checks

Peak heap memory is measured with the standard-library `tracemalloc` module — no
`psutil` or RSS sampling, so it is deterministic and cross-platform. Memory is
measured in a separate pass from timing (so it never pollutes the timings), and
only allocations *inside* the fit/transform are counted; input data allocated
beforehand is excluded.

This makes large-input memory behaviour visible and regression-testable:

- Materializing paths (`runtime.apply_rule`, adapter transforms) allocate an
  output per row, so `B/row` should stay roughly constant as size grows.
- `streaming.partial_fit` keeps a bounded sketch, so its **peak memory stays
  roughly flat** even as the input grows by orders of magnitude — verified by
  `tests/test_benchmarks.py`.

Pass `--no-memory` to time only and skip the memory pass.

## Reproducibility

Inputs are generated from a fixed seed (`SEED = 1234`), so two runs on the same
machine are directly comparable. Each benchmark is timed `--repeats` times
(default 3) and the **best** (minimum) wall-clock time is reported, which is the
most stable estimator for short CPU-bound work.

## Comparing before and after a change

`run()` returns a list of `BenchmarkResult` dataclasses, and `--json` emits the
same data, so a PR can diff `before.json` against `after.json`. Absolute numbers
depend on the machine; compare relative throughput (`rows/s`) on the same host.

## Performance tradeoffs

- The pure-Python `apply_rule` path trades raw speed for zero dependencies and
  portability. The pandas/Polars adapters are faster on large frames because
  they vectorize; the runtime path wins on startup cost and deployment size.
- `equal_frequency.fit` sorts its input, so it is `O(n log n)`; `equal_width.fit`
  only scans for min/max and is `O(n)`.
- High-cardinality categorical fitting is dominated by the frequency count, which
  is `O(n)` in the number of rows plus `O(k log k)` in the number of distinct
  categories.
