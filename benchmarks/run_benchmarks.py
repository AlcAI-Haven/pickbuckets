"""Reproducible micro-benchmarks for pickbuckets.

Run the full suite::

    python benchmarks/run_benchmarks.py

Quick smoke run (small sizes, used by CI and tests)::

    python benchmarks/run_benchmarks.py --quick

The script is deterministic: inputs are generated from a fixed seed so two runs
on the same machine produce comparable numbers. Optional adapter benchmarks
(pandas, Polars) are skipped automatically when the library is not installed, so
the core path can be benchmarked with no heavy dependencies. Results are printed
as a table and also returned by :func:`run` for programmatic comparison.

Each benchmark also reports peak heap memory (via :mod:`tracemalloc`) so large
inputs can be checked for memory regressions. The ``streaming.partial_fit``
benchmark exists specifically as a memory check: its bounded sketch keeps peak
memory roughly flat as the input size grows. Pass ``--no-memory`` to time only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from pickbuckets import EqualFrequencyBucket, EqualWidthBucket, RareCategoryBucket
from pickbuckets.experimental import StreamingEqualFrequencyBucket
from pickbuckets.runtime import apply_rule

SEED = 1234
SIZES: dict[str, int] = {"small": 1_000, "medium": 100_000, "large": 1_000_000}
QUICK_SIZES: dict[str, int] = {"small": 1_000, "medium": 10_000}
STREAM_CHUNK = 10_000


@dataclass
class BenchmarkResult:
    name: str
    size: int
    seconds: float
    throughput_per_s: float
    peak_bytes: int | None = None

    @property
    def bytes_per_row(self) -> float | None:
        if self.peak_bytes is None or self.size == 0:
            return None
        return self.peak_bytes / self.size


def _time(fn: Callable[[], Any], *, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = perf_counter()
        fn()
        best = min(best, perf_counter() - start)
    return best


def _peak_memory(fn: Callable[[], Any]) -> int:
    """Peak Python heap allocated *inside* ``fn``, in bytes.

    Uses :mod:`tracemalloc` (stdlib, deterministic, cross-platform) so no
    ``psutil``/RSS dependency is needed. Input data allocated before the call is
    not counted, so this isolates the memory a fit/transform itself requires.
    """

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()
    return peak


def _bench(
    name: str,
    size: int,
    fn: Callable[[], Any],
    *,
    repeats: int,
    measure_memory: bool,
) -> BenchmarkResult:
    seconds = _time(fn, repeats=repeats)
    throughput = size / seconds if seconds > 0 else float("inf")
    peak = _peak_memory(fn) if measure_memory else None
    return BenchmarkResult(
        name=name,
        size=size,
        seconds=seconds,
        throughput_per_s=throughput,
        peak_bytes=peak,
    )


def _numeric_data(size: int, rng: random.Random) -> list[float]:
    return [rng.gauss(0.0, 1.0) for _ in range(size)]


def _categorical_data(size: int, rng: random.Random, cardinality: int) -> list[str]:
    # Zipf-ish skew so rare-category folding has something to do.
    weights = [1.0 / (rank + 1) for rank in range(cardinality)]
    categories = [f"cat_{index}" for index in range(cardinality)]
    return rng.choices(categories, weights=weights, k=size)


def _streaming_fit(data: Sequence[float], chunk: int = STREAM_CHUNK) -> None:
    bucket = StreamingEqualFrequencyBucket(n_bins=10, max_centroids=256)
    for start in range(0, len(data), chunk):
        bucket.partial_fit(data[start : start + chunk])
    bucket.finalize()


def run(
    sizes: dict[str, int],
    *,
    repeats: int = 3,
    measure_memory: bool = True,
) -> list[BenchmarkResult]:
    rng = random.Random(SEED)
    results: list[BenchmarkResult] = []

    for label, size in sizes.items():
        numeric = _numeric_data(size, rng)

        results.append(
            _bench(
                f"equal_width.fit[{label}]",
                size,
                lambda d=numeric: EqualWidthBucket(n_bins=10).fit(d),
                repeats=repeats,
                measure_memory=measure_memory,
            )
        )
        fitted_width = EqualWidthBucket(n_bins=10).fit(numeric)
        results.append(
            _bench(
                f"equal_width.transform[{label}]",
                size,
                lambda b=fitted_width, d=numeric: b.transform(d),
                repeats=repeats,
                measure_memory=measure_memory,
            )
        )

        results.append(
            _bench(
                f"equal_frequency.fit[{label}]",
                size,
                lambda d=numeric: EqualFrequencyBucket(
                    n_bins=10, duplicates="drop"
                ).fit(d),
                repeats=repeats,
                measure_memory=measure_memory,
            )
        )

        fitted = EqualFrequencyBucket(n_bins=10, duplicates="drop").fit(numeric)
        results.append(
            _bench(
                f"runtime.apply_rule[{label}]",
                size,
                lambda r=fitted.rules_, d=numeric: apply_rule(r, d),
                repeats=repeats,
                measure_memory=measure_memory,
            )
        )

        # Streaming fit holds a bounded sketch, so its peak memory stays roughly
        # flat as `size` grows — the memory check for large inputs.
        results.append(
            _bench(
                f"streaming.partial_fit[bounded_memory][{label}]",
                size,
                lambda d=numeric: _streaming_fit(d),
                repeats=repeats,
                measure_memory=measure_memory,
            )
        )

        categorical = _categorical_data(size, rng, cardinality=max(50, size // 100))
        results.append(
            _bench(
                f"rare_category.fit[high_cardinality][{label}]",
                size,
                lambda d=categorical: RareCategoryBucket(min_frequency=5).fit(d),
                repeats=repeats,
                measure_memory=measure_memory,
            )
        )

        results.extend(
            _adapter_results(label, size, numeric, repeats, measure_memory)
        )

    return results


def _adapter_results(
    label: str,
    size: int,
    numeric: Sequence[float],
    repeats: int,
    measure_memory: bool,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    fitted = EqualWidthBucket(n_bins=10).fit(list(numeric))

    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        series = pd.Series(numeric)
        results.append(
            _bench(
                f"pandas.transform[{label}]",
                size,
                lambda b=fitted, s=series: b.transform(s),
                repeats=repeats,
                measure_memory=measure_memory,
            )
        )

    try:
        import polars as pl
    except ImportError:
        pl = None
    if pl is not None:
        ps = pl.Series(list(numeric))
        results.append(
            _bench(
                f"polars.transform[{label}]",
                size,
                lambda b=fitted, s=ps: b.transform(s),
                repeats=repeats,
                measure_memory=measure_memory,
            )
        )

    return results


def _format_peak(item: BenchmarkResult) -> tuple[str, str]:
    if item.peak_bytes is None:
        return "-", "-"
    per_row = item.bytes_per_row
    per_row_text = "-" if per_row is None else f"{per_row:.1f}"
    return f"{item.peak_bytes / 1024:,.1f}", per_row_text


def format_table(results: Sequence[BenchmarkResult]) -> str:
    header = (
        f"{'benchmark':<46}{'size':>12}{'seconds':>14}{'rows/s':>16}"
        f"{'peak KB':>14}{'B/row':>10}"
    )
    lines = [header, "-" * len(header)]
    for item in results:
        peak_kb, per_row = _format_peak(item)
        lines.append(
            f"{item.name:<46}{item.size:>12,}{item.seconds:>14.6f}"
            f"{item.throughput_per_s:>16,.0f}{peak_kb:>14}{per_row:>10}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pickbuckets benchmarks")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only small sizes (smoke test).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON instead of a table.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repeats; the best (minimum) is reported.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Skip peak-memory measurement (timing only).",
    )
    args = parser.parse_args(argv)

    sizes = QUICK_SIZES if args.quick else SIZES
    results = run(
        sizes, repeats=args.repeats, measure_memory=not args.no_memory
    )

    if args.json:
        print(json.dumps([asdict(item) for item in results], indent=2))
    else:
        print(format_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
