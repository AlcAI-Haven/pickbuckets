from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

BENCH_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "run_benchmarks.py"
)


def _load_benchmarks():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("pb_benchmarks", BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve annotations by module name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quick_run_is_reproducible() -> None:
    benchmarks = _load_benchmarks()
    first = benchmarks.run({"tiny": 500}, repeats=1)
    second = benchmarks.run({"tiny": 500}, repeats=1)
    assert [r.name for r in first] == [r.name for r in second]
    assert all(r.size == 500 for r in first)
    assert all(r.seconds >= 0 for r in first)


def test_format_table_smoke() -> None:
    benchmarks = _load_benchmarks()
    results = benchmarks.run({"tiny": 200}, repeats=1)
    table = benchmarks.format_table(results)
    assert "benchmark" in table
    assert "rows/s" in table


def test_main_quick_exits_zero(capsys) -> None:  # type: ignore[no-untyped-def]
    benchmarks = _load_benchmarks()
    code = benchmarks.main(["--quick", "--repeats", "1"])
    assert code == 0
    captured = capsys.readouterr()
    assert "equal_width" in captured.out
    assert "peak KB" in captured.out


def test_memory_measurement_can_be_toggled() -> None:
    benchmarks = _load_benchmarks()
    measured = benchmarks.run({"tiny": 500}, repeats=1, measure_memory=True)
    assert all(r.peak_bytes is not None and r.peak_bytes > 0 for r in measured)
    skipped = benchmarks.run({"tiny": 500}, repeats=1, measure_memory=False)
    assert all(r.peak_bytes is None for r in skipped)


def test_streaming_fit_memory_is_bounded_for_large_inputs() -> None:
    benchmarks = _load_benchmarks()
    small_data = benchmarks._numeric_data(20_000, random.Random(0))
    large_data = benchmarks._numeric_data(80_000, random.Random(1))

    small_peak = benchmarks._peak_memory(lambda: benchmarks._streaming_fit(small_data))
    large_peak = benchmarks._peak_memory(lambda: benchmarks._streaming_fit(large_data))

    # Input grew 4x; the bounded sketch must not grow memory proportionally.
    assert small_peak > 0
    assert large_peak < small_peak * 2


def test_eager_transform_memory_scales_with_output() -> None:
    from pickbuckets import EqualWidthBucket
    from pickbuckets.runtime import apply_rule

    benchmarks = _load_benchmarks()
    small_data = benchmarks._numeric_data(20_000, random.Random(0))
    large_data = benchmarks._numeric_data(80_000, random.Random(1))
    rule = EqualWidthBucket(n_bins=10).fit(small_data).rules_

    small_peak = benchmarks._peak_memory(lambda: apply_rule(rule, small_data))
    large_peak = benchmarks._peak_memory(lambda: apply_rule(rule, large_data))

    # A materializing transform allocates an output per row: roughly linear.
    assert large_peak > small_peak
