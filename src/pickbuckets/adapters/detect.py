"""Lazy framework detection and 1-D input/output adaptation.

The functions here never ``import pandas`` or ``import polars`` themselves.
They look the modules up in :data:`sys.modules`; if a user handed us a
``pandas.Series`` the pandas module is necessarily already imported, so this is
sufficient to recognise the type without making the core depend on it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any


class Container(Enum):
    """Recognised 1-D input container kinds."""

    SEQUENCE = "sequence"
    NUMPY = "numpy"
    PANDAS_SERIES = "pandas_series"
    POLARS_SERIES = "polars_series"


def _module(name: str) -> Any | None:
    return sys.modules.get(name)


def _is_numpy(obj: Any) -> bool:
    np = _module("numpy")
    return np is not None and isinstance(obj, np.ndarray)


def _is_pandas_series(obj: Any) -> bool:
    pd = _module("pandas")
    return pd is not None and isinstance(obj, pd.Series)


def _is_polars_series(obj: Any) -> bool:
    pl = _module("polars")
    return pl is not None and isinstance(obj, pl.Series)


def detect_container(obj: Any) -> Container:
    """Classify a 1-D input object without importing optional libraries."""

    if _is_pandas_series(obj):
        return Container.PANDAS_SERIES
    if _is_polars_series(obj):
        return Container.POLARS_SERIES
    if _is_numpy(obj):
        return Container.NUMPY
    return Container.SEQUENCE


def adapt_1d(
    values: Iterable[Any],
) -> tuple[list[Any], Callable[[list[Any]], Any]]:
    """Return ``(plain_values, rebuild)`` for a 1-D input.

    ``plain_values`` is a list suitable for the pure-Python runtime. ``rebuild``
    wraps a list of transformed values back into the original container type,
    preserving a pandas index/name where one was provided. Polars Series are
    rebuilt via the vectorized path (see :mod:`pickbuckets.adapters.polars_io`).
    """

    kind = detect_container(values)

    if kind is Container.PANDAS_SERIES:
        pd = _module("pandas")
        assert pd is not None
        series: Any = values
        index = series.index
        name = series.name

        def rebuild_pandas(result: list[Any]) -> Any:
            return pd.Series(result, index=index, name=name)

        return list(values), rebuild_pandas

    if kind is Container.POLARS_SERIES:
        series_pl: Any = values
        name = series_pl.name

        def rebuild_polars(result: list[Any]) -> Any:
            pl = _module("polars")
            assert pl is not None
            return pl.Series(name, result)

        return list(values), rebuild_polars

    # NumPy arrays and plain sequences both round-trip to a list.
    return list(values), lambda result: result
