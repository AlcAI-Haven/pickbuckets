"""Tabular adapters for AutoBucket.

Supports pandas ``DataFrame``, Polars ``DataFrame``/``LazyFrame`` and a
dependency-free ``dict[str, list]`` "frame". Detection is lazy: a library is
only consulted if it is already imported.
"""

from __future__ import annotations

import sys
from collections.abc import Hashable, Mapping
from enum import Enum
from typing import Any

from pickbuckets.exceptions import InvalidBucketingError


class FrameKind(Enum):
    DICT = "dict"
    PANDAS = "pandas"
    POLARS = "polars"
    POLARS_LAZY = "polars_lazy"


def _module(name: str) -> Any | None:
    return sys.modules.get(name)


def detect_frame(obj: Any) -> FrameKind | None:
    pd = _module("pandas")
    if pd is not None and isinstance(obj, pd.DataFrame):
        return FrameKind.PANDAS
    pl = _module("polars")
    if pl is not None:
        if isinstance(obj, pl.DataFrame):
            return FrameKind.POLARS
        if isinstance(obj, pl.LazyFrame):
            return FrameKind.POLARS_LAZY
    if isinstance(obj, Mapping):
        return FrameKind.DICT
    return None


def column_names(obj: Any, kind: FrameKind) -> list[Hashable]:
    if kind is FrameKind.PANDAS:
        return list(obj.columns)
    if kind is FrameKind.POLARS:
        return list(obj.columns)
    if kind is FrameKind.POLARS_LAZY:
        return list(obj.collect_schema().names())
    return list(obj.keys())


def _dtype_kind_from_values(values: list[Any]) -> str:
    from pickbuckets.runtime.apply import is_missing

    seen = [value for value in values if not is_missing(value)]
    if not seen:
        return "categorical"
    if all(isinstance(value, bool) for value in seen):
        return "categorical"
    if all(isinstance(value, (int, float)) for value in seen):
        return "numeric"
    return "categorical"


def column_kind(obj: Any, kind: FrameKind, name: Hashable) -> str:
    """Return 'numeric', 'categorical' or 'unsupported' for a column."""

    if kind is FrameKind.PANDAS:
        import pandas as pd

        dtype = obj[name].dtype
        if pd.api.types.is_bool_dtype(dtype):
            return "categorical"
        if pd.api.types.is_numeric_dtype(dtype):
            return "numeric"
        if (
            pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
        ):
            return "categorical"
        return "unsupported"

    if kind in (FrameKind.POLARS, FrameKind.POLARS_LAZY):
        import polars as pl

        if kind is FrameKind.POLARS:
            dtype = obj.schema[name]
        else:
            dtype = obj.collect_schema()[name]
        if dtype == pl.Boolean:
            return "categorical"
        if dtype.is_numeric():
            return "numeric"
        if dtype in (pl.Utf8, pl.Categorical) or dtype == pl.String:
            return "categorical"
        return "unsupported"

    return _dtype_kind_from_values(list(obj[name]))


def column_values(obj: Any, kind: FrameKind, name: Hashable) -> list[Any]:
    """Materialize one column as a plain Python list (for fitting)."""

    if kind is FrameKind.PANDAS:
        return list(obj[name].tolist())
    if kind is FrameKind.POLARS:
        return list(obj.get_column(name).to_list())
    if kind is FrameKind.POLARS_LAZY:
        return list(obj.select(name).collect().get_column(name).to_list())
    return list(obj[name])


def pandas_index(obj: Any) -> Any:
    return obj.index


def build_dataframe(
    kind: FrameKind,
    order: list[Hashable],
    transformed: dict[Hashable, list[Any]],
    *,
    index: Any = None,
) -> Any:
    """Rebuild a frame from per-column transformed lists (pandas/dict paths)."""

    if kind is FrameKind.PANDAS:
        import pandas as pd

        return pd.DataFrame({name: transformed[name] for name in order}, index=index)
    return {name: transformed[name] for name in order}


def require_frame(obj: Any) -> FrameKind:
    kind = detect_frame(obj)
    if kind is None:
        raise InvalidBucketingError(
            "AutoBucket expects a pandas/Polars DataFrame, a Polars LazyFrame, "
            "or a mapping of column name to values."
        )
    if kind is FrameKind.DICT:
        _validate_mapping_frame(obj)
    return kind


def _validate_mapping_frame(obj: Mapping[Any, Any]) -> None:
    lengths: list[int] = []
    for name, values in obj.items():
        if isinstance(values, (str, bytes)):
            raise InvalidBucketingError(
                f"Column {name!r} must be a sequence of values, not a string."
            )
        try:
            lengths.append(len(values))
        except TypeError as exc:
            raise InvalidBucketingError(
                f"Column {name!r} must be a sized sequence of values."
            ) from exc
    if len(set(lengths)) > 1:
        raise InvalidBucketingError(
            "All columns in a mapping frame must have the same length."
        )
