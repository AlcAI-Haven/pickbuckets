"""Optional input/output adapters around the dependency-free rule engine.

Nothing in this package is imported by ``import pickbuckets``. The helpers here
only treat an object as a pandas/Polars/NumPy container when that library is
*already imported* by the caller, so the core stays dependency-light and the
no-heavy-deps runtime gate keeps passing.
"""

from pickbuckets.adapters.detect import (
    Container,
    adapt_1d,
    detect_container,
)

__all__ = ["Container", "adapt_1d", "detect_container"]
