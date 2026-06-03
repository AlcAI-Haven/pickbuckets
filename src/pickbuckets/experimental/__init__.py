"""Experimental, not-yet-stable bucketers.

Everything in this namespace is opt-in and may change between minor releases.
Online and streaming approximations live here so they stay clearly separated
from the exact bucketers in :mod:`pickbuckets`.
"""

from pickbuckets.experimental.streaming import (
    StreamingEqualFrequencyBucket,
    StreamingHistogram,
)

__all__ = [
    "StreamingEqualFrequencyBucket",
    "StreamingHistogram",
]
