"""Optional matplotlib plotting helpers for fitted bucketers.

These helpers are behind the ``pickbuckets[plot]`` extra and are never imported
by the core package. Every function returns a matplotlib ``Axes`` so callers can
customise the figure afterwards (titles, colours, saving) and decide when, or
whether, to display it.
"""

from __future__ import annotations

from typing import Any

from pickbuckets.exceptions import InvalidBucketingError, NotFittedError
from pickbuckets.rules import Rule

__all__ = [
    "plot_bucket_counts",
    "plot_target_rate",
    "plot_woe",
]


def plot_bucket_counts(obj: Any, *, ax: Any = None, **bar_kwargs: Any) -> Any:
    """Plot the per-bucket observation count from a fitted rule.

    ``obj`` may be a fitted bucketer or a :class:`~pickbuckets.Rule`. Works for
    numeric (``bin_counts``) and categorical (``output_counts``) rules.
    """

    rule = _resolve_rule(obj)
    counts = rule.fit_stats.get("bin_counts") or rule.fit_stats.get("output_counts")
    if not counts:
        raise InvalidBucketingError(
            "This rule has no stored bucket counts to plot. Fit on data that "
            "records bin_counts/output_counts first."
        )
    labels = [str(row["label"]) for row in counts]
    values = [row["count"] for row in counts]

    ax = _new_axes(ax)
    ax.bar(labels, values, **bar_kwargs)
    ax.set_xlabel(rule.feature_name or "bucket")
    ax.set_ylabel("count")
    ax.set_title(_title("Bucket counts", rule))
    _rotate_labels(ax)
    return ax


def plot_target_rate(obj: Any, *, ax: Any = None, **plot_kwargs: Any) -> Any:
    """Plot the per-bucket target (event) rate for a supervised rule."""

    rule = _resolve_rule(obj)
    summary = _supervised_summary(rule)
    labels = [str(row["label"]) for row in summary]
    rates = [row["target_rate"] for row in summary]

    ax = _new_axes(ax)
    marker = plot_kwargs.pop("marker", "o")
    ax.plot(labels, rates, marker=marker, **plot_kwargs)
    ax.set_xlabel(rule.feature_name or "bucket")
    ax.set_ylabel("target rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(_title("Target rate per bucket", rule))
    _rotate_labels(ax)
    return ax


def plot_woe(obj: Any, *, ax: Any = None, **bar_kwargs: Any) -> Any:
    """Plot the per-bucket Weight of Evidence for a supervised rule."""

    rule = _resolve_rule(obj)
    summary = _supervised_summary(rule)
    labels = [str(row["label"]) for row in summary]
    woe = [row["woe"] for row in summary]

    ax = _new_axes(ax)
    ax.bar(labels, woe, **bar_kwargs)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(rule.feature_name or "bucket")
    ax.set_ylabel("WoE")
    iv = rule.fit_stats.get("iv")
    title = "Weight of Evidence per bucket"
    if isinstance(iv, (int, float)):
        title = f"{title} (IV={iv:.4f})"
    ax.set_title(_title(title, rule))
    _rotate_labels(ax)
    return ax


def _resolve_rule(obj: Any) -> Rule:
    if isinstance(obj, Rule):
        return obj
    rule = getattr(obj, "rules_", None)
    if isinstance(rule, Rule):
        return rule
    raise NotFittedError(
        "Pass a fitted bucketer or a Rule; this object exposes no fitted rule."
    )


def _supervised_summary(rule: Rule) -> list[dict[str, Any]]:
    summary = rule.fit_stats.get("bin_summary")
    if not summary:
        raise InvalidBucketingError(
            "This rule has no supervised bin_summary; target-rate and WoE plots "
            "require a supervised bucketer (WoE, ChiMerge, or decision tree)."
        )
    return [dict(row) for row in summary]


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise InvalidBucketingError(
            "Plotting requires matplotlib; install pickbuckets[plot]."
        ) from exc
    return plt


def _new_axes(ax: Any) -> Any:
    if ax is not None:
        return ax
    plt = _require_matplotlib()
    _, ax = plt.subplots()
    return ax


def _rotate_labels(ax: Any) -> None:
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")


def _title(base: str, rule: Rule) -> str:
    if rule.feature_name:
        return f"{base}: {rule.feature_name}"
    return base
