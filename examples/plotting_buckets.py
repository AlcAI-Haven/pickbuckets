"""Plot bucket quality with the optional matplotlib helpers.

Requires the plotting extra::

    python -m pip install "pickbuckets[plot]"

Each helper returns a matplotlib Axes, so you can customise the figure and
decide when (or whether) to display or save it.
"""

from pickbuckets import WoEBucket
from pickbuckets.plotting import plot_bucket_counts, plot_target_rate, plot_woe


def main() -> None:
    import matplotlib.pyplot as plt

    scores = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    defaulted = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    bucket = WoEBucket(n_bins=4, output="woe").fit(scores, defaulted)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_bucket_counts(bucket, ax=axes[0])
    plot_target_rate(bucket, ax=axes[1])
    plot_woe(bucket, ax=axes[2])
    fig.tight_layout()

    fig.savefig("bucket_quality.png")
    print("Wrote bucket_quality.png")


if __name__ == "__main__":
    main()
