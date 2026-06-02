"""Save a portable rule and apply it without the original bucketer."""

from pickbuckets import EqualWidthBucket, Rule
from pickbuckets.runtime import apply_rule


def main() -> None:
    bucket = EqualWidthBucket(n_bins=4, labels="interval").fit(
        [12, 18, 24, 36, 48, 72]
    )
    payload = bucket.to_json()

    restored_rule = Rule.from_json(payload)
    new_values = [12, 20, 50, None]

    print("Serialized rule:")
    print(payload)
    print()
    print("Applied from Rule only:")
    print(apply_rule(restored_rule, new_values))


if __name__ == "__main__":
    main()
