class PickBucketsError(Exception):
    """Base exception for pickbuckets."""


class NotFittedError(PickBucketsError):
    """Raised when transform is called before fit."""


class RuleSchemaError(PickBucketsError):
    """Raised when a rule payload cannot be loaded safely."""


class BoundaryError(PickBucketsError):
    """Raised when a value falls outside allowed numeric boundaries."""


class UnknownCategoryError(PickBucketsError):
    """Raised when a category is not present in a categorical rule."""


class InvalidBucketingError(PickBucketsError, ValueError):
    """Raised when inputs or parameters cannot produce valid buckets."""

