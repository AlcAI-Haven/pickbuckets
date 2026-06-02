# ML Rule Gallery

This page shows the kind of portable JSON rules `pickbuckets` is designed to
produce and apply. The examples are shaped like ML feature and scoring rules
that need to survive handoff from notebooks to training jobs, feature stores,
batch scoring, online inference, monitoring, and model governance.

## When To Use This

Quick recap: use `pickbuckets` when a bucketed ML feature or model score needs
to be stable, inspectable, serialized as JSON, and applied the same way outside
the training environment.

Common fits include tabular feature engineering, feature-store transforms,
model score banding, explainability slices, drift monitoring, fraud detection,
credit scoring, churn prediction, recommender ranking, pricing models,
healthcare or insurance risk, and regulated ML workflows.

## JSON Rule Examples

Each example below is valid rule JSON that `Rule.from_json(...)` can load.

### Propensity Score Bands

```json
{
  "boundary_strategy": "error",
  "category_mapping": null,
  "closed": "left",
  "edges": [0.0, 0.05, 0.2, 0.5, 0.8, 1.0],
  "feature_name": "churn_propensity_score",
  "fit_stats": {
    "calibration_window": "2026-01-01/2026-03-31",
    "model_name": "retention_xgb_v12",
    "n_bins": 5,
    "score_range": [0.0, 1.0]
  },
  "kind": "numeric",
  "labels": [
    "very_low_risk",
    "low_risk",
    "medium_risk",
    "high_risk",
    "critical_risk"
  ],
  "missing_label": "score_missing",
  "missing_strategy": "error",
  "overflow_label": "Overflow",
  "package_version": "0.3.0",
  "schema_version": "1.1",
  "underflow_label": "Underflow",
  "unknown_category_strategy": "other",
  "unknown_label": "Other"
}
```

### Feature Store Recency Buckets

```json
{
  "boundary_strategy": "underflow_overflow",
  "category_mapping": null,
  "closed": "left",
  "edges": [0, 1, 7, 30, 90, 365, "inf"],
  "feature_name": "days_since_last_session",
  "fit_stats": {
    "feature_store": "user_activity_daily",
    "n_bins": 6,
    "snapshot": "2026-03-31"
  },
  "kind": "numeric",
  "labels": [
    "same_day",
    "this_week",
    "this_month",
    "this_quarter",
    "this_year",
    "dormant"
  ],
  "missing_label": "no_session_history",
  "missing_strategy": "separate",
  "overflow_label": "stale_history",
  "package_version": "0.3.0",
  "schema_version": "1.1",
  "underflow_label": "invalid_negative_recency",
  "unknown_category_strategy": "other",
  "unknown_label": "Other"
}
```

### Categorical Channel Folding

```json
{
  "boundary_strategy": "clip",
  "category_mapping": {
    "affiliate": "partner",
    "direct": "owned",
    "email": "owned",
    "organic_search": "organic",
    "paid_search": "paid",
    "paid_social": "paid",
    "referral": "partner",
    "sms": "owned"
  },
  "closed": "left",
  "edges": null,
  "feature_name": "acquisition_channel",
  "fit_stats": {
    "min_frequency": 500,
    "training_rows": 2400000,
    "window": "2026-01-01/2026-03-31"
  },
  "kind": "categorical",
  "labels": [
    "owned",
    "paid",
    "organic",
    "partner",
    "rare_or_new"
  ],
  "missing_label": "rare_or_new",
  "missing_strategy": "separate",
  "overflow_label": "Overflow",
  "package_version": "0.3.0",
  "schema_version": "1.1",
  "underflow_label": "Underflow",
  "unknown_category_strategy": "other",
  "unknown_label": "rare_or_new"
}
```

### Fraud Transaction Velocity Bands

```json
{
  "boundary_strategy": "underflow_overflow",
  "category_mapping": null,
  "closed": "left",
  "edges": [0, 1, 3, 10, 25, 75, "inf"],
  "feature_name": "transactions_last_10_minutes",
  "fit_stats": {
    "aggregation": "rolling_10m",
    "feature_family": "velocity",
    "training_window": "2026-01-01/2026-03-31"
  },
  "kind": "numeric",
  "labels": [
    "none",
    "single",
    "low_velocity",
    "medium_velocity",
    "high_velocity",
    "extreme_velocity"
  ],
  "missing_label": "velocity_missing",
  "missing_strategy": "separate",
  "overflow_label": "velocity_overflow",
  "package_version": "0.3.0",
  "schema_version": "1.1",
  "underflow_label": "invalid_negative_velocity",
  "unknown_category_strategy": "other",
  "unknown_label": "Other"
}
```

These JSON payloads can be loaded with `Rule.from_json(...)` and applied with
`pickbuckets.runtime.apply_rule(...)`, including in services that do not import
the original training libraries.
