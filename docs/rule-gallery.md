# Rule Gallery

This page shows the kind of portable JSON rules `pickbuckets` is designed to
produce and apply. The examples stay focused on credit scoring, fraud
detection, and regulatory/compliance risk models, where bucket definitions are
usually part of the reviewable model contract.

## When To Use This

Use `pickbuckets` when the bucket definition itself needs to be durable,
inspectable, and portable across tools in focused risk-modeling workflows.

- Credit scoring: freeze score, income, utilization, or delinquency bands so
  model training and production scoring use the same reviewed cut points.
- Fraud detection: share amount, velocity, merchant category, device, or country
  buckets between feature generation, online decisions, and investigation jobs.
- Regulatory and compliance risk models: keep jurisdiction, exposure, capital,
  or review-tier rules in JSON so changes can be diffed, approved, and
  reproduced.

## JSON Rule Examples

Each example below is shaped like a rule a model-risk, fraud, or credit team
could inspect in code review.

### Credit Score Bands

```json
{
  "boundary_strategy": "underflow_overflow",
  "category_mapping": null,
  "closed": "left",
  "edges": [300, 580, 670, 740, 800, 850],
  "feature_name": "credit_score",
  "fit_stats": {
    "max": 850,
    "min": 300,
    "n_bins": 5,
    "source": "approved_training_sample_2026_q1"
  },
  "kind": "numeric",
  "labels": [
    "deep_subprime",
    "subprime",
    "near_prime",
    "prime",
    "super_prime"
  ],
  "missing_label": "manual_review",
  "missing_strategy": "separate",
  "overflow_label": "score_above_supported_range",
  "package_version": "0.3.0",
  "schema_version": "1.1",
  "underflow_label": "score_below_supported_range",
  "unknown_category_strategy": "other",
  "unknown_label": "Other"
}
```

### Fraud Transaction Amount Bands

```json
{
  "boundary_strategy": "error",
  "category_mapping": null,
  "closed": "left",
  "edges": [0, 25, 100, 500, 2500, "inf"],
  "feature_name": "transaction_amount_usd",
  "fit_stats": {
    "currency": "USD",
    "n_bins": 5,
    "training_window": "2026-01-01/2026-03-31"
  },
  "kind": "numeric",
  "labels": [
    "micro",
    "small",
    "medium",
    "large",
    "exceptional"
  ],
  "missing_label": "amount_missing",
  "missing_strategy": "separate",
  "overflow_label": "Overflow",
  "package_version": "0.3.0",
  "schema_version": "1.1",
  "underflow_label": "Underflow",
  "unknown_category_strategy": "other",
  "unknown_label": "Other"
}
```

### Regulatory Jurisdiction Tiers

```json
{
  "boundary_strategy": "clip",
  "category_mapping": {
    "CA": "standard_market",
    "DE": "standard_market",
    "FR": "standard_market",
    "GB": "standard_market",
    "HK": "enhanced_due_diligence",
    "IR": "restricted",
    "KP": "restricted",
    "SG": "standard_market",
    "US": "standard_market"
  },
  "closed": "left",
  "edges": null,
  "feature_name": "customer_jurisdiction",
  "fit_stats": {
    "policy": "sanctions_screening_2026_02",
    "reviewed_by": "model_risk_governance"
  },
  "kind": "categorical",
  "labels": [
    "standard_market",
    "enhanced_due_diligence",
    "restricted",
    "unmapped_jurisdiction"
  ],
  "missing_label": "unmapped_jurisdiction",
  "missing_strategy": "separate",
  "overflow_label": "Overflow",
  "package_version": "0.3.0",
  "schema_version": "1.1",
  "underflow_label": "Underflow",
  "unknown_category_strategy": "other",
  "unknown_label": "unmapped_jurisdiction"
}
```

These JSON payloads can be loaded with `Rule.from_json(...)` and applied with
`pickbuckets.runtime.apply_rule(...)`, including in services that do not import
the original training libraries.
