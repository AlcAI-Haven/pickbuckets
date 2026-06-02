import json

import pytest

from pickbuckets import Rule, RuleSchemaError


def test_rule_json_is_human_readable_and_sorted():
    rule = Rule(kind="numeric", edges=[0, 1], labels=[0])
    payload = rule.to_json()

    assert "\n" in payload
    assert json.loads(payload)["kind"] == "numeric"


def test_rule_json_encodes_infinite_edges_portably():
    rule = Rule(kind="numeric", edges=[float("-inf"), 0, float("inf")], labels=[0, 1])
    payload = rule.to_json()

    assert json.loads(payload)["edges"] == ["-inf", 0.0, "inf"]
    assert Rule.from_json(payload).edges == [float("-inf"), 0.0, float("inf")]


def test_rule_rejects_unknown_major_schema_version():
    data = Rule(kind="numeric", edges=[0, 1], labels=[0]).to_dict()
    data["schema_version"] = "99.0"

    with pytest.raises(RuleSchemaError):
        Rule.from_dict(data)


def test_old_rule_schema_loads_and_reexports_as_current_minor_version():
    data = Rule(kind="numeric", edges=[0, 1], labels=[0]).to_dict()
    data["schema_version"] = "1.0"
    data.pop("underflow_label")
    data.pop("overflow_label")

    restored = Rule.from_dict(data)

    assert restored.schema_version == Rule.CURRENT_SCHEMA_VERSION
    assert restored.to_dict()["schema_version"] == Rule.CURRENT_SCHEMA_VERSION
    assert restored.underflow_label == "Underflow"
    assert restored.overflow_label == "Overflow"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("closed", "right"),
        ("missing_strategy", "unknown"),
        ("boundary_strategy", "unknown"),
        ("unknown_category_strategy", "unknown"),
    ],
)
def test_rule_rejects_unknown_policy_values(field, value):
    data = Rule(kind="numeric", edges=[0, 1], labels=[0]).to_dict()
    data[field] = value

    with pytest.raises(RuleSchemaError):
        Rule.from_dict(data)


def test_rule_rejects_non_string_policy_values():
    with pytest.raises(RuleSchemaError):
        Rule(kind="numeric", edges=[0, 1], labels=[0], missing_strategy=["separate"])


def test_rule_rejects_nan_edges():
    with pytest.raises(RuleSchemaError):
        Rule(kind="numeric", edges=[0, float("nan")], labels=[0])


def test_rule_rejects_unsorted_edges():
    with pytest.raises(RuleSchemaError):
        Rule(kind="numeric", edges=[0, 10, 5], labels=[0, 1])


def test_categorical_rule_normalizes_mapping_keys_to_strings():
    rule = Rule(
        kind="categorical",
        category_mapping={1: "one"},
        labels=["one", "Other"],
    )

    assert rule.category_mapping == {"1": "one"}


def test_categorical_rule_rejects_mapping_labels_outside_declared_labels():
    with pytest.raises(RuleSchemaError, match="not present in labels"):
        Rule(
            kind="categorical",
            category_mapping={"a": "A"},
            labels=["Other"],
        )


def test_categorical_rule_rejects_unknown_other_label_outside_labels():
    with pytest.raises(RuleSchemaError, match="unknown_label"):
        Rule(
            kind="categorical",
            category_mapping={"a": "A"},
            labels=["A"],
            unknown_label="Other",
        )


def test_categorical_rule_rejects_empty_labels():
    with pytest.raises(RuleSchemaError, match="at least one label"):
        Rule(kind="categorical", category_mapping={}, labels=[])
