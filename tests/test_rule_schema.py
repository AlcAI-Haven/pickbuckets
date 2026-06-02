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


def test_rule_rejects_nan_edges():
    with pytest.raises(RuleSchemaError):
        Rule(kind="numeric", edges=[0, float("nan")], labels=[0])


def test_rule_rejects_unsorted_edges():
    with pytest.raises(RuleSchemaError):
        Rule(kind="numeric", edges=[0, 10, 5], labels=[0, 1])
