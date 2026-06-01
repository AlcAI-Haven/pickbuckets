import json

import pytest

from pickbuckets import Rule, RuleSchemaError


def test_rule_json_is_human_readable_and_sorted():
    rule = Rule(kind="numeric", edges=[0, 1], labels=[0])
    payload = rule.to_json()

    assert "\n" in payload
    assert json.loads(payload)["kind"] == "numeric"


def test_rule_rejects_unknown_major_schema_version():
    data = Rule(kind="numeric", edges=[0, 1], labels=[0]).to_dict()
    data["schema_version"] = "99.0"

    with pytest.raises(RuleSchemaError):
        Rule.from_dict(data)

