import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agentlightning.reward import (
    find_reward_spans,
    get_last_reward,
    get_reward_value,
    is_reward_span,
)
from agentlightning.types.tracer import SpanNames


@dataclass
class FakeSpan:
    name: str
    attributes: Optional[Dict[str, Any]] = None


def test_get_reward_value_from_agentops_dict() -> None:
    span = FakeSpan(
        name="any",
        attributes={
            "agentops.task.output": {"type": "reward", "value": 3.5},
        },
    )

    assert get_reward_value(span) == 3.5


def test_get_reward_value_from_agentops_json_string() -> None:
    payload = json.dumps({"type": "reward", "value": 1.25})
    span = FakeSpan(name="any", attributes={"agentops.entity.output": payload})

    assert get_reward_value(span) == 1.25


def test_get_reward_value_from_reward_span_attributes() -> None:
    span = FakeSpan(
        name=SpanNames.REWARD.value,
        attributes={"reward": 0.75},
    )

    assert get_reward_value(span) == 0.75


def test_get_reward_value_returns_none_when_not_reward() -> None:
    span = FakeSpan(name="any", attributes={"agentops.task.output": {"foo": "bar"}})

    assert get_reward_value(span) is None


def test_is_reward_span_matches_reward_value() -> None:
    span = FakeSpan(
        name="whatever",
        attributes={"agentops.task.output": {"type": "reward", "value": 4.2}},
    )

    assert is_reward_span(span) is True


def test_is_reward_span_false_when_no_reward() -> None:
    span = FakeSpan(name="absent", attributes={"agentops.entity.output": {"value": 1}})

    assert is_reward_span(span) is False


def test_find_reward_spans_filters_correctly() -> None:
    reward_span = FakeSpan(
        name=SpanNames.REWARD.value,
        attributes={"reward": 2.0},
    )
    non_reward_span = FakeSpan(name="other", attributes={})

    spans = find_reward_spans([non_reward_span, reward_span, non_reward_span])

    assert spans == [reward_span]


def test_get_last_reward_returns_last_reward_value() -> None:
    spans = [
        FakeSpan(name="first", attributes={}),
        FakeSpan(name=SpanNames.REWARD.value, attributes={"reward": 1.0}),
        FakeSpan(name="agentops", attributes={"agentops.task.output": {"type": "reward", "value": 5.5}}),
    ]

    assert get_last_reward(spans) == 5.5


def test_get_last_reward_returns_none_when_no_reward() -> None:
    spans = [
        FakeSpan(name="first", attributes={}),
        FakeSpan(name="second", attributes={"agentops.task.output": {"foo": "bar"}}),
    ]

    assert get_last_reward(spans) is None
