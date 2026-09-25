"""Tests du client LLM (options transmises au SDK OpenAI)."""

from __future__ import annotations

from llm_client import SYSTEM_PROMPT, build_client, query_model
from tests.fakes import FakeOpenAIClient, capital_tokens, make_completion


def test_query_model_requests_logprobs_with_system_prompt():
    client = FakeOpenAIClient(make_completion(capital_tokens()))

    query_model(client, "phi4-mini", "Question ?")

    call = client.calls[0]
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert call["logprobs"] is True
    assert call["top_logprobs"] == 3


def test_query_model_omits_max_tokens_by_default():
    client = FakeOpenAIClient(make_completion(capital_tokens()))

    query_model(client, "phi4-mini", "Question ?")

    assert "max_tokens" not in client.calls[0]


def test_query_model_forwards_max_tokens_when_given():
    client = FakeOpenAIClient(make_completion(capital_tokens()))

    query_model(client, "phi4-mini", "Question ?", max_tokens=10)

    assert client.calls[0]["max_tokens"] == 10


def test_build_client_applies_timeout_and_retries():
    client = build_client(timeout=12.0, max_retries=0)

    assert client.timeout == 12.0
    assert client.max_retries == 0
