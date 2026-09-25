"""Tests de l'orchestration LLM → analyse POS → décision de blocage."""

from __future__ import annotations

import math

import httpx
import openai
import pytest

from detection_service import (
    LLMResponseError,
    LLMUnavailableError,
    detect_hallucination,
    extract_token_probabilities,
    is_answer_blocked,
    logprob_to_percent,
)
from tests.fakes import (
    CAPITAL_POS,
    FakeFrenchNlp,
    FakeOpenAIClient,
    capital_tokens,
    make_completion,
    make_token,
)

QUESTION = "Quelle est la capitale de la France ?"


def test_logprob_to_percent_converts_log_probability():
    assert logprob_to_percent(0.0) == pytest.approx(100.0)
    assert logprob_to_percent(math.log(0.5)) == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("weakest", "threshold", "expected"),
    [(69.9, 70, True), (70.0, 70, False), (95.0, 70, False), (0.0, 0, False)],
)
def test_is_answer_blocked_only_strictly_below_threshold(weakest, threshold, expected):
    assert is_answer_blocked(weakest, threshold) is expected


def test_extract_token_probabilities_keeps_alternatives():
    tokens = extract_token_probabilities([make_token(" Paris", 0.4, {" Paris": 0.4, " Lyon": 0.3})])

    assert len(tokens) == 1
    assert tokens[0].token == " Paris"
    assert tokens[0].probability == pytest.approx(40.0)
    assert [(alt.token, round(alt.probability, 6)) for alt in tokens[0].alternatives] == [
        (" Paris", 40.0),
        (" Lyon", 30.0),
    ]


def test_extract_token_probabilities_handles_missing_top_logprobs():
    tokens = extract_token_probabilities([make_token("La", 0.99, alternatives=None)])

    assert tokens[0].alternatives == ()


def test_detect_blocks_answer_when_critical_word_is_below_threshold():
    client = FakeOpenAIClient(make_completion(capital_tokens(paris_probability=0.40)))

    result = detect_hallucination(client, FakeFrenchNlp(CAPITAL_POS), "phi4-mini", QUESTION, threshold=70)

    assert result.is_blocked is True
    assert result.raw_answer == "La capitale est Paris"
    assert result.analysis.weakest_token == " Paris"
    assert result.analysis.weakest_probability == pytest.approx(40.0)
    # Seuls « capitale » (NOUN) et « Paris » (PROPN) sont critiques.
    assert [a.word for a in result.analysis.analyses] == ["capitale", "Paris"]
    assert len(result.tokens) == 4


def test_detect_lets_answer_through_when_confident():
    client = FakeOpenAIClient(make_completion(capital_tokens(paris_probability=0.97)))

    result = detect_hallucination(client, FakeFrenchNlp(CAPITAL_POS), "phi4-mini", QUESTION, threshold=70)

    assert result.is_blocked is False
    assert result.threshold == 70


def test_detect_ignores_non_critical_words_with_low_probability():
    tokens = capital_tokens(paris_probability=0.97)
    tokens[0] = make_token("La", 0.05)  # déterminant très incertain : sans importance
    client = FakeOpenAIClient(make_completion(tokens))

    result = detect_hallucination(client, FakeFrenchNlp(CAPITAL_POS), "phi4-mini", QUESTION, threshold=70)

    assert result.is_blocked is False


def test_detect_forwards_model_question_and_token_limit_to_llm():
    client = FakeOpenAIClient(make_completion(capital_tokens()))

    detect_hallucination(client, FakeFrenchNlp(CAPITAL_POS), "phi4-mini", QUESTION, 70, max_tokens=42)

    call = client.calls[0]
    assert call["model"] == "phi4-mini"
    assert call["messages"][-1] == {"role": "user", "content": QUESTION}
    assert call["logprobs"] is True
    assert call["max_tokens"] == 42


def test_detect_wraps_connection_errors_without_leaking_the_question():
    error = openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:11434/v1"))
    client = FakeOpenAIClient(error=error)

    with pytest.raises(LLMUnavailableError) as exc_info:
        detect_hallucination(client, FakeFrenchNlp(CAPITAL_POS), "phi4-mini", QUESTION, 70)

    assert str(exc_info.value) == "APIConnectionError"
    assert QUESTION not in str(exc_info.value)


def test_detect_rejects_response_without_logprobs():
    client = FakeOpenAIClient(make_completion(token_data=None, content="Paris"))

    with pytest.raises(LLMResponseError):
        detect_hallucination(client, FakeFrenchNlp(CAPITAL_POS), "phi4-mini", QUESTION, 70)


def test_detect_rejects_response_without_choices():
    client = FakeOpenAIClient(make_completion(capital_tokens()))
    client._response.choices = []

    with pytest.raises(LLMResponseError):
        detect_hallucination(client, FakeFrenchNlp(CAPITAL_POS), "phi4-mini", QUESTION, 70)
