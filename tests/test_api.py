"""Tests de l'API REST (FastAPI), avec un faux détecteur injecté."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from api import ApiResources, create_app
from config import DEFAULT_THRESHOLD, MAX_QUESTION_LENGTH, SAFETY_MESSAGE
from detection_service import (
    DetectionResult,
    LLMResponseError,
    LLMUnavailableError,
    detect_hallucination,
)
from tests.fakes import CAPITAL_POS, FakeFrenchNlp, FakeOpenAIClient, capital_tokens, make_completion

ALLOWED_ORIGIN = "https://mon-site.fr"
QUESTION = "Quelle est la capitale de la France ?"


def fake_detector(paris_probability: float, calls: list[tuple[str, float]] | None = None) -> Callable[[str, float], DetectionResult]:
    """Détecteur réel branché sur un faux LLM et un faux spaCy."""
    client = FakeOpenAIClient(make_completion(capital_tokens(paris_probability)))
    nlp = FakeFrenchNlp(CAPITAL_POS)

    def detect(question: str, threshold: float) -> DetectionResult:
        if calls is not None:
            calls.append((question, threshold))
        return detect_hallucination(client, nlp, "fake-model", question, threshold)

    return detect


def failing_detector(error: Exception) -> Callable[[str, float], DetectionResult]:
    """Détecteur qui lève systématiquement `error`."""

    def detect(question: str, threshold: float) -> DetectionResult:
        raise error

    return detect


def make_client(
    detect: Callable[[str, float], DetectionResult],
    daily_limit: int = 0,
    expose_blocked_answers: bool = True,
) -> TestClient:
    """Construit l'app avec le détecteur donné (le `with` déclenche le lifespan)."""
    app = create_app(
        load_resources=lambda: ApiResources(detect=detect, model_name="fake-model"),
        allowed_origins=[ALLOWED_ORIGIN],
        daily_limit=daily_limit,
        expose_blocked_answers=expose_blocked_answers,
    )
    return TestClient(app)


def test_health_reports_model():
    with make_client(fake_detector(0.97)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "fake-model"}


def test_analyze_returns_answer_when_confident():
    with make_client(fake_detector(0.97)) as client:
        response = client.post("/analyze", json={"question": QUESTION, "threshold": 70})

    body = response.json()
    assert response.status_code == 200
    assert body["blocked"] is False
    assert body["answer"] == "La capitale est Paris"
    assert body["raw_answer"] == "La capitale est Paris"
    assert body["model"] == "fake-model"
    assert [t["token"] for t in body["tokens"]] == ["La", " capitale", " est", " Paris"]
    assert body["tokens"][3]["alternatives"][1] == {"token": " Lyon", "probability": pytest.approx(30.0)}


def test_analyze_replaces_unreliable_answer_with_safety_message():
    with make_client(fake_detector(0.40)) as client:
        response = client.post("/analyze", json={"question": QUESTION, "threshold": 70})

    body = response.json()
    assert response.status_code == 200
    assert body["blocked"] is True
    assert body["answer"] == SAFETY_MESSAGE
    assert body["raw_answer"] == "La capitale est Paris"
    assert body["weakest_token"] == " Paris"
    assert body["weakest_probability"] == pytest.approx(40.0)
    assert [c["pos"] for c in body["critical_tokens"]] == ["NOUN", "PROPN"]


def test_analyze_hides_blocked_answer_when_exposure_is_disabled():
    with make_client(fake_detector(0.40), expose_blocked_answers=False) as client:
        response = client.post("/analyze", json={"question": QUESTION})

    body = response.json()
    assert body["blocked"] is True
    assert body["answer"] == SAFETY_MESSAGE
    assert body["raw_answer"] is None
    assert body["tokens"] == [] and body["critical_tokens"] == [] and body["weakest_token"] == ""
    assert "Paris" not in response.text


def test_analyze_keeps_confident_answer_when_exposure_is_disabled():
    with make_client(fake_detector(0.97), expose_blocked_answers=False) as client:
        body = client.post("/analyze", json={"question": QUESTION}).json()

    assert body["raw_answer"] == "La capitale est Paris"
    assert len(body["tokens"]) == 4


def test_analyze_strips_question_and_uses_default_threshold():
    calls: list[tuple[str, float]] = []
    with make_client(fake_detector(0.97, calls)) as client:
        client.post("/analyze", json={"question": f"  {QUESTION}\n"})

    assert calls == [(QUESTION, DEFAULT_THRESHOLD)]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * (MAX_QUESTION_LENGTH + 1)},
        {"question": QUESTION, "threshold": 101},
        {"question": QUESTION, "threshold": -1},
        {"question": QUESTION, "model": "llama3:70b"},
        {},
    ],
)
def test_analyze_rejects_invalid_payloads(payload):
    calls: list[tuple[str, float]] = []
    with make_client(fake_detector(0.97, calls)) as client:
        response = client.post("/analyze", json=payload)

    assert response.status_code == 422
    assert calls == []


@pytest.mark.parametrize("raw_threshold", ["NaN", "Infinity", "-Infinity"])
def test_analyze_rejects_non_finite_threshold_without_crashing(raw_threshold):
    with make_client(fake_detector(0.97)) as client:
        response = client.post(
            "/analyze",
            content=f'{{"question": "q", "threshold": {raw_threshold}}}',
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422


def test_validation_errors_do_not_echo_the_question():
    too_long = "MARQUEUR-CONFIDENTIEL-7f3a " + "x" * MAX_QUESTION_LENGTH
    with make_client(fake_detector(0.97)) as client:
        response = client.post("/analyze", json={"question": too_long})

    assert response.status_code == 422
    assert "MARQUEUR-CONFIDENTIEL-7f3a" not in response.text
    assert response.json()["detail"][0]["loc"] == ["body", "question"]


def test_analyze_returns_503_when_llm_is_unavailable():
    with make_client(failing_detector(LLMUnavailableError("APIConnectionError"))) as client:
        response = client.post("/analyze", json={"question": QUESTION})

    assert response.status_code == 503


def test_analyze_returns_502_when_llm_response_is_unusable():
    with make_client(failing_detector(LLMResponseError("pas de logprobs"))) as client:
        response = client.post("/analyze", json={"question": QUESTION})

    assert response.status_code == 502


def test_analyze_enforces_daily_quota():
    with make_client(fake_detector(0.97), daily_limit=2) as client:
        statuses = [client.post("/analyze", json={"question": QUESTION}).status_code for _ in range(3)]

    assert statuses == [200, 200, 429]


def test_cors_allows_configured_origin_only():
    with make_client(fake_detector(0.97)) as client:
        allowed = client.options(
            "/analyze",
            headers={"Origin": ALLOWED_ORIGIN, "Access-Control-Request-Method": "POST"},
        )
        refused = client.options(
            "/analyze",
            headers={"Origin": "https://site-malveillant.example", "Access-Control-Request-Method": "POST"},
        )

    assert allowed.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
    assert "access-control-allow-origin" not in refused.headers


def test_logs_never_contain_question_or_answer(caplog):
    secret_question = "MARQUEUR-CONFIDENTIEL-7f3a : ma question contient des données sensibles ?"
    caplog.set_level(logging.DEBUG)
    with make_client(fake_detector(0.40)) as client:
        client.post("/analyze", json={"question": secret_question})
    with make_client(failing_detector(LLMUnavailableError("APIConnectionError"))) as client:
        client.post("/analyze", json={"question": secret_question})

    logged = "\n".join(record.getMessage() for record in caplog.records)
    # Garde-fou : les logs applicatifs sont bien capturés (test non vide).
    assert "Analyse en" in logged and "LLM indisponible" in logged
    assert "MARQUEUR-CONFIDENTIEL-7f3a" not in logged
    assert "La capitale est Paris" not in logged
