"""Orchestration de la détection d'hallucinations, indépendante de toute UI.

Enchaîne : appel LLM avec logprobs → analyse grammaticale (spaCy) → décision
de blocage. Aucune dépendance à FastAPI ni à Streamlit : ce module est
réutilisable depuis l'API REST, un batch ou des tests.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import openai
from spacy.language import Language

from confidence_analyzer import AnalysisResult, find_critical_tokens
from llm_client import query_model


class LLMUnavailableError(RuntimeError):
    """Le LLM n'a pas pu être joint (service arrêté, file pleine ou délai dépassé)."""


class LLMResponseError(RuntimeError):
    """Le LLM a répondu, mais sans les logprobs indispensables à l'analyse."""


@dataclass(frozen=True)
class AlternativeToken:
    """Une alternative envisagée par le modèle pour une position donnée."""

    token: str
    probability: float


@dataclass(frozen=True)
class TokenProbability:
    """Probabilité du token effectivement généré et de ses alternatives."""

    token: str
    probability: float
    alternatives: tuple[AlternativeToken, ...]


@dataclass(frozen=True)
class DetectionResult:
    """Résultat complet d'une détection, prêt à être sérialisé."""

    raw_answer: str
    is_blocked: bool
    threshold: float
    analysis: AnalysisResult
    tokens: tuple[TokenProbability, ...]


def logprob_to_percent(logprob: float) -> float:
    """Convertit une log-probabilité (base e) en pourcentage [0, 100]."""
    return math.exp(logprob) * 100


def is_answer_blocked(weakest_probability: float, threshold: float) -> bool:
    """Indique si la réponse doit être bloquée.

    La réponse est bloquée quand le maillon faible (probabilité la plus basse
    parmi les mots critiques) est strictement sous le seuil.
    """
    return weakest_probability < threshold


def extract_token_probabilities(token_data: Iterable[Any]) -> tuple[TokenProbability, ...]:
    """Convertit les logprobs brutes du SDK OpenAI en probabilités (%) sérialisables.

    Args:
        token_data: éléments `choices[0].logprobs.content` (attributs `token`,
            `logprob` et `top_logprobs`).
    """
    return tuple(
        TokenProbability(
            token=item.token,
            probability=logprob_to_percent(item.logprob),
            alternatives=tuple(
                AlternativeToken(token=alt.token, probability=logprob_to_percent(alt.logprob))
                for alt in (item.top_logprobs or ())
            ),
        )
        for item in token_data
    )


def detect_hallucination(
    client: openai.OpenAI,
    nlp: Language,
    model: str,
    question: str,
    threshold: float,
    max_tokens: int | None = None,
) -> DetectionResult:
    """Pose la question au LLM et décide si la réponse est assez fiable.

    Args:
        client: client OpenAI pointant vers Ollama.
        nlp: pipeline spaCy français (étiquetage grammatical).
        model: nom du modèle Ollama.
        question: question de l'utilisateur (déjà validée et nettoyée).
        threshold: seuil de confiance minimum, en pourcentage.
        max_tokens: limite de tokens générés (`None` = pas de limite).

    Raises:
        LLMUnavailableError: le LLM est injoignable, surchargé ou trop lent.
        LLMResponseError: la réponse ne contient pas de logprobs exploitables.
    """
    try:
        response = query_model(client, model, question, max_tokens=max_tokens)
    except openai.OpenAIError as exc:
        # Seul le type d'erreur est remonté : jamais la question (RGPD).
        raise LLMUnavailableError(type(exc).__name__) from exc

    if not response.choices:
        raise LLMResponseError("La réponse du LLM ne contient aucun choix.")
    choice = response.choices[0]
    if choice.logprobs is None or not choice.logprobs.content:
        raise LLMResponseError("La réponse du LLM ne contient pas de logprobs.")

    token_data = choice.logprobs.content
    analysis = find_critical_tokens(token_data, nlp)
    return DetectionResult(
        raw_answer=(choice.message.content or "").strip(),
        is_blocked=is_answer_blocked(analysis.weakest_probability, threshold),
        threshold=threshold,
        analysis=analysis,
        tokens=extract_token_probabilities(token_data),
    )
