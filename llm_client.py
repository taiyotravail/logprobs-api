"""Client LLM : interroge Ollama via l'API compatible OpenAI."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from config import OLLAMA_BASE_URL

SYSTEM_PROMPT = (
    "Tu es un assistant expert en analyse de données. Ta règle stricte : répondre à la question en une seule phrase affirmative et factuelle. Tu as l'interdiction de te justifier ou d'ajouter des avertissements."
)


def build_client(timeout: float | None = None, max_retries: int | None = None) -> OpenAI:
    """Construit un client OpenAI pointant vers Ollama.

    Args:
        timeout: délai max (secondes) d'un appel. `None` garde le défaut du SDK.
        max_retries: nombre de nouvelles tentatives. `None` garde le défaut du SDK.
    """
    options: dict[str, Any] = {}
    if timeout is not None:
        options["timeout"] = timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", **options)


def query_model(client: OpenAI, model: str, question: str, max_tokens: int | None = None):
    """Interroge le LLM en demandant les logprobs (top 3 alternatives par token).

    Args:
        client: client OpenAI pointant vers Ollama.
        model: nom du modèle Ollama (ex. `phi4-mini:latest`).
        question: question posée par l'utilisateur.
        max_tokens: limite de tokens générés. `None` = pas de limite explicite.
    """
    options: dict[str, Any] = {}
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        logprobs=True,
        top_logprobs=3,
        **options,
    )
