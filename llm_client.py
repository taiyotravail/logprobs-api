"""Client LLM : interroge Ollama via l'API compatible OpenAI."""

from __future__ import annotations

from openai import OpenAI

from config import OLLAMA_BASE_URL


def build_client() -> OpenAI:
    """Construit un client OpenAI pointant vers Ollama local."""
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def query_model(client: OpenAI, model: str, question: str):
    """Interroge le LLM en demandant les logprobs (top 3 alternatives par token)."""
    return client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant expert en analyse de données. Ta règle stricte : répondre à la question en une seule phrase affirmative et factuelle. Tu as l'interdiction de te justifier ou d'ajouter des avertissements."
                ),
            },
            {"role": "user", "content": question},
        ],
        logprobs=True,
        top_logprobs=3,
    )
