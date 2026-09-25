"""Doublures de test : faux LLM (réponses OpenAI) et faux pipeline spaCy.

Permettent de tester toute la chaîne sans Ollama ni modèle `fr_core_news_md`.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from spacy.tokens import Doc
from spacy.vocab import Vocab


def make_token(token: str, probability: float, alternatives: dict[str, float] | None = None) -> SimpleNamespace:
    """Construit un élément `logprobs.content` au format du SDK OpenAI.

    Args:
        token: texte du token (espace initiale incluse, comme chez le LLM).
        probability: probabilité du token, entre 0 et 1.
        alternatives: alternatives {token: probabilité}. `None` = pas de top_logprobs.
    """
    top_logprobs = (
        None
        if alternatives is None
        else [SimpleNamespace(token=alt, logprob=math.log(p)) for alt, p in alternatives.items()]
    )
    return SimpleNamespace(token=token, logprob=math.log(probability), top_logprobs=top_logprobs)


def make_completion(token_data: list[SimpleNamespace] | None, content: str | None = None) -> SimpleNamespace:
    """Construit une réponse `chat.completions.create` minimale.

    Args:
        token_data: tokens avec logprobs. `None` = réponse sans logprobs.
        content: texte de la réponse. Par défaut, concaténation des tokens.
    """
    if content is None and token_data is not None:
        content = "".join(item.token for item in token_data)
    logprobs = None if token_data is None else SimpleNamespace(content=token_data)
    choice = SimpleNamespace(message=SimpleNamespace(content=content), logprobs=logprobs)
    return SimpleNamespace(choices=[choice])


class FakeOpenAIClient:
    """Faux client OpenAI : renvoie une réponse fixe ou lève une erreur, et mémorise l'appel."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response = response
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class FakeFrenchNlp:
    """Faux pipeline spaCy : découpe sur les espaces simples, POS via un dictionnaire.

    Le texte analysé ne doit contenir ni espace initiale ni espaces multiples
    (sinon le `Doc` reconstruit ne correspondrait plus au texte d'origine).
    """

    def __init__(self, pos_by_word: dict[str, str]) -> None:
        self._pos_by_word = pos_by_word
        self._vocab = Vocab()

    def __call__(self, text: str) -> Doc:
        words = text.split(" ")
        spaces = [True] * (len(words) - 1) + [False]
        pos = [self._pos_by_word.get(word, "X") for word in words]
        doc = Doc(self._vocab, words=words, spaces=spaces, pos=pos)
        assert doc.text == text, "FakeFrenchNlp : texte non supporté"
        return doc


# Réponse type : « La capitale est Paris », où « Paris » est peu probable (40 %).
CAPITAL_POS = {"La": "DET", "capitale": "NOUN", "est": "AUX", "Paris": "PROPN"}


def capital_tokens(paris_probability: float = 0.40) -> list[SimpleNamespace]:
    """Tokens de « La capitale est Paris » avec une probabilité réglable pour « Paris »."""
    return [
        make_token("La", 0.99, {"La": 0.99, "Le": 0.01}),
        make_token(" capitale", 0.90, {" capitale": 0.90, " ville": 0.05}),
        make_token(" est", 0.95, {" est": 0.95}),
        make_token(" Paris", paris_probability, {" Paris": paris_probability, " Lyon": 0.30}),
    ]
