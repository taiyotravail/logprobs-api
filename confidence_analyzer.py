"""Analyse de confiance par token, filtrée par catégorie grammaticale (spaCy)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import spacy

from config import CRITICAL_POS_TAGS


@dataclass
class TokenAnalysis:
    """Résultat d'analyse pour un token grammaticalement critique."""

    token: str
    word: str
    pos: str
    probability: float


@dataclass
class AnalysisResult:
    """Synthèse de l'analyse d'une réponse complète."""

    full_text: str
    analyses: list[TokenAnalysis]
    weakest_probability: float
    weakest_token: str

    @property
    def critical_count(self) -> int:
        return len(self.analyses)


def load_spacy_model() -> spacy.language.Language:
    return spacy.load("fr_core_news_md")


def find_critical_tokens(token_data, nlp: spacy.language.Language) -> AnalysisResult:
    """Reconstruit le texte, identifie les tokens grammaticalement critiques.

    Pour chaque token LLM, on retrouve via spaCy le mot complet auquel il
    appartient. Si ce mot est étiqueté comme nom, nom propre, nombre ou verbe,
    sa probabilité est conservée dans la liste critique.
    """
    full_text = "".join(item.token for item in token_data)
    doc = nlp(full_text)

    char_to_word = {}
    for word in doc:
        for char_idx in range(word.idx, word.idx + len(word.text)):
            char_to_word[char_idx] = word

    analyses: list[TokenAnalysis] = []
    cursor = 0

    for item in token_data:
        token_str = item.token
        probability = math.exp(item.logprob) * 100

        first_non_space = cursor
        while (
            first_non_space < cursor + len(token_str)
            and full_text[first_non_space].isspace()
        ):
            first_non_space += 1
        cursor += len(token_str)

        if first_non_space >= cursor:
            continue

        word = char_to_word.get(first_non_space)
        if word and word.pos_ in CRITICAL_POS_TAGS:
            analyses.append(
                TokenAnalysis(
                    token=token_str,
                    word=word.text,
                    pos=word.pos_,
                    probability=probability,
                )
            )

    if analyses:
        weakest = min(analyses, key=lambda a: a.probability)
        weakest_prob, weakest_token = weakest.probability, weakest.token
    else:
        weakest_prob, weakest_token = 100.0, ""

    return AnalysisResult(
        full_text=full_text,
        analyses=analyses,
        weakest_probability=weakest_prob,
        weakest_token=weakest_token,
    )
