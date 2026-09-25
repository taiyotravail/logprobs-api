"""Détecteur d'hallucinations basé sur l'analyse des logprobs (UI Streamlit).

Pose une question à un LLM local (Ollama), analyse la confiance de chaque token
généré, et bloque la réponse si un mot grammaticalement critique (nom, nombre,
verbe...) a été produit avec une probabilité inférieure au seuil défini.

Toggle de protection : on/off pour démontrer l'effet du filtre. La dernière
réponse est mise en cache pour permettre de basculer instantanément entre
"IA brute" et "IA sécurisée" sans relancer l'inférence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from config import DEFAULT_MODEL, DEFAULT_THRESHOLD, OLLAMA_BASE_URL, SAFETY_MESSAGE
from confidence_analyzer import AnalysisResult, find_critical_tokens, load_spacy_model
from llm_client import build_client, query_model
from plots import plot_token_probabilities


@dataclass
class CachedAnswer:
    """Résultat d'une inférence, conservé en session pour rejouer la vue."""

    question: str
    raw_answer: str
    token_data: Any
    analysis: AnalysisResult


@st.cache_resource(show_spinner="Chargement du modèle linguistique français...")
def _cached_nlp():
    return load_spacy_model()


@st.cache_resource
def _cached_client():
    return build_client()


def render_protection_banner(protection_on: bool) -> None:
    """Bandeau visuel — pensé pour la démo vidéo (gros et coloré)."""
    if protection_on:
        st.markdown(
            """
            <div style="
                background:linear-gradient(90deg,#16a34a 0%,#22c55e 100%);
                color:white;padding:18px 24px;border-radius:12px;
                font-size:1.15rem;font-weight:600;text-align:center;
                box-shadow:0 4px 12px rgba(34,197,94,0.3);margin-bottom:1rem;">
                Protection anti-hallucination — ACTIVE
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                background:linear-gradient(90deg,#dc2626 0%,#ef4444 100%);
                color:white;padding:18px 24px;border-radius:12px;
                font-size:1.15rem;font-weight:600;text-align:center;
                box-shadow:0 4px 12px rgba(239,68,68,0.3);margin-bottom:1rem;">
                ⚠️ IA non sécurisée — Réponse brute affichée
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_response(raw_answer: str, is_blocked: bool, protection_on: bool) -> None:
    st.subheader("Réponse")
    if protection_on and is_blocked:
        st.error(f"🚫 {SAFETY_MESSAGE}")
        with st.expander("Voir la réponse brute du modèle (bloquée par la sécurité)"):
            st.write(raw_answer or "(vide)")
    elif protection_on:
        st.success(raw_answer or "(vide)")
    else:
        st.warning(raw_answer or "(vide)")
        if is_blocked:
            st.caption(
                "💡 Cette réponse aurait été bloquée si la protection était active "
                "(maillon faible sous le seuil)."
            )


def render_diagnostic(weakest_prob: float, weakest_token: str, threshold: float, critical_count: int) -> None:
    st.subheader("Diagnostic")
    st.metric("Maillon faible", f"{weakest_prob:.1f}%", help=f"Token : {weakest_token!r}")
    st.metric("Seuil", f"{threshold:.0f}%")
    st.metric("Mots critiques", critical_count)


def render_critical_details(analyses) -> None:
    if not analyses:
        return
    with st.expander("Détails des mots critiques (POS)"):
        st.dataframe(
            [
                {
                    "Token": a.token,
                    "Mot complet": a.word,
                    "Type": a.pos,
                    "Confiance (%)": round(a.probability, 2),
                }
                for a in analyses
            ],
            use_container_width=True,
        )


def render_cached_view(cached: CachedAnswer, threshold: float, protection_on: bool) -> None:
    """Rejoue la vue à partir du dernier résultat sans rappeler le LLM."""
    is_blocked = cached.analysis.weakest_probability < threshold

    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_response(cached.raw_answer, is_blocked, protection_on)
    with col_right:
        render_diagnostic(
            cached.analysis.weakest_probability,
            cached.analysis.weakest_token,
            threshold,
            cached.analysis.critical_count,
        )

    st.subheader("Probabilités par token")
    fig = plot_token_probabilities(cached.token_data)
    st.pyplot(fig)
    plt.close(fig)

    render_critical_details(cached.analysis.analyses)


def fetch_and_cache_answer(question: str, model: str) -> bool:
    """Appelle le LLM et stocke le résultat en session. Retourne True si OK."""
    client = _cached_client()
    nlp = _cached_nlp()

    try:
        response = query_model(client, model, question)
    except Exception as exc:
        st.error(
            f"Impossible d'interroger Ollama ({OLLAMA_BASE_URL}). "
            f"Vérifie que le service est lancé (`ollama serve`).\n\n"
            f"Détail : {exc}"
        )
        return False

    token_data = response.choices[0].logprobs.content
    raw_answer = response.choices[0].message.content.strip()
    analysis = find_critical_tokens(token_data, nlp)

    st.session_state.cached_answer = CachedAnswer(
        question=question,
        raw_answer=raw_answer,
        token_data=token_data,
        analysis=analysis,
    )
    return True


def render_sidebar() -> tuple[str, float]:
    with st.sidebar:
        st.header("Paramètres")
        model = st.text_input("Modèle Ollama", value=DEFAULT_MODEL)
        threshold = st.slider(
            "Seuil de confiance minimum (%)",
            min_value=0,
            max_value=100,
            value=DEFAULT_THRESHOLD,
            step=5,
            help=(
                "Si la probabilité du token critique le plus faible passe sous ce "
                "seuil, la réponse est bloquée et un transfert humain est proposé."
            ),
        )
        st.markdown("---")
        st.caption(
            "Le filtre cible les mots grammaticalement critiques (noms, "
            "nombres, verbes, noms propres) via spaCy `fr_core_news_md`."
        )
    return model, float(threshold)


def main() -> None:
    st.set_page_config(
        page_title="Détecteur d'hallucinations LLM",
        layout="wide",
    )
    st.title("Détecteur d'hallucinations LLM")
    st.caption(
        "Analyse les logprobs pour bloquer les réponses peu fiables avant qu'elles "
        "n'atteignent l'utilisateur."
    )

    model, threshold = render_sidebar()

    # Le toggle pilote la vue : ON = filtre actif, OFF = IA brute.
    # Pensé pour la démo : on bascule la vue sans relancer le LLM.
    protection_on = st.toggle(
        "Activer la protection anti-hallucination",
        value=True,
        help=(
            "OFF : l'utilisateur voit la réponse brute du LLM, même si elle est "
            "peu fiable. ON : les réponses dont la confiance tombe sous le seuil "
            "sont interceptées et remplacées par un message de transfert humain."
        ),
    )
    render_protection_banner(protection_on)

    question = st.text_area(
        "Pose une question (de préférence factuelle, ex: une statistique précise) :",
        value="Selon le rapport officiel de l'ARCEP, quel était le nombre exact, à l'unité près, de foyers raccordables à la fibre optique (FttH) en France au premier trimestre 2021 ?",
        height=80,
    )

    col_a, col_b = st.columns([1, 4])
    with col_a:
        analyze_clicked = st.button("Analyser", type="primary", use_container_width=True)
    with col_b:
        if "cached_answer" in st.session_state:
            if st.button("Effacer la réponse", use_container_width=False):
                del st.session_state.cached_answer
                st.rerun()

    if analyze_clicked:
        if not question.strip():
            st.warning("Saisis une question.")
            return
        with st.spinner("Le modèle réfléchit..."):
            fetch_and_cache_answer(question.strip(), model)

    cached = st.session_state.get("cached_answer")
    if cached is not None:
        render_cached_view(cached, threshold, protection_on)


if __name__ == "__main__":
    main()
