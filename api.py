"""API REST du détecteur d'hallucinations (FastAPI).

Endpoints :
- `GET  /health`  : sonde de vie. À appeler dès le chargement du site pour
  « réveiller » le serveur pendant que l'utilisateur tape sa question.
- `POST /analyze` : pose la question au LLM et renvoie la réponse, la décision
  de blocage et le détail des probabilités par token (pour tracer un graphique
  côté site).

Confidentialité (RGPD) : les questions et réponses ne sont jamais écrites dans
les logs ; seuls le type d'erreur, la durée et la décision le sont. Le log
d'accès d'uvicorn (qui contient l'IP du client) est désactivé au lancement
(`--no-access-log`, cf. `start-api.sh`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from config import (
    API_ALLOWED_ORIGINS,
    DAILY_ANALYSIS_LIMIT,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    EXPOSE_BLOCKED_ANSWERS,
    LLM_TIMEOUT_SECONDS,
    MAX_ANSWER_TOKENS,
    MAX_QUESTION_LENGTH,
    SAFETY_MESSAGE,
)
from confidence_analyzer import load_spacy_model
from detection_service import (
    DetectionResult,
    LLMResponseError,
    LLMUnavailableError,
    detect_hallucination,
)
from llm_client import build_client, query_model
from usage_quota import DailyQuota

logger = logging.getLogger("logprobs_api")

Detector = Callable[[str, float], DetectionResult]


@dataclass(frozen=True)
class ApiResources:
    """Ressources lourdes chargées une seule fois au démarrage."""

    detect: Detector
    model_name: str


# --- Schémas d'entrée / sortie --------------------------------------------


class AnalyzeRequest(BaseModel):
    """Corps de `POST /analyze`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="Question posée au LLM (de préférence factuelle).",
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=0,
        le=100,
        description="Seuil de confiance minimum (%) sous lequel la réponse est bloquée.",
    )


class AlternativeOut(BaseModel):
    """Alternative envisagée par le modèle pour un token."""

    token: str
    probability: float


class TokenOut(BaseModel):
    """Token généré, avec sa probabilité et ses alternatives (top 3)."""

    token: str
    probability: float
    alternatives: list[AlternativeOut]


class CriticalTokenOut(BaseModel):
    """Token appartenant à un mot grammaticalement critique."""

    token: str
    word: str
    pos: str
    probability: float


class AnalyzeResponse(BaseModel):
    """Réponse de `POST /analyze`."""

    answer: str = Field(
        description="Texte à afficher : la réponse du LLM, ou le message de sécurité si elle est bloquée."
    )
    blocked: bool
    raw_answer: str | None = Field(
        description=(
            "Réponse brute du LLM. Si elle est bloquée et que EXPOSE_BLOCKED_ANSWERS=false, "
            "vaut null (et `tokens` / `critical_tokens` sont vides)."
        )
    )
    threshold: float
    weakest_probability: float
    weakest_token: str
    critical_tokens: list[CriticalTokenOut]
    tokens: list[TokenOut]
    model: str


class HealthResponse(BaseModel):
    """Réponse de `GET /health`."""

    status: Literal["ok"]
    model: str


# --- Logique ----------------------------------------------------------------


def to_response(result: DetectionResult, model_name: str, expose_blocked_answers: bool) -> AnalyzeResponse:
    """Convertit un résultat de détection en réponse JSON de l'API.

    Args:
        result: résultat de la détection.
        model_name: nom du modèle, renvoyé pour information.
        expose_blocked_answers: si `False`, une réponse bloquée n'apparaît
            nulle part dans le JSON (ni en clair, ni token par token).
    """
    analysis = result.analysis
    if result.is_blocked and not expose_blocked_answers:
        return AnalyzeResponse(
            answer=SAFETY_MESSAGE,
            blocked=True,
            raw_answer=None,
            threshold=result.threshold,
            weakest_probability=analysis.weakest_probability,
            weakest_token="",
            critical_tokens=[],
            tokens=[],
            model=model_name,
        )
    return AnalyzeResponse(
        answer=SAFETY_MESSAGE if result.is_blocked else result.raw_answer,
        blocked=result.is_blocked,
        raw_answer=result.raw_answer,
        threshold=result.threshold,
        weakest_probability=analysis.weakest_probability,
        weakest_token=analysis.weakest_token,
        critical_tokens=[
            CriticalTokenOut(token=a.token, word=a.word, pos=a.pos, probability=a.probability)
            for a in analysis.analyses
        ],
        tokens=[
            TokenOut(
                token=t.token,
                probability=t.probability,
                alternatives=[
                    AlternativeOut(token=alt.token, probability=alt.probability)
                    for alt in t.alternatives
                ],
            )
            for t in result.tokens
        ],
        model=model_name,
    )


def warm_up_model(client: openai.OpenAI, model: str) -> None:
    """Charge le modèle en mémoire dès le démarrage (au mieux).

    Sans ça, le premier visiteur paierait en plus le chargement du modèle.
    Un échec n'empêche pas le démarrage : la première requête le chargera.
    """
    started = time.perf_counter()
    try:
        query_model(client, model, "Bonjour", max_tokens=1)
    except openai.OpenAIError as exc:
        logger.warning("Préchauffage du modèle impossible (%s).", type(exc).__name__)
        return
    logger.info("Modèle %s chargé en %.1fs.", model, time.perf_counter() - started)


def build_production_resources() -> ApiResources:
    """Charge spaCy, se connecte à Ollama et préchauffe le modèle."""
    nlp = load_spacy_model()
    # Pas de nouvelle tentative : sur CPU, relancer une inférence qui a
    # expiré ne ferait qu'allonger la file d'attente.
    client = build_client(timeout=LLM_TIMEOUT_SECONDS, max_retries=0)
    warm_up_model(client, DEFAULT_MODEL)

    def detect(question: str, threshold: float) -> DetectionResult:
        return detect_hallucination(
            client, nlp, DEFAULT_MODEL, question, threshold, max_tokens=MAX_ANSWER_TOKENS
        )

    return ApiResources(detect=detect, model_name=DEFAULT_MODEL)


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Renvoie une erreur 422 sans recopier les valeurs reçues.

    Le gestionnaire par défaut de FastAPI recopie l'entrée fautive : il plante
    sur un seuil `NaN` (non sérialisable en JSON), et sa trace d'erreur
    finirait alors dans les logs avec la question du visiteur (RGPD).
    """
    errors = [
        {"loc": list(error.get("loc", ())), "msg": error.get("msg", ""), "type": error.get("type", "")}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


def create_app(
    load_resources: Callable[[], ApiResources] = build_production_resources,
    allowed_origins: list[str] = API_ALLOWED_ORIGINS,
    daily_limit: int = DAILY_ANALYSIS_LIMIT,
    expose_blocked_answers: bool = EXPOSE_BLOCKED_ANSWERS,
) -> FastAPI:
    """Construit l'application FastAPI.

    Args:
        load_resources: fabrique des ressources lourdes, appelée au démarrage
            (remplaçable par un faux dans les tests).
        allowed_origins: origines autorisées par CORS.
        daily_limit: nombre max d'analyses par jour (0 = illimité).
        expose_blocked_answers: renvoyer ou non le contenu des réponses bloquées.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Chargement bloquant : le serveur n'accepte de connexions qu'une fois
        # prêt. L'hébergeur retient les requêtes pendant ce temps (cold start).
        app.state.resources = await run_in_threadpool(load_resources)
        yield

    app = FastAPI(
        title="Détecteur d'hallucinations LLM",
        description="Analyse les logprobs d'un LLM pour bloquer les réponses peu fiables.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
        max_age=86400,
    )
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    quota = DailyQuota(daily_limit)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Indique que le serveur est démarré et le modèle prêt."""
        resources: ApiResources = request.app.state.resources
        return HealthResponse(status="ok", model=resources.model_name)

    @app.post(
        "/analyze",
        response_model=AnalyzeResponse,
        responses={
            429: {"description": "Quota quotidien de la démo atteint."},
            502: {"description": "Réponse du LLM inexploitable."},
            503: {"description": "LLM indisponible ou surchargé."},
        },
    )
    def analyze(payload: AnalyzeRequest, request: Request) -> AnalyzeResponse:
        """Pose la question au LLM et bloque la réponse si elle est peu fiable."""
        if not quota.try_consume():
            raise HTTPException(
                status_code=429,
                detail="Le quota quotidien de la démo est atteint. Réessayez demain.",
            )

        resources: ApiResources = request.app.state.resources
        started = time.perf_counter()
        try:
            result = resources.detect(payload.question, payload.threshold)
        except LLMUnavailableError as exc:
            logger.warning("LLM indisponible (%s).", exc)
            raise HTTPException(
                status_code=503,
                detail="Le modèle est momentanément indisponible ou surchargé. Réessayez dans quelques secondes.",
            ) from exc
        except LLMResponseError as exc:
            logger.error("Réponse du LLM inexploitable (%s).", exc)
            raise HTTPException(
                status_code=502,
                detail="Le modèle a renvoyé une réponse inexploitable.",
            ) from exc

        logger.info(
            "Analyse en %.1fs — bloquée=%s, maillon faible=%.1f%%.",
            time.perf_counter() - started,
            result.is_blocked,
            result.analysis.weakest_probability,
        )
        return to_response(result, resources.model_name, expose_blocked_answers)

    return app


# Point d'entrée uvicorn (`uvicorn api:app`). Les logs applicatifs ne
# contiennent ni question, ni réponse, ni IP (cf. docstring du module).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
