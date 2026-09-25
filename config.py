"""Constantes globales du détecteur d'hallucinations.

`OLLAMA_BASE_URL` et `DEFAULT_MODEL` sont surchargeables via variables
d'environnement — utile pour pointer vers l'hôte Docker (`host.docker.internal`)
ou pour changer de modèle sans toucher au code.

Les constantes `API_*` / `MAX_*` / `LLM_*` ne concernent que l'API REST
(`api.py`) et sont elles aussi surchargeables via l'environnement.
"""

import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "phi4-mini:latest")

CRITICAL_POS_TAGS = ("PROPN", "NOUN", "NUM", "VERB")

SAFETY_MESSAGE = (
    "Je ne dispose pas de cette information avec une fiabilité suffisante. "
    "Je transfère votre demande à un opérateur humain."
)

DEFAULT_THRESHOLD = 70

# --- API REST -------------------------------------------------------------

# Borne la taille du prompt : protège le CPU du serveur contre les abus.
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "500"))

# Borne la longueur de la réponse générée (le prompt système demande une
# seule phrase, mais un petit modèle peut déborder).
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "150"))

# Temps max d'une inférence avant de rendre la main (CPU lent + file d'attente).
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

# Nombre max d'analyses par jour (tous visiteurs confondus). Garde-fou de
# coût pour rester dans l'offre gratuite de l'hébergeur (ordre de grandeur :
# 100 analyses/jour × ~10 s × 30 jours × 4 vCPU ≈ 120 000 vCPU-s/mois, soit
# ~2/3 des 180 000 gratuits de Cloud Run, hors temps de réveil). 0 = illimité.
DAILY_ANALYSIS_LIMIT = int(os.getenv("DAILY_ANALYSIS_LIMIT", "100"))

# Si `true`, une réponse bloquée reste visible dans le JSON (`raw_answer`,
# `tokens`) : nécessaire pour la démo « protection on/off ». En production,
# mettre `false` : sinon n'importe qui peut lire la réponse bloquée dans les
# outils de développement du navigateur.
EXPOSE_BLOCKED_ANSWERS = os.getenv("EXPOSE_BLOCKED_ANSWERS", "true").strip().lower() in {"1", "true", "yes"}

# Origines autorisées à appeler l'API depuis un navigateur (CORS), séparées
# par des virgules. Ex : "https://mon-site.fr,https://www.mon-site.fr".
API_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("API_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
