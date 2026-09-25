"""Constantes globales du détecteur d'hallucinations.

`OLLAMA_BASE_URL` et `DEFAULT_MODEL` sont surchargeables via variables
d'environnement — utile pour pointer vers l'hôte Docker (`host.docker.internal`)
ou pour changer de modèle sans toucher au code.
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
