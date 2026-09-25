# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Image de production pour le détecteur d'hallucinations LLM.
#
# `--platform=linux/amd64` force la base sur amd64 même quand le build est
# lancé depuis un Mac Apple Silicon. Résultat : l'image produite tourne
# nativement sur Windows / Linux x86_64 (le cas le plus large côté clients).
# Sur Mac ARM elle tournera via émulation QEMU intégrée à Docker Desktop.
#
# Pour un build multi-arch publiable sur un registry :
#   docker buildx build --platform linux/amd64,linux/arm64 -t logprobs-demo .
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
    DEFAULT_MODEL=phi4-mini:latest

WORKDIR /app

# Installation des dépendances Python en couche dédiée pour bénéficier du
# cache Docker (les sources changent plus souvent que les deps).
COPY requirements.txt ./
RUN pip install -r requirements.txt \
 && python -m spacy download fr_core_news_md

# Code applicatif (couche fine, invalidée à chaque changement de source).
COPY app.py config.py llm_client.py confidence_analyzer.py plots.py ./

EXPOSE 8501

# `--server.address=0.0.0.0` indispensable pour que le port soit joignable
# depuis l'hôte via `-p 8501:8501`. `--server.headless=true` évite le prompt
# email d'usage statistique au démarrage.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
