# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Image de l'API REST : Ollama + phi4-mini + FastAPI dans un seul conteneur.
#
# Le modèle est téléchargé au BUILD et embarqué dans l'image (~4 Go) : au
# démarrage à froid, rien à télécharger, seulement à charger en RAM.
#
# Pensée pour Google Cloud Run (écoute sur $PORT, 8080 par défaut), mais
# tourne sur n'importe quel hôte Docker :
#   docker build -t logprobs-api .
#   docker run --rm -p 8080:8080 logprobs-api
#
# L'UI Streamlit a son propre fichier : Dockerfile.streamlit.
# ---------------------------------------------------------------------------

ARG OLLAMA_VERSION=0.33.1
ARG OLLAMA_MODEL=phi4-mini:latest

# --- Étape 1 : Ollama, version CPU uniquement --------------------------------
# L'archive officielle embarque CUDA/ROCm/Vulkan (~1,2 Go) : inutiles sans GPU.
FROM debian:trixie-slim AS ollama
ARG OLLAMA_VERSION
ARG TARGETARCH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl zstd \
 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/ollama \
 && curl -fsSL "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-${TARGETARCH:-amd64}.tar.zst" \
  | zstd -dc \
  | tar -x -C /opt/ollama \
      --exclude='lib/ollama/cuda_*' \
      --exclude='lib/ollama/rocm*' \
      --exclude='lib/ollama/vulkan*' \
      --exclude='lib/ollama/mlx*' \
 && test -x /opt/ollama/bin/ollama

# --- Étape 2 : image finale --------------------------------------------------
FROM python:3.12-slim
ARG OLLAMA_MODEL

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Ollama : écoute en local uniquement, modèle gardé en RAM, une inférence
    # à la fois (CPU) et file d'attente courte (au-delà : HTTP 503).
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_QUEUE=4 \
    OLLAMA_CONTEXT_LENGTH=2048 \
    OLLAMA_BASE_URL=http://127.0.0.1:11434/v1 \
    DEFAULT_MODEL=${OLLAMA_MODEL} \
    PORT=8080

# Binaire dans /usr/local/bin, bibliothèques dans /usr/local/lib/ollama
# (Ollama les cherche dans ../lib/ollama relativement au binaire).
COPY --from=ollama /opt/ollama/ /usr/local/

# Utilisateur non-root. uid 1000 : compatible aussi avec Hugging Face Spaces.
RUN useradd --create-home --uid 1000 app

# 1) Modèle — couche la plus lourde, placée tôt pour rester en cache quand le
#    code ou les dépendances changent. La clé d'identité générée par
#    `ollama serve` est supprimée pour ne pas être figée dans l'image.
USER app
RUN export HOME=/home/app; \
    ollama serve > /dev/null 2>&1 & \
    for attempt in $(seq 1 30); do ollama list > /dev/null 2>&1 && break; sleep 1; done; \
    ollama pull "${DEFAULT_MODEL}" \
 && rm -f "${HOME}/.ollama/id_ed25519" "${HOME}/.ollama/id_ed25519.pub"

# 2) Dépendances Python (installation système, donc en root).
USER root
WORKDIR /app
COPY requirements-api.txt ./
RUN pip install -r requirements-api.txt \
 && python -m spacy download fr_core_news_md

# 3) Code applicatif, en lecture seule pour l'utilisateur `app`.
COPY api.py config.py confidence_analyzer.py detection_service.py llm_client.py usage_quota.py start-api.sh ./

# HOME explicite : Ollama y écrit sa clé d'identité au démarrage.
USER app
ENV HOME=/home/app
EXPOSE 8080
CMD ["sh", "start-api.sh"]
