#!/bin/sh
# Démarre Ollama en arrière-plan, attend qu'il réponde, puis lance l'API.
# Utilisé comme commande de l'image Docker (cf. Dockerfile).
set -eu

ollama serve &
ollama_pid=$!

attempt=0
until ollama list > /dev/null 2>&1; do
  if ! kill -0 "$ollama_pid" 2> /dev/null; then
    echo "Ollama s'est arrêté au démarrage." >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 60 ]; then
    echo "Ollama ne répond pas après 60 s." >&2
    exit 1
  fi
  sleep 1
done

# --no-access-log : pas d'adresse IP de visiteur dans les logs (RGPD).
exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8080}" --no-access-log
