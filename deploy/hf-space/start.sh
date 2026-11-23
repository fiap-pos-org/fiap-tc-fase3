#!/bin/sh
# Sobe o Ollama em segundo plano e, quando ele responder, a API na porta 7860.
set -e
ollama serve &
until ollama list > /dev/null 2>&1; do sleep 1; done
exec uv run --no-sync python -m hospital_assistant.api
