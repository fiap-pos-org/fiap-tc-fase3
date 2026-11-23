#!/bin/sh
# Monta o conteúdo do Hugging Face Space e envia com a CLI `hf` (precisa de `hf auth login`).
# Pré-requisito: models/gguf/ com o GGUF e o Modelfile (gerados pelo export + serviço gguf do compose).
#
# Uso: deploy/publicar_space.sh <usuario>/<nome-do-space>
set -eu

SPACE="${1:?informe o Space, ex.: william-devops/hospital-assistant}"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
PASTA="$(mktemp -d)"
trap 'rm -rf "$PASTA"' EXIT

test -f "$RAIZ/models/gguf/hospital-assistant-q4_k_m.gguf" || { echo "GGUF não encontrado"; exit 1; }

# Só o necessário para a aplicação: código, documentos internos, prontuários anonimizados e modelo.
mkdir -p "$PASTA/data/raw" "$PASTA/data/processed" "$PASTA/models/gguf"
cp "$RAIZ/deploy/hf-space/Dockerfile" "$RAIZ/deploy/hf-space/README.md" "$PASTA/"
mkdir -p "$PASTA/deploy/hf-space" && cp "$RAIZ/deploy/hf-space/start.sh" "$PASTA/deploy/hf-space/"
cp "$RAIZ/pyproject.toml" "$RAIZ/uv.lock" "$PASTA/"
cp -r "$RAIZ/src" "$PASTA/src"
find "$PASTA/src" -name __pycache__ -type d -prune -exec rm -rf {} +
cp -r "$RAIZ/data/raw/protocolos" "$RAIZ/data/raw/modelos" "$PASTA/data/raw/"
cp "$RAIZ/data/processed/prontuarios.jsonl" "$PASTA/data/processed/"
cp "$RAIZ/models/gguf/hospital-assistant-q4_k_m.gguf" "$RAIZ/models/gguf/Modelfile" "$PASTA/models/gguf/"

hf repos create "$SPACE" --type space --sdk docker --public --exist-ok
hf upload "$SPACE" "$PASTA" . --repo-type space --commit-message "Publica o assistente médico"
echo "Publicado: https://huggingface.co/spaces/$SPACE"
