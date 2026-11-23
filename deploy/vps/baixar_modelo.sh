#!/bin/sh
# Baixa o modelo ajustado (GGUF Q4_K_M) e o Modelfile do Hugging Face para models/gguf/.
set -eu
DESTINO="$(cd "$(dirname "$0")/../.." && pwd)/models/gguf"
BASE="https://huggingface.co/william-devops/hospital-assistant-gguf/resolve/main"
mkdir -p "$DESTINO"
for arquivo in hospital-assistant-q4_k_m.gguf Modelfile; do
  curl -fL --retry 3 -o "$DESTINO/$arquivo" "$BASE/$arquivo"
done
ls -la "$DESTINO"
