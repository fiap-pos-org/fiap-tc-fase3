"""Configuração do projeto: caminhos e serviços, centralizados para todos os módulos."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"

MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Serviço do modelo (no docker compose, o host é o container "ollama").
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "hospital-assistant")
# Threads de CPU para a inferência. Usar o número de núcleos físicos da máquina: o padrão do
# Ollama (todos os núcleos lógicos) ficou 6x mais lento num notebook de 16 threads ocupado.
OLLAMA_NUM_THREAD = int(os.environ.get("OLLAMA_NUM_THREAD", "8"))
