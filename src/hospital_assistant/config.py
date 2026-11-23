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

# Estado gerado em runtime (banco do prontuário, índice de busca). No Docker é um volume.
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", ROOT_DIR / "var"))
DB_PATH = STORAGE_DIR / "hospital.db"
INDICE_PATH = STORAGE_DIR / "indice_protocolos.json"

# Serviço do modelo (no docker compose, o host é o container "ollama").
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "hospital-assistant")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
# Threads de CPU para a inferência. Usar o número de núcleos físicos da máquina: o padrão do
# Ollama (todos os núcleos lógicos) ficou 6x mais lento num notebook de 16 threads ocupado.
OLLAMA_NUM_THREAD = int(os.environ.get("OLLAMA_NUM_THREAD", "8"))

# Login da aplicação (deploy público). Sem APP_USUARIO/APP_SENHA, o login fica desligado (uso local).
APP_USUARIO = os.environ.get("APP_USUARIO")
APP_SENHA = os.environ.get("APP_SENHA")
