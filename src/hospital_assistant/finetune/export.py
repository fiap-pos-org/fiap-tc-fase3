"""Mescla o adapter LoRA no modelo base e prepara a importação no Ollama.

Gera em models/merged/ os pesos mesclados (safetensors, fp16) e o tokenizer, e em
models/gguf/ o Modelfile. No docker compose, o serviço `gguf` converte o modelo para GGUF
e quantiza em Q4_K_M (llama.cpp), e o `ollama-init` importa o resultado no Ollama.

Uso: uv run --group train python -m hospital_assistant.finetune.export
"""

import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM

from hospital_assistant.config import MODELS_DIR
from hospital_assistant.finetune.train import ADAPTER_DIR, BASE_MODEL
from hospital_assistant.prompts import SYSTEM_PROMPT

MERGED_DIR = MODELS_DIR / "merged"
GGUF_DIR = MODELS_DIR / "gguf"
GGUF_NOME = "hospital-assistant-q4_k_m.gguf"

# Template ChatML do Qwen2.5 (o mesmo formato usado no treino).
TEMPLATE = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}
{{- end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}"""


def modelfile() -> str:
    return (
        f"FROM /models/gguf/{GGUF_NOME}\n"
        f'TEMPLATE """{TEMPLATE}"""\n'
        f'SYSTEM """{SYSTEM_PROMPT}"""\n'
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|im_start|>"\n'
        "PARAMETER temperature 0\n"
        "PARAMETER num_ctx 4096\n"
    )


def copiar_tokenizer_original() -> None:
    """Copia os arquivos de tokenizer originais do modelo base.

    Regravar com o transformers v5 muda o formato do tokenizer_config.json, que o conversor
    GGUF do llama.cpp (transformers v4) não consegue ler.
    """
    arquivos = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    origem = Path(snapshot_download(BASE_MODEL, allow_patterns=arquivos))
    for nome in arquivos:
        shutil.copy(origem / nome, MERGED_DIR / nome)


def main() -> None:
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).merge_and_unload()
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    copiar_tokenizer_original()
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    (GGUF_DIR / GGUF_NOME).unlink(missing_ok=True)  # força o compose a reconverter
    (GGUF_DIR / "Modelfile").write_text(modelfile(), encoding="utf-8")
    print(f"Modelo mesclado salvo em {MERGED_DIR}")


if __name__ == "__main__":
    main()
