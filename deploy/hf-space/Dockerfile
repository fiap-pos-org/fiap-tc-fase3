# Imagem única (Ollama + aplicação no mesmo container), usada na VPS (deploy/vps) e no
# Hugging Face Spaces. Contexto de build: a raiz do repositório.
FROM ollama/ollama:0.40.14

COPY --from=astral/uv:0.13.5 /uv /bin/uv

# Tudo roda como o usuário de uid 1000 (na imagem do Ollama, "ubuntu"), e os arquivos já nascem
# com esse dono: um "chown -R" no fim duplicaria os ~2,5 GB de modelos numa camada nova.
RUN mkdir -p /app /models/gguf && chown 1000:1000 /app /models/gguf
USER 1000

ENV HOME=/home/ubuntu \
    OLLAMA_MODELS=/home/ubuntu/ollama \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_KEEP_ALIVE=24h \
    OLLAMA_NUM_THREAD=2 \
    STORAGE_DIR=/app/var \
    UV_PYTHON_INSTALL_DIR=/home/ubuntu/python \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Só as dependências de runtime da aplicação (sem dados, treino e testes).
COPY --chown=1000:1000 pyproject.toml uv.lock README.md ./
RUN uv python install 3.12 && uv sync --frozen --no-default-groups --no-install-project

# Modelo ajustado e modelo de embeddings: mudam pouco, então ficam antes do código.
COPY --chown=1000:1000 models/gguf /models/gguf
RUN ollama serve & \
    until ollama list > /dev/null 2>&1; do sleep 1; done && \
    ollama create hospital-assistant -f /models/gguf/Modelfile && \
    ollama pull bge-m3

COPY --chown=1000:1000 src ./src
COPY --chown=1000:1000 data ./data
RUN uv sync --frozen --no-default-groups

# Monta o banco do prontuário e o índice dos protocolos no build, para a aplicação subir rápido.
RUN ollama serve & \
    until ollama list > /dev/null 2>&1; do sleep 1; done && \
    uv run --no-sync python -c "from hospital_assistant.db.database import inicializar; from hospital_assistant.rag.indice import criar_indice; inicializar(); criar_indice()"

COPY --chown=1000:1000 --chmod=755 deploy/hf-space/start.sh /app/start.sh

EXPOSE 7860
ENTRYPOINT ["/app/start.sh"]
