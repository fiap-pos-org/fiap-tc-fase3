# Imagem única para o Hugging Face Spaces: Ollama + aplicação no mesmo container.
# O Spaces não roda docker compose; o compose da raiz continua sendo o jeito de rodar local.
FROM ollama/ollama:0.40.14

COPY --from=astral/uv:0.13.5 /uv /bin/uv

# O Spaces executa como o usuário de uid 1000 (na imagem do Ollama, "ubuntu").
ENV HOME=/home/ubuntu \
    OLLAMA_MODELS=/home/ubuntu/ollama \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_KEEP_ALIVE=24h \
    OLLAMA_NUM_THREAD=2 \
    STORAGE_DIR=/app/var \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Só as dependências de runtime da aplicação (sem dados, treino e testes).
COPY pyproject.toml uv.lock README.md ./
RUN uv python install 3.12 && uv sync --frozen --no-default-groups --no-install-project

COPY src ./src
COPY data ./data
COPY models/gguf /models/gguf
RUN uv sync --frozen --no-default-groups

# No build: importa o modelo ajustado (GGUF Q4_K_M), baixa o modelo de embeddings e monta
# o banco do prontuário e o índice dos protocolos, para o Space subir rápido.
RUN ollama serve & \
    until ollama list > /dev/null 2>&1; do sleep 1; done && \
    ollama create hospital-assistant -f /models/gguf/Modelfile && \
    ollama pull bge-m3 && \
    uv run --no-sync python -c "from hospital_assistant.db.database import inicializar; from hospital_assistant.rag.indice import criar_indice; inicializar(); criar_indice()"

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh && chown -R 1000:1000 /app /home/ubuntu
USER 1000

EXPOSE 7860
ENTRYPOINT ["/app/start.sh"]
