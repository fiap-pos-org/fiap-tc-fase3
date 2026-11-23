FROM python:3.12.15-slim-bookworm

COPY --from=astral/uv:0.13.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Só as dependências de runtime da app (sem dados, treino e testes).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-default-groups

EXPOSE 7860
CMD ["uv", "run", "--no-sync", "python", "-m", "hospital_assistant.api"]
