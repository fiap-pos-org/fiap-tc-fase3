# Tech Challenge: Fase 3

Projeto da Fase 3 da Pós Tech IA para Devs (FIAP): assistente virtual médico de um hospital fictício, com fine-tuning de LLM em dados internos sintéticos e anonimizados, orquestrado com LangChain e LangGraph.

O enunciado completo está em [`desafio/`](desafio/).

## Objetivo

- Ajustar (fine-tuning) uma LLM aberta com protocolos, perguntas frequentes e modelos de documentos do hospital.
- Responder dúvidas clínicas contextualizadas com o prontuário do paciente, citando a fonte.
- Executar fluxos automatizados (exames pendentes, alertas, sugestões) com **aprovação humana obrigatória** antes de qualquer conduta.
- Registrar logs detalhados para auditoria.

## Estrutura

```
├── desafio/                  # enunciado do Tech Challenge
├── src/hospital_assistant/   # código do projeto
│   └── config.py             # caminhos centralizados
└── tests/                    # testes automatizados
```

A estrutura cresce a cada fase (dados, fine-tuning, RAG, grafos, segurança e avaliação).

## Requisitos

- [uv](https://docs.astral.sh/uv/) (instala o Python 3.12 automaticamente)

## Como rodar

```bash
uv sync            # cria o ambiente e instala as dependências
uv run pytest      # roda os testes
uv run ruff check  # lint
```

## Status

Em desenvolvimento.
