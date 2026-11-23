"""Gera os diagramas Mermaid dos fluxos LangGraph em docs/diagramas/.

Uso: uv run python -m hospital_assistant.graph.diagramas
"""

import sqlite3

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver

from hospital_assistant.assistant.qa import criar_cadeia
from hospital_assistant.config import ROOT_DIR
from hospital_assistant.graph.duvida import criar_fluxo_duvida
from hospital_assistant.graph.triagem import criar_fluxo_triagem
from hospital_assistant.rag.indice import criar_indice

SAIDA = ROOT_DIR / "docs" / "diagramas"

if __name__ == "__main__":
    # A estrutura do grafo não depende do modelo nem dos dados: usa dublês para montar.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    indice = criar_indice(DeterministicFakeEmbedding(size=8), path=None)
    cadeia = criar_cadeia(FakeListChatModel(responses=[""]))
    SAIDA.mkdir(parents=True, exist_ok=True)
    fluxos = {
        "fluxo_triagem": criar_fluxo_triagem(conn, indice, cadeia, InMemorySaver()),
        "fluxo_duvida": criar_fluxo_duvida(conn, indice, cadeia),
    }
    for nome, fluxo in fluxos.items():
        (SAIDA / f"{nome}.mmd").write_text(fluxo.get_graph().draw_mermaid(), encoding="utf-8")
        print(SAIDA / f"{nome}.mmd")
