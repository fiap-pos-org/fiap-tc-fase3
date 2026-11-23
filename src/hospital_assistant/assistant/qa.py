"""Pergunta clínica contextualizada: prontuário do banco + trechos de protocolo + modelo ajustado."""

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore

from hospital_assistant.db.database import buscar_paciente
from hospital_assistant.prompts import SYSTEM_PROMPT, format_user_message
from hospital_assistant.rag.indice import buscar


@dataclass
class Contexto:
    """O que foi entregue ao modelo para responder (base da explicabilidade)."""

    pergunta: str
    paciente: dict | None = None
    trechos: list[tuple[str, str]] = field(default_factory=list)

    def mensagens(self) -> list[tuple[str, str]]:
        user = format_user_message(self.pergunta, paciente=self.paciente, trechos=self.trechos)
        return [("system", SYSTEM_PROMPT), ("human", user)]


def montar_contexto(
    pergunta: str,
    indice: InMemoryVectorStore,
    conn: sqlite3.Connection,
    paciente_id: str | None = None,
    k: int = 3,
) -> Contexto:
    paciente = buscar_paciente(conn, paciente_id) if paciente_id else None
    trechos = buscar(indice, pergunta, k=k)
    if paciente:
        # Intercala a busca só pela pergunta (acha, por exemplo, a tabela de valores críticos)
        # com a busca pela pergunta + diagnóstico (acha o protocolo do caso, para condutas).
        do_caso = buscar(indice, f"{pergunta} {paciente['diagnostico']}", k=k)
        intercalados = [t for par in zip(trechos, do_caso, strict=True) for t in par]
        trechos = list(dict.fromkeys(intercalados))[:k]
    return Contexto(pergunta, paciente, trechos)


def responder(contexto: Contexto, llm: BaseChatModel) -> Iterator[str]:
    """Gera a resposta em streaming (texto acumulado)."""
    resposta = ""
    for pedaco in llm.stream(contexto.mensagens()):
        resposta += pedaco.content
        yield resposta
