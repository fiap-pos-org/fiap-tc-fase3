"""Fluxo B: dúvida clínica do médico, com ou sem paciente selecionado.

montar_contexto -> gerar_resposta -> validar_resposta
"""

import sqlite3
from typing import TypedDict

from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph

from hospital_assistant.assistant.qa import Contexto, montar_contexto
from hospital_assistant.safety.guardrails import validar_resposta


class EstadoDuvida(TypedDict, total=False):
    pergunta: str
    paciente_id: str | None
    trace_id: str
    paciente: dict | None
    trechos: list[tuple[str, str]]
    resposta: str
    validacao: dict


def criar_fluxo_duvida(conn: sqlite3.Connection, indice: InMemoryVectorStore, cadeia: Runnable):
    def contexto(estado: EstadoDuvida) -> dict:
        ctx = montar_contexto(estado["pergunta"], indice, conn, estado.get("paciente_id"))
        return {"paciente": ctx.paciente, "trechos": ctx.trechos}

    def gerar_resposta(estado: EstadoDuvida) -> dict:
        ctx = Contexto(estado["pergunta"], estado["paciente"], estado["trechos"])
        return {"resposta": cadeia.invoke({"entrada": ctx.entrada()})}

    def validar(estado: EstadoDuvida) -> dict:
        validacao = validar_resposta(estado["resposta"], estado["paciente"])
        return {"validacao": {"aprovada": validacao.aprovada, "motivos": validacao.motivos}}

    grafo = StateGraph(EstadoDuvida)
    grafo.add_node("montar_contexto", contexto)
    grafo.add_node("gerar_resposta", gerar_resposta)
    grafo.add_node("validar_resposta", validar)
    grafo.add_edge(START, "montar_contexto")
    grafo.add_edge("montar_contexto", "gerar_resposta")
    grafo.add_edge("gerar_resposta", "validar_resposta")
    grafo.add_edge("validar_resposta", END)
    return grafo.compile()
