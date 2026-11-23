"""Fluxo B: dúvida clínica do médico, com ou sem paciente selecionado.

verificar_entrada -> (bloqueada) responder_bloqueio
                  -> (valores críticos do paciente) responder_ferramenta
                  -> montar_contexto -> gerar_resposta -> validar_resposta -> (reprovada) resposta_segura

Pedidos de prescrição, tentativas de burlar as regras e perguntas sem base recebem resposta
padrão sem passar pelo modelo. Respostas reprovadas na validação não são mostradas: viram uma
resposta segura, e o texto original fica na trilha de auditoria. Perguntas sobre valores
críticos do paciente são respondidas pela ferramenta (comparação numérica em código), não pelo
modelo, que erra comparações.
"""

import re
import sqlite3
from typing import TypedDict

from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph

from hospital_assistant.assistant.qa import Contexto, montar_contexto
from hospital_assistant.data.build_instructions import resposta_valores_criticos
from hospital_assistant.db.database import buscar_paciente
from hospital_assistant.observability.auditoria import auditado
from hospital_assistant.rag.indice import similaridade_maxima
from hospital_assistant.safety.entrada import classificar_entrada, resposta_bloqueio
from hospital_assistant.safety.guardrails import validar_resposta

_VALORES_CRITICOS = re.compile(r"cr[ií]tic|comunicad\w* com urg", re.IGNORECASE)

RESPOSTA_REPROVADA = (
    "Não consegui fundamentar uma resposta segura nos protocolos internos ({motivos}). "
    "Consulte os trechos de protocolo listados abaixo ou o especialista de plantão.\n\n"
    "Fontes: [PROT-SEGMED-01 §4]"
)


class EstadoDuvida(TypedDict, total=False):
    pergunta: str
    paciente_id: str | None
    trace_id: str
    categoria: str
    similaridade: float | None
    paciente: dict | None
    trechos: list[tuple[str, str]]
    entrada_modelo: str
    resposta: str
    resposta_original: str
    validacao: dict


def criar_fluxo_duvida(conn: sqlite3.Connection, indice: InMemoryVectorStore, cadeia: Runnable):
    def verificar_entrada(estado: EstadoDuvida) -> dict:
        # Perguntas sobre o paciente selecionado são respondidas pelo prontuário, então o
        # teste de "sem base nos protocolos" só vale para perguntas gerais.
        similaridade = (
            None if estado.get("paciente_id") else similaridade_maxima(indice, estado["pergunta"])
        )
        categoria = classificar_entrada(estado["pergunta"], similaridade)
        if (
            categoria == "clinica"
            and estado.get("paciente_id")
            and _VALORES_CRITICOS.search(estado["pergunta"])
        ):
            categoria = "valores_criticos"
        return {"categoria": categoria, "similaridade": similaridade}

    def responder_ferramenta(estado: EstadoDuvida) -> dict:
        paciente = buscar_paciente(conn, estado["paciente_id"])
        resposta, refs = resposta_valores_criticos(paciente)
        fontes = "; ".join(f"[{r}]" for r in refs)
        return {
            "paciente": paciente,
            "resposta": f"{resposta}\n\nFontes: {fontes}",
            "trechos": [],
            "validacao": {
                "aprovada": True,
                "motivos": ["respondida pela ferramenta de valores críticos"],
            },
        }

    def responder_bloqueio(estado: EstadoDuvida) -> dict:
        return {
            "resposta": resposta_bloqueio(estado["categoria"]),
            "trechos": [],
            "paciente": None,
            "validacao": {
                "aprovada": True,
                "motivos": [f"bloqueada na entrada: {estado['categoria']}"],
            },
        }

    def contexto(estado: EstadoDuvida) -> dict:
        ctx = montar_contexto(estado["pergunta"], indice, conn, estado.get("paciente_id"))
        return {"paciente": ctx.paciente, "trechos": ctx.trechos}

    def gerar_resposta(estado: EstadoDuvida) -> dict:
        entrada = Contexto(estado["pergunta"], estado["paciente"], estado["trechos"]).entrada()
        return {"entrada_modelo": entrada, "resposta": cadeia.invoke({"entrada": entrada})}

    def validar(estado: EstadoDuvida) -> dict:
        validacao = validar_resposta(estado["resposta"], estado["paciente"], estado["trechos"])
        return {"validacao": {"aprovada": validacao.aprovada, "motivos": validacao.motivos}}

    def resposta_segura(estado: EstadoDuvida) -> dict:
        motivos = "; ".join(estado["validacao"]["motivos"])
        return {
            "resposta_original": estado["resposta"],
            "resposta": RESPOSTA_REPROVADA.format(motivos=motivos),
        }

    grafo = StateGraph(EstadoDuvida)
    for nome, no in [
        ("verificar_entrada", verificar_entrada),
        ("responder_bloqueio", responder_bloqueio),
        ("responder_ferramenta", responder_ferramenta),
        ("montar_contexto", contexto),
        ("gerar_resposta", gerar_resposta),
        ("validar_resposta", validar),
        ("resposta_segura", resposta_segura),
    ]:
        grafo.add_node(nome, auditado(conn, "duvida", nome, no))
    grafo.add_edge(START, "verificar_entrada")
    destinos = {"clinica": "montar_contexto", "valores_criticos": "responder_ferramenta"}
    grafo.add_conditional_edges(
        "verificar_entrada",
        lambda e: destinos.get(e["categoria"], "responder_bloqueio"),
        ["montar_contexto", "responder_ferramenta", "responder_bloqueio"],
    )
    grafo.add_edge("responder_bloqueio", END)
    grafo.add_edge("responder_ferramenta", END)
    grafo.add_edge("montar_contexto", "gerar_resposta")
    grafo.add_edge("gerar_resposta", "validar_resposta")
    grafo.add_conditional_edges(
        "validar_resposta",
        lambda e: END if e["validacao"]["aprovada"] else "resposta_segura",
        [END, "resposta_segura"],
    )
    grafo.add_edge("resposta_segura", END)
    return grafo.compile()
