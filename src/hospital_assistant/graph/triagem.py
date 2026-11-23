"""Fluxo A: triagem automatizada de um paciente, com aprovação humana obrigatória.

carregar_prontuario -> verificar_pendencias -> checar_valores_criticos -> [emitir_alertas]
-> buscar_protocolos -> gerar_sugestao -> validar_seguranca -> aprovacao_humana (pausa)
-> registrar_conduta

As verificações (pendências, valores críticos, segurança) são feitas em código; o modelo só
redige a sugestão a partir do prontuário e dos trechos de protocolo. Nenhuma sugestão vale
sem a decisão do médico, que retoma o fluxo pausado.
"""

import sqlite3
from typing import TypedDict

from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from hospital_assistant.assistant.ferramentas import criar_ferramentas
from hospital_assistant.assistant.qa import Contexto
from hospital_assistant.db.database import registrar_alerta, registrar_conduta
from hospital_assistant.safety.guardrails import fontes_citadas, validar_resposta

PERGUNTA_TRIAGEM = "Qual a conduta inicial prevista para este caso?"


class EstadoTriagem(TypedDict, total=False):
    paciente_id: str
    trace_id: str
    paciente: dict | None
    pendentes: list[str]
    criticos: list[dict]
    alertas: list[str]
    trechos: list[tuple[str, str]]
    sugestao: str
    validacao: dict
    decisao: dict
    status: str


def criar_fluxo_triagem(
    conn: sqlite3.Connection, indice: InMemoryVectorStore, cadeia: Runnable, checkpointer
):
    ferramentas = criar_ferramentas(conn, indice)

    def carregar_prontuario(estado: EstadoTriagem) -> dict:
        paciente = ferramentas["consultar_prontuario"].invoke(
            {"paciente_id": estado["paciente_id"]}
        )
        return {
            "paciente": paciente,
            "status": "em_andamento" if paciente else "paciente_inexistente",
        }

    def verificar_pendencias(estado: EstadoTriagem) -> dict:
        pid = estado["paciente_id"]
        return {"pendentes": ferramentas["listar_exames_pendentes"].invoke({"paciente_id": pid})}

    def checar_valores_criticos(estado: EstadoTriagem) -> dict:
        pid = estado["paciente_id"]
        return {"criticos": ferramentas["verificar_valores_criticos"].invoke({"paciente_id": pid})}

    def emitir_alertas(estado: EstadoTriagem) -> dict:
        alertas = []
        for c in estado["criticos"]:
            mensagem = (
                f"VALOR CRÍTICO {estado['paciente_id']}: {c['exame']} {c['valor']} "
                f"(limite {c['limite']}). Comunicar o médico responsável (PROT-VCRIT-01 §4)."
            )
            registrar_alerta(conn, estado["paciente_id"], estado["trace_id"], mensagem)
            alertas.append(mensagem)
        return {"alertas": alertas}

    def buscar_protocolos(estado: EstadoTriagem) -> dict:
        consulta = f"{PERGUNTA_TRIAGEM} {estado['paciente']['diagnostico']}"
        return {"trechos": ferramentas["buscar_protocolos"].invoke({"consulta": consulta, "k": 3})}

    def gerar_sugestao(estado: EstadoTriagem) -> dict:
        contexto = Contexto(PERGUNTA_TRIAGEM, estado["paciente"], estado["trechos"])
        return {"sugestao": cadeia.invoke({"entrada": contexto.entrada()})}

    def validar_seguranca(estado: EstadoTriagem) -> dict:
        validacao = validar_resposta(estado["sugestao"], estado["paciente"])
        return {"validacao": {"aprovada": validacao.aprovada, "motivos": validacao.motivos}}

    def aprovacao_humana(estado: EstadoTriagem) -> dict:
        # Pausa o fluxo até o médico decidir; o estado fica salvo no checkpointer.
        decisao = interrupt(
            {
                "paciente_id": estado["paciente_id"],
                "sugestao": estado["sugestao"],
                "alertas": estado.get("alertas", []),
                "pendentes": estado["pendentes"],
            }
        )
        return {"decisao": decisao}

    def registrar(estado: EstadoTriagem) -> dict:
        if not estado["validacao"]["aprovada"]:
            status, medico, comentario = (
                "bloqueada",
                None,
                "; ".join(estado["validacao"]["motivos"]),
            )
        else:
            decisao = estado["decisao"]
            status = "aprovada" if decisao["aprovar"] else "rejeitada"
            medico, comentario = decisao["medico"], decisao.get("comentario")
        registrar_conduta(
            conn,
            estado["paciente_id"],
            estado["trace_id"],
            estado["sugestao"],
            fontes_citadas(estado["sugestao"]),
            status,
            medico,
            comentario,
        )
        return {"status": status}

    grafo = StateGraph(EstadoTriagem)
    for nome, no in [
        ("carregar_prontuario", carregar_prontuario),
        ("verificar_pendencias", verificar_pendencias),
        ("checar_valores_criticos", checar_valores_criticos),
        ("emitir_alertas", emitir_alertas),
        ("buscar_protocolos", buscar_protocolos),
        ("gerar_sugestao", gerar_sugestao),
        ("validar_seguranca", validar_seguranca),
        ("aprovacao_humana", aprovacao_humana),
        ("registrar_conduta", registrar),
    ]:
        grafo.add_node(nome, no)

    grafo.add_edge(START, "carregar_prontuario")
    grafo.add_conditional_edges(
        "carregar_prontuario",
        lambda e: "verificar_pendencias" if e["paciente"] else END,
        ["verificar_pendencias", END],
    )
    grafo.add_edge("verificar_pendencias", "checar_valores_criticos")
    grafo.add_conditional_edges(
        "checar_valores_criticos",
        lambda e: "emitir_alertas" if e["criticos"] else "buscar_protocolos",
        ["emitir_alertas", "buscar_protocolos"],
    )
    grafo.add_edge("emitir_alertas", "buscar_protocolos")
    grafo.add_edge("buscar_protocolos", "gerar_sugestao")
    grafo.add_edge("gerar_sugestao", "validar_seguranca")
    # Sugestão reprovada na validação não chega ao médico: é registrada como bloqueada.
    grafo.add_conditional_edges(
        "validar_seguranca",
        lambda e: "aprovacao_humana" if e["validacao"]["aprovada"] else "registrar_conduta",
        ["aprovacao_humana", "registrar_conduta"],
    )
    grafo.add_edge("aprovacao_humana", "registrar_conduta")
    grafo.add_edge("registrar_conduta", END)
    return grafo.compile(checkpointer=checkpointer)
