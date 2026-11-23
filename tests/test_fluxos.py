import sqlite3

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from hospital_assistant.assistant.qa import criar_cadeia
from hospital_assistant.config import PROCESSED_DIR
from hospital_assistant.data.exames import EXAMES_LAB
from hospital_assistant.data.generate_patients import read_jsonl
from hospital_assistant.db.database import (
    buscar_paciente,
    conectar,
    listar_registros,
    popular,
    registrar_resultado,
)
from hospital_assistant.graph.duvida import criar_fluxo_duvida
from hospital_assistant.graph.triagem import criar_fluxo_triagem
from hospital_assistant.rag.indice import criar_indice
from hospital_assistant.safety.guardrails import validar_resposta

SUGESTAO_OK = "Seguir o protocolo.\n\nFontes: [PRONTUÁRIO: diagnóstico]; [PROT-CAD-01 §4]"
SUGESTAO_FONTE_FALSA = "Seguir o protocolo.\n\nFontes: [PROT-INVENTADO-02 §3]"


@pytest.fixture
def conn(tmp_path):
    conn = conectar(tmp_path / "teste.db")
    popular(conn, read_jsonl(PROCESSED_DIR / "prontuarios.jsonl"))
    return conn


@pytest.fixture(scope="module")
def indice():
    return criar_indice(DeterministicFakeEmbedding(size=64), path=None)


def _fluxo(conn, indice, resposta):
    cadeia = criar_cadeia(FakeListChatModel(responses=[resposta]))
    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    return criar_fluxo_triagem(conn, indice, cadeia, saver)


def _com_critico(conn) -> str:
    for i in range(1, 201):
        p = buscar_paciente(conn, f"PAC-{i:04d}")
        if any(
            "valor" in e and EXAMES_LAB[e["codigo"]].is_critico(e["valor"]) for e in p["exames"]
        ):
            return p["id"]
    raise AssertionError("nenhum paciente com valor crítico")


def test_triagem_pausa_para_aprovacao_e_registra_decisao(conn, indice):
    fluxo = _fluxo(conn, indice, SUGESTAO_OK)
    pid = _com_critico(conn)
    cfg = {"configurable": {"thread_id": "t1"}}

    resultado = fluxo.invoke({"paciente_id": pid, "trace_id": "t1"}, cfg)
    assert "__interrupt__" in resultado  # parou esperando o médico
    assert resultado["criticos"] and resultado["alertas"]
    assert fluxo.get_state(cfg).next == ("aprovacao_humana",)
    assert len(listar_registros(conn, "alertas", "t1")) == len(resultado["criticos"])
    assert listar_registros(conn, "condutas", "t1") == []  # nada vale antes da aprovação

    final = fluxo.invoke(Command(resume={"aprovar": True, "medico": "Dra. Teste CRM 123"}), cfg)
    assert final["status"] == "aprovada"
    conduta = listar_registros(conn, "condutas", "t1")[0]
    assert conduta["status"] == "aprovada" and conduta["medico"] == "Dra. Teste CRM 123"


def test_triagem_rejeitada_pelo_medico(conn, indice):
    fluxo = _fluxo(conn, indice, SUGESTAO_OK)
    cfg = {"configurable": {"thread_id": "t2"}}
    fluxo.invoke({"paciente_id": "PAC-0001", "trace_id": "t2"}, cfg)
    final = fluxo.invoke(
        Command(resume={"aprovar": False, "medico": "Dr. X", "comentario": "não se aplica"}), cfg
    )
    assert final["status"] == "rejeitada"
    assert listar_registros(conn, "condutas", "t2")[0]["comentario"] == "não se aplica"


def test_triagem_bloqueia_sugestao_com_fonte_inventada_sem_chegar_ao_medico(conn, indice):
    fluxo = _fluxo(conn, indice, SUGESTAO_FONTE_FALSA)
    cfg = {"configurable": {"thread_id": "t3"}}
    final = fluxo.invoke({"paciente_id": "PAC-0001", "trace_id": "t3"}, cfg)
    assert "__interrupt__" not in final
    assert final["status"] == "bloqueada"
    assert "fonte inexistente" in listar_registros(conn, "condutas", "t3")[0]["comentario"]


def test_triagem_paciente_inexistente(conn, indice):
    fluxo = _fluxo(conn, indice, SUGESTAO_OK)
    cfg = {"configurable": {"thread_id": "t4"}}
    final = fluxo.invoke({"paciente_id": "PAC-9999", "trace_id": "t4"}, cfg)
    assert final["status"] == "paciente_inexistente"


def test_novo_resultado_critico_do_laboratorio_dispara_alerta(conn, indice):
    """Contexto atualizado: um resultado liberado agora entra na próxima triagem."""
    pid = "PAC-0001"
    fluxo = _fluxo(conn, indice, SUGESTAO_OK)
    antes = fluxo.invoke(
        {"paciente_id": pid, "trace_id": "t5"}, {"configurable": {"thread_id": "t5"}}
    )
    registrar_resultado(conn, pid, "potassio", 6.9)
    fluxo = _fluxo(conn, indice, SUGESTAO_OK)
    depois = fluxo.invoke(
        {"paciente_id": pid, "trace_id": "t6"}, {"configurable": {"thread_id": "t6"}}
    )
    assert len(depois["criticos"]) == len(antes["criticos"]) + 1
    assert any("Potássio" in a for a in depois["alertas"])


def test_fluxo_duvida_responde_e_valida(conn, indice):
    cadeia = criar_cadeia(FakeListChatModel(responses=[SUGESTAO_OK]))
    final = criar_fluxo_duvida(conn, indice, cadeia).invoke(
        {"pergunta": "Conduta?", "paciente_id": "PAC-0005", "trace_id": "d1"}
    )
    assert final["resposta"] == SUGESTAO_OK
    assert final["paciente"]["id"] == "PAC-0005" and len(final["trechos"]) == 3
    assert final["validacao"]["aprovada"]


def test_guardrails():
    assert not validar_resposta("Texto sem fonte.").aprovada
    assert not validar_resposta(
        "Prescrevo ceftriaxona 1 g EV.\n\nFontes: [PROT-PAC-01 §4]"
    ).aprovada
    alergico = {"alergias": ["penicilina"]}
    assert not validar_resposta("Usar amoxicilina.\n\nFontes: [PROT-PAC-01 §4]", alergico).aprovada
    conflito = "Há conflito com a alergia a amoxicilina.\n\nFontes: [PRONTUÁRIO: alergias]"
    assert validar_resposta(conflito, alergico).aprovada
