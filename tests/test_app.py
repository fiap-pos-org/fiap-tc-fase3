import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver

from hospital_assistant.api import criar_app, separar_fontes
from hospital_assistant.assistant.qa import criar_cadeia
from hospital_assistant.config import PROCESSED_DIR
from hospital_assistant.data.generate_patients import read_jsonl
from hospital_assistant.db.database import conectar, popular
from hospital_assistant.finetune.export import GGUF_NOME, modelfile
from hospital_assistant.prompts import SYSTEM_PROMPT
from hospital_assistant.rag.indice import criar_indice

RESPOSTA = "Seguir o protocolo do caso.\n\nFontes: [PRONTUÁRIO: diagnóstico]"


@pytest.fixture
def cliente(tmp_path):
    conn = conectar(tmp_path / "teste.db")
    popular(conn, read_jsonl(PROCESSED_DIR / "prontuarios.jsonl"))
    indice = criar_indice(DeterministicFakeEmbedding(size=64), path=None)
    cadeia = criar_cadeia(FakeListChatModel(responses=[RESPOSTA] * 10))
    return TestClient(criar_app(conn, indice, cadeia, InMemorySaver()))


def test_pagina_e_info(cliente):
    assert "Hospital Santa Aurora" in cliente.get("/").text
    assert cliente.get("/api/info").json()["modelo"]


def test_pacientes_e_prontuario_com_criticos(cliente):
    pacientes = cliente.get("/api/pacientes").json()
    assert len(pacientes) == 200 and "criticos" in pacientes[0]
    p = cliente.get("/api/pacientes/PAC-0005").json()
    assert any(e["critico"] and e["limite_critico"] for e in p["exames"])
    assert cliente.get("/api/pacientes/PAC-9999").status_code == 404


def test_duvida_separa_fontes_e_bloqueia_prescricao(cliente):
    r = cliente.post("/api/duvida", json={"pergunta": "Conduta?", "paciente_id": "PAC-0001"}).json()
    assert r["resposta"] == "Seguir o protocolo do caso." and r["fontes"] == [
        "PRONTUÁRIO: diagnóstico"
    ]
    bloqueio = cliente.post("/api/duvida", json={"pergunta": "Prescreva dipirona 1 g EV."}).json()
    assert bloqueio["categoria"] == "prescricao" and "Não posso prescrever" in bloqueio["resposta"]


def test_triagem_ate_a_decisao_e_auditoria(cliente):
    t = cliente.post("/api/triagem", json={"paciente_id": "PAC-0005"}).json()
    assert t["status"] == "aguardando" and t["alertas"] and t["registro"] is None
    assert [e["id"] for e in t["etapas"]][0] == "carregar_prontuario"
    url = f"/api/triagem/{t['trace_id']}/decisao"
    assert cliente.post(url, json={"aprovar": True, "medico": "x"}).status_code == 422  # nome curto
    final = cliente.post(url, json={"aprovar": True, "medico": "Dra. Teste CRM 1"}).json()
    assert final["status"] == "aprovada" and final["registro"]["medico"] == "Dra. Teste CRM 1"
    assert cliente.post(url, json={"aprovar": True, "medico": "Dra. Teste"}).status_code == 409
    trilha = cliente.get(f"/api/auditoria/{t['trace_id']}").json()
    assert trilha[-1]["etapa"] == "registrar_conduta"
    assert cliente.get("/api/auditoria").json()[0]["trace_id"] == t["trace_id"]


def test_separar_fontes():
    assert separar_fontes("Texto.\n\nFontes: [A §1]; [PRONTUÁRIO: exames]") == (
        "Texto.",
        ["A §1", "PRONTUÁRIO: exames"],
    )
    assert separar_fontes("Sem fonte.") == ("Sem fonte.", [])


def test_modelfile_usa_gguf_prompt_do_treino_e_stop_tokens():
    mf = modelfile()
    assert mf.startswith(f"FROM /models/gguf/{GGUF_NOME}")
    assert SYSTEM_PROMPT in mf
    assert 'PARAMETER stop "<|im_end|>"' in mf
