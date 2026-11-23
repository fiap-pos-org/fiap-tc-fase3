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
    assert len(pacientes) == 200 and {"criticos", "pendentes"} <= pacientes[0].keys()
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
    assert [e["id"] for e in t["etapas"]][:2] == ["requisicao", "carregar_prontuario"]
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


@pytest.fixture
def cliente_com_login(tmp_path):
    conn = conectar(tmp_path / "teste.db")
    popular(conn, read_jsonl(PROCESSED_DIR / "prontuarios.jsonl"))
    indice = criar_indice(DeterministicFakeEmbedding(size=64), path=None)
    cadeia = criar_cadeia(FakeListChatModel(responses=[RESPOSTA] * 5))
    app = criar_app(conn, indice, cadeia, InMemorySaver(), usuario="medico", senha="s3nha-forte")
    return TestClient(app)


def test_login_protege_a_api_e_a_pagina(cliente_com_login):
    c = cliente_com_login
    assert c.get("/api/pacientes").status_code == 401
    assert "Acesso restrito" in c.get("/").text  # sem sessão, a raiz mostra o login
    assert c.get("/api/info").status_code == 200  # informação pública
    assert c.post("/api/login", json={"usuario": "medico", "senha": "errada"}).status_code == 401
    assert (
        c.post("/api/login", json={"usuario": "medico", "senha": "s3nha-forte"}).status_code == 200
    )
    assert c.get("/api/pacientes").status_code == 200
    assert "Dúvida clínica" in c.get("/").text
    assert c.get("/api/sessao").json() == {"login_ativo": True, "usuario": "medico"}


def test_cookie_adulterado_nao_autentica(cliente_com_login):
    c = cliente_com_login
    c.post("/api/login", json={"usuario": "medico", "senha": "s3nha-forte"})
    nome, expira, assinatura = c.cookies.get("sessao").split("|")
    c.cookies.clear()
    c.cookies.set("sessao", f"admin|{expira}|{assinatura}")
    assert c.get("/api/pacientes").status_code == 401


def test_auditoria_registra_quem_perguntou(cliente_com_login):
    c = cliente_com_login
    c.post("/api/login", json={"usuario": "medico", "senha": "s3nha-forte"})
    r = c.post("/api/duvida", json={"pergunta": "Conduta?", "paciente_id": "PAC-0001"}).json()
    primeira = c.get(f"/api/auditoria/{r['trace_id']}").json()[0]
    assert primeira["etapa"] == "requisicao"
    assert primeira["dados"] == {"usuario": "medico", "pergunta": "Conduta?"}
