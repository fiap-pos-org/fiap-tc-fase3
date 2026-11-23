import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from hospital_assistant.assistant.qa import montar_contexto, responder
from hospital_assistant.config import PROCESSED_DIR
from hospital_assistant.data.documents import load_documents
from hospital_assistant.data.generate_patients import read_jsonl
from hospital_assistant.db.database import buscar_paciente, conectar, listar_pacientes, popular
from hospital_assistant.prompts import format_patient_context
from hospital_assistant.rag.indice import buscar, criar_indice, montar_trechos


@pytest.fixture(scope="module")
def prontuarios():
    return read_jsonl(PROCESSED_DIR / "prontuarios.jsonl")


@pytest.fixture(scope="module")
def conn(prontuarios, tmp_path_factory):
    conn = conectar(tmp_path_factory.mktemp("db") / "teste.db")
    popular(conn, prontuarios)
    return conn


@pytest.fixture(scope="module")
def indice():
    return criar_indice(DeterministicFakeEmbedding(size=64), path=None)


def test_banco_tem_todos_os_pacientes(conn, prontuarios):
    assert len(listar_pacientes(conn)) == len(prontuarios)
    assert buscar_paciente(conn, "PAC-9999") is None


def test_prontuario_do_banco_gera_o_mesmo_contexto_do_treino(conn, prontuarios):
    """O texto entregue ao modelo na inferência tem que ser igual ao do treino."""
    for original in prontuarios[:20]:
        do_banco = buscar_paciente(conn, original["id"])
        assert format_patient_context(do_banco) == format_patient_context(original)


def test_trechos_sem_secao_1_e_com_referencia():
    trechos = montar_trechos(load_documents())
    refs = [t.metadata["ref"] for t in trechos]
    assert len(trechos) == 17 * 5  # 17 documentos, seções 2 a 6
    assert not any(r.endswith("§1") for r in refs)
    assert "PROT-SEPSE-01 §4" in refs


def test_busca_retorna_k_trechos_com_referencia(indice):
    resultado = buscar(indice, "antibiótico na sepse", k=3)
    assert len(resultado) == 3
    assert all(" §" in ref and texto for ref, texto in resultado)


def test_contexto_com_paciente_inclui_prontuario_e_trechos(conn, indice):
    contexto = montar_contexto("Há valor crítico?", indice, conn, paciente_id="PAC-0001")
    assert contexto.paciente["id"] == "PAC-0001"
    assert len(contexto.trechos) == 3
    sistema, usuario = contexto.mensagens()
    assert sistema[0] == "system"
    assert "[PRONTUÁRIO PAC-0001]" in usuario[1]
    assert "Trechos de protocolos:" in usuario[1]
    assert usuario[1].endswith("Pergunta: Há valor crítico?")


def test_responder_acumula_o_streaming(conn, indice):
    contexto = montar_contexto("Pergunta geral", indice, conn)
    llm = FakeListChatModel(responses=["Resposta.\n\nFontes: [PROT-SEPSE-01 §4]"])
    partes = list(responder(contexto, llm))
    assert partes[-1] == "Resposta.\n\nFontes: [PROT-SEPSE-01 §4]"
