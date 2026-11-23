"""Testes sobre os arquivos gerados pelo pipeline (data/processed e data/eval)."""

import re

import pytest

from hospital_assistant.config import EVAL_DIR, PROCESSED_DIR, RAW_DIR
from hospital_assistant.data.anonymize import find_pii
from hospital_assistant.data.documents import load_documents
from hospital_assistant.data.generate_patients import read_jsonl

SPLITS = ("train", "val", "test")


@pytest.fixture(scope="module")
def splits():
    return {nome: read_jsonl(PROCESSED_DIR / f"{nome}.jsonl") for nome in SPLITS}


@pytest.fixture(scope="module")
def nomes_reais():
    brutos = read_jsonl(RAW_DIR / "prontuarios.jsonl")
    return {p["nome"] for p in brutos} | {p["cpf"] for p in brutos}


def test_documentos_internos_tem_seis_secoes():
    docs = load_documents()
    assert len(docs) == 17
    assert all([s.numero for s in d.secoes] == [1, 2, 3, 4, 5, 6] for d in docs)


def test_dataset_final_nao_tem_dado_pessoal(splits, nomes_reais):
    for linhas in splits.values():
        for ex in linhas:
            texto = " ".join(m["content"] for m in ex["messages"])
            assert not find_pii(texto), ex["id"]
            assert not any(n in texto for n in nomes_reais), ex["id"]


def test_prontuarios_processados_nao_tem_dado_pessoal(nomes_reais):
    for p in read_jsonl(PROCESSED_DIR / "prontuarios.jsonl"):
        texto = str(p)
        assert not find_pii(texto), p["id"]
        assert not any(n in texto for n in nomes_reais), p["id"]


def test_sem_vazamento_de_grupo_entre_splits(splits):
    grupos = {nome: {ex["grupo"] for ex in linhas} for nome, linhas in splits.items()}
    assert not grupos["train"] & grupos["val"]
    assert not grupos["train"] & grupos["test"]
    assert not grupos["val"] & grupos["test"]


def test_toda_resposta_termina_com_fontes_validas(splits):
    refs = {f"{d.id} §{s.numero}" for d in load_documents() for s in d.secoes}
    for linhas in splits.values():
        for ex in linhas:
            resposta = ex["messages"][-1]["content"]
            ultima_linha = resposta.splitlines()[-1]
            assert ultima_linha.startswith("Fontes: "), ex["id"]
            for citada in re.findall(r"\[([^\]]+)\]", ultima_linha):
                assert citada in refs or citada.startswith("PRONTUÁRIO:"), (ex["id"], citada)


def test_conjuntos_de_avaliacao_nao_aparecem_no_treino(splits):
    perguntas_treino = " ".join(ex["messages"][1]["content"] for ex in splits["train"])
    for item in read_jsonl(EVAL_DIR / "ataques.jsonl"):
        assert item["pergunta"] not in perguntas_treino, item["id"]
    for item in read_jsonl(EVAL_DIR / "pubmedqa_200.jsonl"):
        assert item["pergunta"] not in perguntas_treino, item["id"]
