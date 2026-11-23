"""Pipeline de dados completo: gera, pré-processa, anonimiza, monta, cura e divide.

Uso: uv run python -m hospital_assistant.data.pipeline
"""

import json
from collections import Counter

from hospital_assistant.config import PROCESSED_DIR, RAW_DIR
from hospital_assistant.data.anonymize import anonymize_patients
from hospital_assistant.data.build_instructions import build_all
from hospital_assistant.data.curate import curate, split_por_grupo
from hospital_assistant.data.documents import load_documents
from hospital_assistant.data.generate_patients import generate_patients, read_jsonl, write_jsonl

SEED = 42


def run() -> dict:
    docs = load_documents()
    faq = read_jsonl(RAW_DIR / "faq.jsonl")

    brutos = generate_patients(seed=SEED)
    write_jsonl(brutos, RAW_DIR / "prontuarios.jsonl")

    anonimizados, stats_anon = anonymize_patients(brutos)
    write_jsonl(anonimizados, PROCESSED_DIR / "prontuarios.jsonl")

    exemplos = build_all(docs, faq, anonimizados, seed=SEED)
    refs_validas = {f"{d.id} §{s.numero}" for d in docs for s in d.secoes}
    nomes_reais = [p["nome"] for p in brutos]
    aprovados, descartes = curate(exemplos, refs_validas, nomes_reais)

    splits = split_por_grupo(aprovados, seed=SEED)
    for nome, linhas in splits.items():
        write_jsonl(linhas, PROCESSED_DIR / f"{nome}.jsonl")

    relatorio = {
        "documentos": {
            "protocolos": sum(d.tipo == "protocolo" for d in docs),
            "modelos": sum(d.tipo == "modelo" for d in docs),
        },
        "faq": len(faq),
        "prontuarios": len(brutos),
        "anonimizacao": dict(stats_anon),
        "exemplos_gerados": len(exemplos),
        "curadoria_descartes": dict(descartes),
        "exemplos_aprovados": len(aprovados),
        "splits": {
            nome: dict(Counter(ex["tipo"] for ex in linhas)) | {"total": len(linhas)}
            for nome, linhas in splits.items()
        },
    }
    (PROCESSED_DIR / "relatorio_dados.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return relatorio


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
