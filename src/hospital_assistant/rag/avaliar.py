"""Avalia a busca nos protocolos com as perguntas frequentes (fontes corretas conhecidas).

Mede hit@1 (a fonte certa vem em 1º lugar) e hit@3 (aparece entre os 3 trechos entregues
ao modelo). Uso: uv run python -m hospital_assistant.rag.avaliar
"""

import json

from hospital_assistant.config import RAW_DIR, ROOT_DIR
from hospital_assistant.data.generate_patients import read_jsonl
from hospital_assistant.rag.indice import buscar, criar_indice

SAIDA = ROOT_DIR / "docs" / "avaliacao" / "busca.json"


def avaliar(indice, perguntas: list[tuple[str, set[str]]]) -> dict:
    hit1 = hit3 = 0
    erros = []
    for pergunta, esperadas in perguntas:
        refs = [ref for ref, _ in buscar(indice, pergunta, k=3)]
        hit1 += refs[0] in esperadas
        hit3 += bool(esperadas & set(refs))
        if refs[0] not in esperadas:
            erros.append({"pergunta": pergunta, "esperadas": sorted(esperadas), "obtidas": refs})
    n = len(perguntas)
    return {
        "perguntas": n,
        "hit@1": round(hit1 / n, 3),
        "hit@3": round(hit3 / n, 3),
        "erros": erros,
    }


if __name__ == "__main__":
    faq = read_jsonl(RAW_DIR / "faq.jsonl")
    perguntas = [(p, set(item["fontes"])) for item in faq for p in item["perguntas"]]
    resultado = avaliar(criar_indice(), perguntas)
    SAIDA.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"hit@1 {resultado['hit@1']:.1%} | hit@3 {resultado['hit@3']:.1%} -> {SAIDA}")
