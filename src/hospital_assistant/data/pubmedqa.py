"""Amostra estratificada do PubMedQA (PQA-L, rotulado por especialistas) para avaliação.

Usado só na avaliação, nunca no treino: mede conhecimento biomédico geral (yes/no/maybe)
e se o fine-tuning degradou o modelo base (esquecimento catastrófico).
Fonte: https://github.com/pubmedqa/pubmedqa (licença MIT).

Uso: uv run python -m hospital_assistant.data.pubmedqa
"""

import json
import random
import urllib.request
from collections import defaultdict

from hospital_assistant.config import EVAL_DIR
from hospital_assistant.data.generate_patients import write_jsonl

URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"


def sample_pubmedqa(dados: dict, n: int = 200, seed: int = 42) -> list[dict]:
    por_rotulo: dict[str, list[str]] = defaultdict(list)
    for pmid, item in sorted(dados.items()):
        por_rotulo[item["final_decision"]].append(pmid)

    rng = random.Random(seed)
    escolhidos = []
    for pmids in por_rotulo.values():
        k = round(n * len(pmids) / len(dados))
        escolhidos += rng.sample(pmids, k)
    return [
        {
            "id": f"PMQA-{pmid}",
            "pergunta": dados[pmid]["QUESTION"],
            "contexto": " ".join(dados[pmid]["CONTEXTS"]),
            "resposta": dados[pmid]["final_decision"],
        }
        for pmid in sorted(escolhidos)
    ]


if __name__ == "__main__":
    with urllib.request.urlopen(URL, timeout=60) as resp:
        dados = json.load(resp)
    amostra = sample_pubmedqa(dados)
    write_jsonl(amostra, EVAL_DIR / "pubmedqa_200.jsonl")
    print(f"{len(amostra)} exemplos salvos em {EVAL_DIR / 'pubmedqa_200.jsonl'}")
