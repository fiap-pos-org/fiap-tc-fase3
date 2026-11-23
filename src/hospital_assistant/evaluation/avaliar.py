"""Avaliação: modelo original (Qwen2.5-1.5B-Instruct) x fine-tuning v1 x fine-tuning v2.

Os dois rodam no mesmo Ollama, com a mesma quantização (Q4_K_M), o mesmo prompt e
temperatura 0 (reprodutível). As respostas ficam em docs/avaliacao/respostas_*.jsonl e a
execução retoma de onde parou; o resumo das métricas vai para docs/avaliacao/metricas.json.

Uso: uv run python -m hospital_assistant.evaluation.avaliar [teste|pubmedqa|ataques|resumo|tudo]
"""

import json
import statistics
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings

from hospital_assistant.assistant.qa import criar_cadeia
from hospital_assistant.config import (
    EMBEDDING_MODEL,
    EVAL_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_THREAD,
    PROCESSED_DIR,
    ROOT_DIR,
)
from hospital_assistant.data.generate_patients import read_jsonl
from hospital_assistant.evaluation.metricas import (
    acerto_fontes,
    resposta_segura,
    rotulo_pubmedqa,
    rouge_l,
)
from hospital_assistant.prompts import SYSTEM_PROMPT, format_user_message

MODELOS = {
    "original": "qwen2.5:1.5b-instruct",
    "v1": "hospital-assistant-v1",  # fine-tuning v1
    "v2": "hospital-assistant",  # fine-tuning v2 (recusas com trechos irrelevantes)
}
# Ordem de apresentação; "sistema" = travas + modelo v2, "sistema_v1" = travas + modelo v1.
ORDEM = ["original", "v1", "v2", "sistema_v1", "sistema"]
SAIDA = ROOT_DIR / "docs" / "avaliacao"

SYSTEM_PUBMEDQA = (
    "You are a biomedical research assistant. Read the context and answer the question with a "
    "single word: yes, no or maybe."
)


def _llm(modelo: str, num_predict: int = 512) -> ChatOllama:
    return ChatOllama(
        model=modelo,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_thread=OLLAMA_NUM_THREAD,
        num_predict=num_predict,
    )


def _gerar(llm: ChatOllama, system: str, user: str) -> tuple[str, float]:
    inicio = time.perf_counter()
    texto = llm.invoke([("system", system), ("human", user)]).content
    return texto, round(time.perf_counter() - inicio, 2)


def _executar(arquivo: Path, itens: list[dict], gerar) -> None:
    """Roda `gerar(item)` para cada item ainda não salvo em `arquivo` (retomável)."""
    feitos = {(r["id"], r["modelo"]) for r in read_jsonl(arquivo)} if arquivo.exists() else set()
    pendentes = [i for i in itens if (i["id"], i["modelo"]) not in feitos]
    with arquivo.open("a", encoding="utf-8") as f:
        for n, item in enumerate(pendentes, 1):
            linha = gerar(item)
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{arquivo.name}: {n}/{len(pendentes)} {item['modelo']} {item['id']}", flush=True)


def avaliar_teste() -> None:
    exemplos = read_jsonl(PROCESSED_DIR / "test.jsonl")
    llms = {nome: _llm(modelo) for nome, modelo in MODELOS.items()}

    def gerar(item):
        ex = item["exemplo"]
        resposta, segundos = _gerar(
            llms[item["modelo"]], SYSTEM_PROMPT, ex["messages"][1]["content"]
        )
        return {
            "id": ex["id"],
            "modelo": item["modelo"],
            "tipo": ex["tipo"],
            "resposta": resposta,
            "referencia": ex["messages"][2]["content"],
            "segundos": segundos,
        }

    itens = [{"id": ex["id"], "modelo": m, "exemplo": ex} for m in MODELOS for ex in exemplos]
    _executar(SAIDA / "respostas_teste.jsonl", itens, gerar)


def avaliar_pubmedqa() -> None:
    perguntas = read_jsonl(EVAL_DIR / "pubmedqa_200.jsonl")
    llms = {nome: _llm(modelo, num_predict=8) for nome, modelo in MODELOS.items()}

    def gerar(item):
        q = item["q"]
        user = f"Context: {q['contexto']}\n\nQuestion: {q['pergunta']}\nAnswer (yes, no or maybe):"
        resposta, segundos = _gerar(llms[item["modelo"]], SYSTEM_PUBMEDQA, user)
        return {
            "id": q["id"],
            "modelo": item["modelo"],
            "resposta": resposta,
            "rotulo": rotulo_pubmedqa(resposta),
            "esperado": q["resposta"],
            "segundos": segundos,
        }

    itens = [{"id": q["id"], "modelo": m, "q": q} for m in MODELOS for q in perguntas]
    _executar(SAIDA / "respostas_pubmedqa.jsonl", itens, gerar)


def avaliar_ataques() -> None:
    """Só o modelo (original e ajustado) e o sistema completo (travas + modelo ajustado)."""
    from hospital_assistant.db.database import inicializar
    from hospital_assistant.graph.duvida import criar_fluxo_duvida
    from hospital_assistant.rag.indice import criar_indice

    ataques = read_jsonl(EVAL_DIR / "ataques.jsonl")
    llms = {nome: _llm(modelo) for nome, modelo in MODELOS.items()}
    fluxo = criar_fluxo_duvida(inicializar(), criar_indice(), criar_cadeia(llms["v2"]))

    def gerar(item):
        a = item["ataque"]
        if item["modelo"] == "sistema":
            inicio = time.perf_counter()
            estado = fluxo.invoke(
                {"pergunta": a["pergunta"], "trace_id": f"aval-{uuid.uuid4().hex[:8]}"}
            )
            resposta, segundos = estado["resposta"], round(time.perf_counter() - inicio, 2)
            etapa = (
                estado["categoria"]
                if estado["categoria"] != "clinica"
                else ("validacao" if not estado["validacao"]["aprovada"] else "modelo")
            )
        else:
            user = format_user_message(a["pergunta"])
            resposta, segundos = _gerar(llms[item["modelo"]], SYSTEM_PROMPT, user)
            etapa = "modelo"
        return {
            "id": a["id"],
            "modelo": item["modelo"],
            "categoria": a["categoria"],
            "resposta": resposta,
            "segura": resposta_segura(resposta),
            "barrado_em": etapa,
            "segundos": segundos,
        }

    itens = [
        {"id": a["id"], "modelo": m, "ataque": a} for m in [*MODELOS, "sistema"] for a in ataques
    ]
    _executar(SAIDA / "respostas_ataques.jsonl", itens, gerar)


def _media(valores: list[float]) -> float:
    return round(statistics.fmean(valores), 3) if valores else 0.0


def resumir() -> dict:
    resumo: dict = {"modelos": MODELOS, "temperatura": 0}

    # Só os exemplos do conjunto de teste atual (respostas de versões antigas ficam no arquivo).
    ids_teste = {ex["id"] for ex in read_jsonl(PROCESSED_DIR / "test.jsonl")}
    teste = [r for r in read_jsonl(SAIDA / "respostas_teste.jsonl") if r["id"] in ids_teste]
    emb = OllamaEmbeddings(
        model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL, num_thread=OLLAMA_NUM_THREAD
    )
    vetores = emb.embed_documents([r["resposta"] for r in teste] + [r["referencia"] for r in teste])
    n = len(teste)
    for i, r in enumerate(teste):
        a, b = vetores[i], vetores[n + i]
        r["semantica"] = sum(x * y for x, y in zip(a, b, strict=True)) / (
            sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5
        )
    por = defaultdict(list)
    for r in teste:
        por[(r["modelo"], "todos")].append(r)
        por[(r["modelo"], r["tipo"])].append(r)
    resumo["teste"] = {}
    for (modelo, tipo), linhas in sorted(por.items()):
        fontes = [acerto_fontes(r["resposta"], r["referencia"]) for r in linhas]
        resumo["teste"].setdefault(modelo, {})[tipo] = {
            "n": len(linhas),
            "rouge_l": _media([rouge_l(r["resposta"], r["referencia"]) for r in linhas]),
            "similaridade_semantica": _media([r["semantica"] for r in linhas]),
            "fontes_precisao": _media([f["precisao"] for f in fontes]),
            "fontes_revocacao": _media([f["revocacao"] for f in fontes]),
            "fontes_exatas": _media([f["exata"] for f in fontes]),
            "com_linha_fontes": _media(["Fontes:" in r["resposta"] for r in linhas]),
            "segundos": _media([r["segundos"] for r in linhas]),
        }

    pubmed = read_jsonl(SAIDA / "respostas_pubmedqa.jsonl")
    resumo["pubmedqa"] = {}
    for modelo in [m for m in ORDEM if any(r["modelo"] == m for r in pubmed)]:
        linhas = [r for r in pubmed if r["modelo"] == modelo]
        resumo["pubmedqa"][modelo] = {
            "n": len(linhas),
            "acuracia": _media([r["rotulo"] == r["esperado"] for r in linhas]),
            "sem_rotulo": sum(r["rotulo"] is None for r in linhas),
            "distribuicao": {
                k: sum(r["rotulo"] == k for r in linhas) for k in ("yes", "no", "maybe")
            },
        }

    ataques = read_jsonl(SAIDA / "respostas_ataques.jsonl")
    resumo["ataques"] = {}
    for modelo in [m for m in ORDEM if any(r["modelo"] == m for r in ataques)]:
        linhas = [r for r in ataques if r["modelo"] == modelo]
        categorias = sorted({r["categoria"] for r in linhas})
        resumo["ataques"][modelo] = {
            "seguras": _media([r["segura"] for r in linhas]),
            "por_categoria": {
                c: _media([r["segura"] for r in linhas if r["categoria"] == c]) for c in categorias
            },
            "barrado_em": {
                e: sum(r["barrado_em"] == e for r in linhas)
                for e in sorted({r["barrado_em"] for r in linhas})
            },
        }

    busca = SAIDA / "busca.json"
    if busca.exists():
        b = json.loads(busca.read_text())
        resumo["busca"] = {"hit@1": b["hit@1"], "hit@3": b["hit@3"], "perguntas": b["perguntas"]}
    (SAIDA / "metricas.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n")
    return resumo


if __name__ == "__main__":
    etapa = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    SAIDA.mkdir(parents=True, exist_ok=True)
    if etapa in ("teste", "tudo"):
        avaliar_teste()
    if etapa in ("ataques", "tudo"):
        avaliar_ataques()
    if etapa in ("pubmedqa", "tudo"):
        avaliar_pubmedqa()
    if etapa in ("resumo", "tudo"):
        print(json.dumps(resumir(), ensure_ascii=False, indent=2))
