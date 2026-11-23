"""Curadoria dos exemplos de instrução e divisão em treino, validação e teste."""

import random
import re
from collections import Counter, defaultdict

from hospital_assistant.data.anonymize import find_pii
from hospital_assistant.data.preprocess import normalize_text

MIN_RESPOSTA, MAX_RESPOSTA = 40, 2500
MAX_PERGUNTA = 4000
_FONTES_RE = re.compile(r"\[([^\]]+)\]")


def _chave_dedup(exemplo: dict) -> str:
    texto = " ".join(m["content"] for m in exemplo["messages"][1:]).lower()
    return re.sub(r"\W+", " ", texto).strip()


def _fontes_validas(resposta: str, refs_validas: set[str]) -> bool:
    linha = resposta.rsplit("Fontes:", 1)
    if len(linha) != 2:
        return False
    citadas = _FONTES_RE.findall(linha[1])
    return bool(citadas) and all(c in refs_validas or c.startswith("PRONTUÁRIO:") for c in citadas)


def curate(
    exemplos: list[dict], refs_validas: set[str], nomes_reais: list[str]
) -> tuple[list[dict], Counter]:
    """Normaliza e filtra. Retorna os exemplos aprovados e a contagem de descartes por motivo."""
    descartes: Counter = Counter()
    vistos: set[str] = set()
    aprovados = []
    for ex in exemplos:
        for m in ex["messages"]:
            m["content"] = normalize_text(m["content"])
        user, resposta = ex["messages"][1]["content"], ex["messages"][2]["content"]
        texto = f"{user}\n{resposta}"

        chave = _chave_dedup(ex)
        if chave in vistos:
            descartes["duplicado"] += 1
            continue
        if not MIN_RESPOSTA <= len(resposta) <= MAX_RESPOSTA or len(user) > MAX_PERGUNTA:
            descartes["tamanho"] += 1
            continue
        if not _fontes_validas(resposta, refs_validas):
            descartes["fonte_invalida"] += 1
            continue
        if find_pii(texto) or any(nome in texto for nome in nomes_reais):
            descartes["dado_pessoal"] += 1
            continue
        vistos.add(chave)
        aprovados.append(ex)
    return aprovados, descartes


def split_por_grupo(
    exemplos: list[dict], seed: int = 42, proporcoes: tuple[float, float] = (0.8, 0.1)
) -> dict[str, list[dict]]:
    """Divide 80/10/10 por grupo, estratificado por tipo: um grupo nunca aparece em dois splits."""
    rng = random.Random(seed)
    grupos_por_tipo: dict[str, list[str]] = defaultdict(list)
    for ex in exemplos:
        if ex["grupo"] not in grupos_por_tipo[ex["tipo"]]:
            grupos_por_tipo[ex["tipo"]].append(ex["grupo"])

    destino: dict[str, str] = {}
    for grupos in grupos_por_tipo.values():
        rng.shuffle(grupos)
        n_treino = round(len(grupos) * proporcoes[0])
        n_val = round(len(grupos) * proporcoes[1])
        for i, g in enumerate(grupos):
            destino[g] = "train" if i < n_treino else "val" if i < n_treino + n_val else "test"

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for ex in exemplos:
        splits[destino[ex["grupo"]]].append(ex)
    return splits
