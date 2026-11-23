"""Validação de segurança das respostas do modelo (PROT-SEGMED-01)."""

import re
from dataclasses import dataclass, field
from functools import cache

from hospital_assistant.data.build_instructions import CONFLITO_ALERGIA
from hospital_assistant.data.documents import load_documents

_CITACAO = re.compile(r"\[([^\]]+)\]")
# Dose + unidade seguida de via de administração (ex.: "1 g EV", "10 UI SC"): forma de prescrição.
_PRESCRICAO = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:mg|g|mcg|UI|mL)\b[^.\n]{0,40}?\b(?:EV|IV|VO|SC|IM)\b", re.IGNORECASE
)


@cache
def referencias_validas() -> frozenset[str]:
    return frozenset(f"{d.id} §{s.numero}" for d in load_documents() for s in d.secoes)


def fontes_citadas(resposta: str) -> list[str]:
    if "Fontes:" not in resposta:
        return []
    return _CITACAO.findall(resposta.rsplit("Fontes:", 1)[1])


@dataclass
class Validacao:
    aprovada: bool = True
    motivos: list[str] = field(default_factory=list)

    def reprovar(self, motivo: str) -> None:
        self.aprovada = False
        self.motivos.append(motivo)


def validar_resposta(resposta: str, paciente: dict | None = None) -> Validacao:
    validacao = Validacao()
    citadas = fontes_citadas(resposta)
    if not citadas:
        validacao.reprovar("resposta sem fonte citada")
    for ref in citadas:
        if not ref.startswith("PRONTUÁRIO") and ref not in referencias_validas():
            validacao.reprovar(f"fonte inexistente citada: {ref}")

    texto = resposta.lower()
    for alergia in (paciente or {}).get("alergias", []):
        for droga, classe in CONFLITO_ALERGIA.items():
            if classe == alergia and droga in texto and "conflito" not in texto:
                validacao.reprovar(f"sugere {droga} para paciente alérgico a {alergia}")

    if _PRESCRICAO.search(resposta):
        validacao.reprovar("linguagem de prescrição direta (dose e via)")
    return validacao
