"""Anonimização de prontuários em três camadas.

1. Identificadores conhecidos do próprio registro (nome, acompanhante, endereço).
2. Expressões regulares para documentos e contatos (CPF, CNS, telefone, e-mail, CEP).
3. NER do spaCy (pt_core_news_md) para nomes e locais que sobraram no texto livre.

Campos estruturados identificadores são descartados; a data de nascimento vira idade.
O mapeamento pseudônimo -> pessoa NÃO é salvo, para impedir a reidentificação.
"""

import re
from collections import Counter
from datetime import date
from functools import cache

import spacy

from hospital_assistant.data.preprocess import normalize_text

PII_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "CPF": re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b"),
    "CNS": re.compile(r"\b[1-9]\d{2} \d{4} \d{4} \d{4}\b|\b[1-9]\d{14}\b"),
    "TELEFONE": re.compile(r"(?:\+55\s?)?(?:\(\d{2}\)|\b\d{2})\s?9?\d{4}-\d{4}\b"),
    "CEP": re.compile(r"\b\d{5}-\d{3}\b"),
}
# Rótulos do NER que tratamos como dado pessoal.
_NER_LABELS = {"PER": "NOME", "LOC": "LOCAL"}
# Formato de nome próprio: palavras capitalizadas, com conectivos (da, de, do, dos...).
_NOME_PROPRIO = re.compile(
    r"^(?:(?:Sr|Sra|Srta|Dr|Dra)\.? )?[A-ZÀ-Ý][a-zà-ÿ'-]+(?: (?:d[aeo]s?|[A-ZÀ-Ý][a-zà-ÿ'-]+))*$"
)

CAMPOS_DESCARTADOS = ("atendimento", "nome", "cpf", "cns", "telefone", "email", "endereco")


@cache
def _nlp():
    return spacy.load("pt_core_news_md", disable=["parser", "lemmatizer"])


def _substitui(texto: str, alvo: str, marcador: str, stats: Counter, tipo: str) -> str:
    if not alvo:
        return texto
    novo, n = re.subn(re.escape(alvo), marcador, texto)
    stats[tipo] += n
    return novo


def anonymize_text(
    texto: str, conhecidos: dict[str, str] | None = None, stats: Counter | None = None
) -> str:
    """Anonimiza texto livre. `conhecidos` mapeia valor real -> marcador (ex.: nome -> PAC-0001)."""
    stats = stats if stats is not None else Counter()
    texto = normalize_text(texto)

    # Camada 1: identificadores conhecidos, do mais longo para o mais curto.
    for valor, marcador in sorted((conhecidos or {}).items(), key=lambda kv: -len(kv[0])):
        texto = _substitui(texto, valor, marcador, stats, "CONHECIDO")

    # Camada 2: padrões de documentos e contatos.
    for tipo, padrao in PII_PATTERNS.items():
        texto, n = padrao.subn(f"[{tipo}]", texto)
        stats[tipo] += n

    # Camada 3: NER para nomes e locais residuais.
    doc = _nlp()(texto)
    for ent in reversed(doc.ents):
        tipo = _NER_LABELS.get(ent.label_)
        if tipo is None or not _NOME_PROPRIO.match(ent.text):
            continue
        # Palavra única no início de frase costuma ser falso positivo ("Hipótese", "Familiar");
        # nomes reais nessa posição já foram pegos pela camada 1.
        inicio_de_frase = texto[: ent.start_char].rstrip()[-1:] in {"", ".", ":"}
        if len(ent) == 1 and inicio_de_frase:
            continue
        texto = texto[: ent.start_char] + f"[{tipo}]" + texto[ent.end_char :]
        stats[f"NER_{tipo}"] += 1
    return texto


def _idade(nascimento: str, referencia: str) -> int:
    nasc = date.fromisoformat(nascimento)
    ref = date.fromisoformat(referencia[:10])
    return ref.year - nasc.year - ((ref.month, ref.day) < (nasc.month, nasc.day))


def anonymize_patient(registro: dict, pseudonimo: str, stats: Counter) -> dict:
    nome = registro["nome"]
    acompanhante = registro["nota_admissao"].split(" por ", 1)[1].split(",", 1)[0].split(" ", 1)[1]
    conhecidos = {
        nome: pseudonimo,
        nome.split()[0]: pseudonimo,
        acompanhante: "[ACOMPANHANTE]",
        registro["endereco"]: "[ENDERECO]",
        registro["endereco"].split(", ")[0]: "[ENDERECO]",
        registro["endereco"].split(", ")[-1]: "[CIDADE]",
    }
    anon = {k: v for k, v in registro.items() if k not in CAMPOS_DESCARTADOS}
    anon.pop("data_nascimento")
    anon = {
        "id": pseudonimo,
        "idade": _idade(registro["data_nascimento"], registro["data_internacao"]),
        **anon,
    }
    anon["nota_admissao"] = anonymize_text(registro["nota_admissao"], conhecidos, stats)
    anon["evolucao"] = anonymize_text(registro["evolucao"], conhecidos, stats)
    stats["CAMPOS_DESCARTADOS"] += len(CAMPOS_DESCARTADOS) + 1  # +1: data de nascimento
    return anon


def anonymize_patients(registros: list[dict]) -> tuple[list[dict], Counter]:
    stats: Counter = Counter()
    anon = [anonymize_patient(r, f"PAC-{i + 1:04d}", stats) for i, r in enumerate(registros)]
    return anon, stats


def find_pii(texto: str) -> list[str]:
    """Retorna os tipos de dado pessoal detectados por regex (usado na curadoria e nos testes)."""
    return [tipo for tipo, padrao in PII_PATTERNS.items() if padrao.search(texto)]
