"""Normalização de texto aplicada antes da anonimização e da montagem do dataset."""

import re
import unicodedata

_UNIDADES = {
    r"\bmeq/l\b": "mEq/L",
    r"\bmg/dl\b": "mg/dL",
    r"\bmmol/l\b": "mmol/L",
    r"\bg/dl\b": "g/dL",
    r"\bml/kg/h\b": "mL/kg/h",
}
_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ESPACOS = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _CONTROLE.sub("", text)
    for padrao, unidade in _UNIDADES.items():
        text = re.sub(padrao, unidade, text, flags=re.IGNORECASE)
    text = "\n".join(_ESPACOS.sub(" ", linha).strip() for linha in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()
