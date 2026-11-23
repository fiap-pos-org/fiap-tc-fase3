"""Leitura dos documentos internos (protocolos e modelos) em Markdown com front-matter."""

import re
from dataclasses import dataclass
from pathlib import Path

from hospital_assistant.config import RAW_DIR

_SECTION_RE = re.compile(r"^## (\d+)\. (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    numero: int
    titulo: str
    texto: str


@dataclass(frozen=True)
class Document:
    id: str
    titulo: str
    versao: str
    vigencia: str
    setor: str
    tipo: str  # "protocolo" ou "modelo"
    secoes: tuple[Section, ...]

    def secao(self, numero: int) -> Section:
        return next(s for s in self.secoes if s.numero == numero)


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    _, header, body = text.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    return meta, body


def parse_document(path: Path, tipo: str) -> Document:
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
    matches = list(_SECTION_RE.finditer(body))
    secoes = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        texto = body[m.end() : end].strip()
        secoes.append(Section(int(m.group(1)), m.group(2).strip(), texto))
    return Document(
        id=meta["id"],
        titulo=meta["titulo"],
        versao=meta["versao"],
        vigencia=meta["vigencia"],
        setor=meta["setor"],
        tipo=tipo,
        secoes=tuple(secoes),
    )


def load_documents(raw_dir: Path = RAW_DIR) -> list[Document]:
    docs = [parse_document(p, "protocolo") for p in sorted((raw_dir / "protocolos").glob("*.md"))]
    docs += [parse_document(p, "modelo") for p in sorted((raw_dir / "modelos").glob("*.md"))]
    return docs
