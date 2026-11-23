"""Busca semântica nos protocolos e modelos internos (RAG).

Cada seção de documento vira um trecho com referência `ID §n`, a mesma usada nas citações.
Os embeddings vêm do bge-m3 (multilíngue) servido pelo Ollama. Com ~100 trechos, um índice
em memória basta. Ele é salvo em disco e só é remontado quando o conteúdo dos documentos ou
o modelo de embedding mudam (montar do zero leva ~40 s em CPU).
"""

import hashlib
from pathlib import Path

from langchain_core.documents import Document as Trecho
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from hospital_assistant.config import (
    EMBEDDING_MODEL,
    INDICE_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_THREAD,
)
from hospital_assistant.data.documents import Document, load_documents


def montar_trechos(docs: list[Document]) -> list[Trecho]:
    trechos = []
    for doc in docs:
        for secao in doc.secoes:
            # O §1 (objetivo/escopo ou indicação) resume o documento todo e "roubava" o 1º lugar
            # da busca de seções mais específicas: tirá-lo subiu o acerto em 1º de 72% para 80%.
            if secao.numero == 1:
                continue
            ref = f"{doc.id} §{secao.numero}"
            trechos.append(
                Trecho(
                    # O título do documento entra no texto indexado para dar contexto à busca.
                    page_content=f"{doc.titulo}. {secao.titulo}: {secao.texto}",
                    metadata={"ref": ref, "texto": f"{secao.titulo}: {secao.texto}"},
                )
            )
    return trechos


def _assinatura(trechos: list[Trecho]) -> str:
    conteudo = EMBEDDING_MODEL + "".join(t.page_content for t in trechos)
    return hashlib.sha256(conteudo.encode()).hexdigest()


def criar_indice(
    embeddings: Embeddings | None = None, path: Path | None = INDICE_PATH
) -> InMemoryVectorStore:
    """Carrega o índice salvo se estiver atualizado; senão monta e salva. `path=None` não persiste."""
    embeddings = embeddings or OllamaEmbeddings(
        model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL, num_thread=OLLAMA_NUM_THREAD
    )
    trechos = montar_trechos(load_documents())
    assinatura = _assinatura(trechos)
    if path is not None:
        marcador = path.with_suffix(".sha256")
        if path.exists() and marcador.exists() and marcador.read_text() == assinatura:
            return InMemoryVectorStore.load(str(path), embeddings)

    indice = InMemoryVectorStore.from_documents(trechos, embeddings)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        indice.dump(str(path))
        marcador.write_text(assinatura)
    return indice


def buscar(indice: InMemoryVectorStore, pergunta: str, k: int = 3) -> list[tuple[str, str]]:
    """Retorna os k trechos mais próximos como pares (referência, texto)."""
    return [
        (t.metadata["ref"], t.metadata["texto"]) for t in indice.similarity_search(pergunta, k=k)
    ]


def similaridade_maxima(indice: InMemoryVectorStore, pergunta: str) -> float:
    """Similaridade (cosseno) entre a pergunta e o trecho mais próximo."""
    return indice.similarity_search_with_score(pergunta, k=1)[0][1]
