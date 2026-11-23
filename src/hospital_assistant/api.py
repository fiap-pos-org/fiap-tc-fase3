"""API web do assistente (FastAPI) e o front-end estático (Tabler) em /.

Rotas:
- GET  /api/info, /api/pacientes, /api/pacientes/{id}
- POST /api/duvida                       fluxo B (dúvida clínica)
- POST /api/triagem                      fluxo A até a pausa de aprovação
- POST /api/triagem/{trace_id}/decisao   decisão do médico (retoma o fluxo)
- GET  /api/auditoria, /api/auditoria/{trace_id}

Uso: uv run python -m hospital_assistant.api (ou via docker compose)
"""

import logging
import sqlite3
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

from hospital_assistant.config import MODEL_NAME
from hospital_assistant.data.exames import EXAMES_LAB
from hospital_assistant.db.database import buscar_paciente, listar_pacientes, listar_registros
from hospital_assistant.graph.duvida import criar_fluxo_duvida
from hospital_assistant.graph.triagem import criar_fluxo_triagem
from hospital_assistant.observability.auditoria import listar_execucoes, trilha
from hospital_assistant.prompts import fmt_exame, fmt_limite_critico
from hospital_assistant.safety.guardrails import fontes_citadas

ESTATICOS = Path(__file__).parent / "static"

ETAPAS = {
    "carregar_prontuario": "Carregar prontuário",
    "verificar_pendencias": "Verificar exames pendentes",
    "checar_valores_criticos": "Checar valores críticos",
    "emitir_alertas": "Emitir alertas à equipe",
    "buscar_protocolos": "Buscar protocolos",
    "gerar_sugestao": "Gerar sugestão (modelo ajustado)",
    "validar_seguranca": "Validar segurança",
    "aprovacao_humana": "Aprovação do médico",
    "registrar_conduta": "Registrar conduta",
    "verificar_entrada": "Verificar entrada",
    "responder_bloqueio": "Responder pela trava de segurança",
    "montar_contexto": "Montar contexto",
    "gerar_resposta": "Gerar resposta (modelo ajustado)",
    "validar_resposta": "Validar resposta",
    "resposta_segura": "Substituir por resposta segura",
}


class PedidoDuvida(BaseModel):
    pergunta: str = Field(min_length=1, max_length=2000)
    paciente_id: str | None = None


class PedidoTriagem(BaseModel):
    paciente_id: str


class Decisao(BaseModel):
    aprovar: bool
    medico: str = Field(min_length=3, max_length=200)
    comentario: str = Field(default="", max_length=1000)


def separar_fontes(resposta: str) -> tuple[str, list[str]]:
    """Separa o texto da linha 'Fontes:' (o front mostra as fontes como etiquetas)."""
    if "Fontes:" not in resposta:
        return resposta.strip(), []
    return resposta.rsplit("Fontes:", 1)[0].strip(), fontes_citadas(resposta)


def _exame(e: dict) -> dict:
    critico = "valor" in e and EXAMES_LAB[e["codigo"]].is_critico(e["valor"])
    nome, _, resultado = (
        fmt_exame(e).partition(": ") if e["status"] == "resultado" else (e["nome"], "", "")
    )
    return {
        "nome": nome,
        "resultado": resultado,
        "status": e["status"],
        "critico": critico,
        "limite_critico": fmt_limite_critico(e["codigo"]) if critico else None,
        "data": e.get("data") or e.get("data_solicitacao"),
    }


def criar_app(conn: sqlite3.Connection, indice, cadeia, checkpointer) -> FastAPI:
    fluxo_duvida = criar_fluxo_duvida(conn, indice, cadeia)
    fluxo_triagem = criar_fluxo_triagem(conn, indice, cadeia, checkpointer)
    # Um fluxo por vez: o modelo atende uma requisição por vez e o SQLite é compartilhado.
    trava = threading.Lock()
    app = FastAPI(title="Assistente Médico - Hospital Santa Aurora")

    def _etapas(trace_id: str) -> list[dict]:
        return [
            {"id": e["etapa"], "nome": ETAPAS.get(e["etapa"], e["etapa"]), "ms": e["duracao_ms"]}
            for e in trilha(conn, trace_id)
        ]

    def _triagem(trace_id: str) -> dict:
        snapshot = fluxo_triagem.get_state({"configurable": {"thread_id": trace_id}})
        estado = snapshot.values
        texto, fontes = separar_fontes(estado.get("sugestao", ""))
        condutas = listar_registros(conn, "condutas", trace_id)
        return {
            "trace_id": trace_id,
            "paciente_id": estado.get("paciente_id"),
            "status": "aguardando" if snapshot.next else estado.get("status"),
            "etapas": _etapas(trace_id),
            "alertas": estado.get("alertas", []),
            "pendentes": estado.get("pendentes", []),
            "sugestao": texto,
            "fontes": fontes,
            "trechos": [{"ref": r, "texto": t} for r, t in estado.get("trechos", [])],
            "validacao": estado.get("validacao"),
            "registro": condutas[-1] if condutas else None,
        }

    @app.get("/api/info")
    def info() -> dict:
        return {"modelo": MODEL_NAME}

    @app.get("/api/pacientes")
    def pacientes() -> list[dict]:
        resultado = []
        for p in listar_pacientes(conn):
            completo = buscar_paciente(conn, p["id"])
            exames = [_exame(e) for e in completo["exames"]]
            resultado.append(
                {
                    **p,
                    "criticos": sum(e["critico"] for e in exames),
                    "pendentes": sum(e["status"] == "pendente" for e in exames),
                }
            )
        return resultado

    @app.get("/api/pacientes/{paciente_id}")
    def paciente(paciente_id: str) -> dict:
        p = buscar_paciente(conn, paciente_id)
        if p is None:
            raise HTTPException(404, "Paciente não encontrado")
        return {
            **{k: v for k, v in p.items() if k != "exames"},
            "exames": [_exame(e) for e in p["exames"]],
        }

    @app.post("/api/duvida")
    def duvida(pedido: PedidoDuvida) -> dict:
        trace_id = uuid.uuid4().hex[:12]
        with trava:
            estado = fluxo_duvida.invoke(
                {
                    "pergunta": pedido.pergunta,
                    "paciente_id": pedido.paciente_id,
                    "trace_id": trace_id,
                }
            )
        texto, fontes = separar_fontes(estado["resposta"])
        return {
            "trace_id": trace_id,
            "resposta": texto,
            "fontes": fontes,
            "trechos": [{"ref": r, "texto": t} for r, t in estado.get("trechos", [])],
            "categoria": estado["categoria"],
            "substituida": "resposta_original" in estado,
            "validacao": estado.get("validacao"),
        }

    @app.post("/api/triagem")
    def triagem(pedido: PedidoTriagem) -> dict:
        if buscar_paciente(conn, pedido.paciente_id) is None:
            raise HTTPException(404, "Paciente não encontrado")
        trace_id = uuid.uuid4().hex[:12]
        with trava:
            fluxo_triagem.invoke(
                {"paciente_id": pedido.paciente_id, "trace_id": trace_id},
                {"configurable": {"thread_id": trace_id}},
            )
        return _triagem(trace_id)

    @app.post("/api/triagem/{trace_id}/decisao")
    def decidir(trace_id: str, decisao: Decisao) -> dict:
        cfg = {"configurable": {"thread_id": trace_id}}
        if not fluxo_triagem.get_state(cfg).next:
            raise HTTPException(409, "Esta triagem não está aguardando decisão")
        with trava:
            fluxo_triagem.invoke(Command(resume=decisao.model_dump()), cfg)
        return _triagem(trace_id)

    @app.get("/api/auditoria")
    def auditoria() -> list[dict]:
        return listar_execucoes(conn)

    @app.get("/api/auditoria/{trace_id}")
    def auditoria_trilha(trace_id: str) -> list[dict]:
        eventos = trilha(conn, trace_id)
        if not eventos:
            raise HTTPException(404, "Execução não encontrada")
        return [{**e, "nome": ETAPAS.get(e["etapa"], e["etapa"])} for e in eventos]

    app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(ESTATICOS / "index.html")

    return app


def main() -> None:
    import uvicorn
    from langchain_ollama import ChatOllama
    from langgraph.checkpoint.sqlite import SqliteSaver

    from hospital_assistant.assistant.qa import criar_cadeia
    from hospital_assistant.config import OLLAMA_BASE_URL, OLLAMA_NUM_THREAD, STORAGE_DIR
    from hospital_assistant.db.database import inicializar
    from hospital_assistant.rag.indice import criar_indice

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    conn = inicializar()
    llm = ChatOllama(
        model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.2, num_thread=OLLAMA_NUM_THREAD
    )
    checkpointer = SqliteSaver(
        sqlite3.connect(STORAGE_DIR / "checkpoints.db", check_same_thread=False)
    )
    app = criar_app(conn, criar_indice(), criar_cadeia(llm), checkpointer)
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
