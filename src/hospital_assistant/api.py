"""API web do assistente (FastAPI) e o front-end estático (Tabler) em /.

Rotas:
- GET  /api/info, /api/pacientes, /api/pacientes/{id}
- POST /api/duvida                       fluxo B (dúvida clínica)
- POST /api/triagem                      fluxo A até a pausa de aprovação
- POST /api/triagem/{trace_id}/decisao   decisão do médico (retoma o fluxo)
- GET  /api/auditoria, /api/auditoria/{trace_id}
- POST /api/login, /api/logout; GET /api/sessao   (login ativo se APP_USUARIO/APP_SENHA definidos)

Uso: uv run python -m hospital_assistant.api (ou via docker compose)
"""

import hashlib
import hmac
import logging
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

from hospital_assistant.config import APP_SENHA, APP_USUARIO, MODEL_NAME
from hospital_assistant.data.exames import EXAMES_LAB
from hospital_assistant.db.database import buscar_paciente, listar_pacientes, listar_registros
from hospital_assistant.graph.duvida import criar_fluxo_duvida
from hospital_assistant.graph.triagem import criar_fluxo_triagem
from hospital_assistant.observability.auditoria import listar_execucoes, registrar_evento, trilha
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
    "responder_ferramenta": "Responder pela ferramenta de valores críticos",
    "montar_contexto": "Montar contexto",
    "gerar_resposta": "Gerar resposta (modelo ajustado)",
    "validar_resposta": "Validar resposta",
    "resposta_segura": "Substituir por resposta segura",
    "requisicao": "Requisição recebida",
}


class PedidoDuvida(BaseModel):
    pergunta: str = Field(min_length=1, max_length=2000)
    paciente_id: str | None = None


class PedidoTriagem(BaseModel):
    paciente_id: str


class Credenciais(BaseModel):
    usuario: str = Field(max_length=100)
    senha: str = Field(max_length=200)


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


COOKIE = "sessao"
DURACAO_SESSAO = 8 * 3600


def criar_app(
    conn: sqlite3.Connection,
    indice,
    cadeia,
    checkpointer,
    usuario: str | None = APP_USUARIO,
    senha: str | None = APP_SENHA,
) -> FastAPI:
    fluxo_duvida = criar_fluxo_duvida(conn, indice, cadeia)
    fluxo_triagem = criar_fluxo_triagem(conn, indice, cadeia, checkpointer)
    # Um fluxo por vez: o modelo atende uma requisição por vez e o SQLite é compartilhado.
    trava = threading.Lock()
    app = FastAPI(title="Assistente Médico - Hospital Santa Aurora")

    # ---- login: cookie assinado com HMAC (segredo novo a cada início = sessões expiram no restart)
    login_ativo = bool(usuario and senha)
    segredo = secrets.token_bytes(32)

    def _assinar(nome: str, expira: int) -> str:
        assinatura = hmac.new(segredo, f"{nome}|{expira}".encode(), hashlib.sha256).hexdigest()
        return f"{nome}|{expira}|{assinatura}"

    def _usuario_da_sessao(request: Request) -> str | None:
        if not login_ativo:
            return "local"
        try:
            nome, expira, assinatura = request.cookies.get(COOKIE, "").split("|")
        except ValueError:
            return None
        esperado = _assinar(nome, int(expira)).rsplit("|", 1)[1]
        if hmac.compare_digest(assinatura, esperado) and int(expira) > time.time():
            return nome
        return None

    def usuario_atual(request: Request) -> str:
        nome = _usuario_da_sessao(request)
        if nome is None:
            raise HTTPException(401, "Faça login para continuar")
        return nome

    def _registrar_requisicao(
        trace_id: str, fluxo: str, nome: str, paciente_id, dados: dict
    ) -> None:
        """Registra quem pediu o quê (PROT-SEGMED-01 §6) antes de o fluxo rodar."""
        registrar_evento(
            conn, trace_id, fluxo, "requisicao", paciente_id, {"usuario": nome, **dados}, 0
        )

    @app.get("/api/sessao")
    def sessao(request: Request) -> dict:
        return {"login_ativo": login_ativo, "usuario": _usuario_da_sessao(request)}

    @app.post("/api/login")
    def login(credenciais: Credenciais, request: Request, response: Response) -> dict:
        if not login_ativo:
            return {"usuario": "local"}
        ok = hmac.compare_digest(credenciais.usuario, usuario) & hmac.compare_digest(
            credenciais.senha, senha
        )
        if not ok:
            time.sleep(1)  # freia tentativas de força bruta
            raise HTTPException(401, "Usuário ou senha inválidos")
        https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
        response.set_cookie(
            COOKIE,
            _assinar(usuario, int(time.time()) + DURACAO_SESSAO),
            max_age=DURACAO_SESSAO,
            httponly=True,
            samesite="lax",
            secure=https,
        )
        return {"usuario": usuario}

    @app.post("/api/logout")
    def logout(response: Response) -> dict:
        response.delete_cookie(COOKIE)
        return {"ok": True}

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
    def pacientes(_: str = Depends(usuario_atual)) -> list[dict]:
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
    def paciente(paciente_id: str, _: str = Depends(usuario_atual)) -> dict:
        p = buscar_paciente(conn, paciente_id)
        if p is None:
            raise HTTPException(404, "Paciente não encontrado")
        return {
            **{k: v for k, v in p.items() if k != "exames"},
            "exames": [_exame(e) for e in p["exames"]],
        }

    @app.post("/api/duvida")
    def duvida(pedido: PedidoDuvida, nome: str = Depends(usuario_atual)) -> dict:
        trace_id = uuid.uuid4().hex[:12]
        _registrar_requisicao(
            trace_id, "duvida", nome, pedido.paciente_id, {"pergunta": pedido.pergunta}
        )
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
    def triagem(pedido: PedidoTriagem, nome: str = Depends(usuario_atual)) -> dict:
        if buscar_paciente(conn, pedido.paciente_id) is None:
            raise HTTPException(404, "Paciente não encontrado")
        trace_id = uuid.uuid4().hex[:12]
        _registrar_requisicao(trace_id, "triagem", nome, pedido.paciente_id, {})
        with trava:
            fluxo_triagem.invoke(
                {"paciente_id": pedido.paciente_id, "trace_id": trace_id},
                {"configurable": {"thread_id": trace_id}},
            )
        return _triagem(trace_id)

    @app.post("/api/triagem/{trace_id}/decisao")
    def decidir(trace_id: str, decisao: Decisao, nome: str = Depends(usuario_atual)) -> dict:
        cfg = {"configurable": {"thread_id": trace_id}}
        if not fluxo_triagem.get_state(cfg).next:
            raise HTTPException(409, "Esta triagem não está aguardando decisão")
        with trava:
            fluxo_triagem.invoke(Command(resume={**decisao.model_dump(), "usuario": nome}), cfg)
        return _triagem(trace_id)

    @app.get("/api/auditoria")
    def auditoria(_: str = Depends(usuario_atual)) -> list[dict]:
        return listar_execucoes(conn)

    @app.get("/api/auditoria/{trace_id}")
    def auditoria_trilha(trace_id: str, _: str = Depends(usuario_atual)) -> list[dict]:
        eventos = trilha(conn, trace_id)
        if not eventos:
            raise HTTPException(404, "Execução não encontrada")
        return [{**e, "nome": ETAPAS.get(e["etapa"], e["etapa"])} for e in eventos]

    app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")

    @app.get("/", include_in_schema=False)
    def index(request: Request) -> FileResponse:
        if _usuario_da_sessao(request) is None:
            return FileResponse(ESTATICOS / "login.html")
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
