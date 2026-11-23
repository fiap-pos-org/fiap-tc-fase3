"""Trilha de auditoria dos fluxos: um evento por etapa, ligado pelo trace_id da execução.

Cada evento vai para a tabela `auditoria` do banco (consulta na interface e por linha de
comando) e para o log da aplicação em JSON (stdout, visível em `docker compose logs app`).

Uso: uv run python -m hospital_assistant.observability.auditoria [trace_id]
"""

import json
import logging
import sqlite3
import sys
import time
from collections.abc import Callable
from datetime import datetime

from hospital_assistant.config import MODEL_NAME

logger = logging.getLogger("auditoria")


def _serializavel(dados: dict) -> dict:
    """Resume o estado para o log: o prontuário completo vira só o id do paciente."""
    resumo = {}
    for chave, valor in dados.items():
        if chave == "paciente":
            resumo["paciente"] = (valor or {}).get("id")
        else:
            resumo[chave] = valor
    return resumo


def registrar_evento(
    conn: sqlite3.Connection,
    trace_id: str,
    fluxo: str,
    etapa: str,
    paciente_id: str | None,
    dados: dict,
    duracao_ms: int,
) -> None:
    evento = {
        "trace_id": trace_id,
        "fluxo": fluxo,
        "etapa": etapa,
        "paciente_id": paciente_id,
        "dados": _serializavel(dados),
        "duracao_ms": duracao_ms,
        "criado_em": datetime.now().isoformat(timespec="milliseconds"),
    }
    conn.execute(
        "INSERT INTO auditoria (trace_id, fluxo, etapa, paciente_id, dados, duracao_ms, criado_em)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            trace_id,
            fluxo,
            etapa,
            paciente_id,
            json.dumps(evento["dados"], ensure_ascii=False, default=str),
            duracao_ms,
            evento["criado_em"],
        ),
    )
    conn.commit()
    logger.info(json.dumps(evento, ensure_ascii=False, default=str))


def auditado(conn: sqlite3.Connection, fluxo: str, etapa: str, no: Callable) -> Callable:
    """Envolve um nó do grafo: registra o que ele produziu e quanto tempo levou."""

    def executar(estado: dict) -> dict:
        inicio = time.perf_counter()
        saida = no(estado)
        dados = dict(saida)
        if etapa.startswith("gerar_"):
            dados["modelo"] = MODEL_NAME
        registrar_evento(
            conn,
            estado["trace_id"],
            fluxo,
            etapa,
            estado.get("paciente_id"),
            dados,
            round((time.perf_counter() - inicio) * 1000),
        )
        return saida

    return executar


def listar_execucoes(conn: sqlite3.Connection, limite: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT trace_id, fluxo, paciente_id, MIN(criado_em) AS inicio, COUNT(*) AS etapas"
        " FROM auditoria GROUP BY trace_id ORDER BY inicio DESC LIMIT ?",
        (limite,),
    )
    return [dict(r) for r in rows]


def trilha(conn: sqlite3.Connection, trace_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT etapa, dados, duracao_ms, criado_em FROM auditoria WHERE trace_id = ? ORDER BY id",
        (trace_id,),
    )
    return [{**dict(r), "dados": json.loads(r["dados"])} for r in rows]


if __name__ == "__main__":
    from hospital_assistant.db.database import inicializar

    conn = inicializar()
    if len(sys.argv) < 2:
        for e in listar_execucoes(conn, 20):
            print(
                f"{e['inicio']}  {e['trace_id']}  {e['fluxo']:8}  {e['paciente_id'] or '-':9}  {e['etapas']} etapas"
            )
    else:
        for evento in trilha(conn, sys.argv[1]):
            print(f"[{evento['criado_em']}] {evento['etapa']} ({evento['duracao_ms']} ms)")
            print(json.dumps(evento["dados"], ensure_ascii=False, indent=2))
