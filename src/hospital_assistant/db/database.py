"""Banco estruturado do prontuário (SQLite), populado com os prontuários anonimizados.

O modelo nunca escreve SQL nem grava no banco: as consultas são funções parametrizadas
chamadas pelo fluxo. As únicas escritas são do próprio sistema: alertas de valores críticos,
condutas com a decisão do médico, a trilha de auditoria e a chegada de resultados de exames (simulação do laboratório).
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from hospital_assistant.config import DB_PATH, PROCESSED_DIR
from hospital_assistant.data.exames import EXAMES_LAB

SCHEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    id TEXT PRIMARY KEY,
    idade INTEGER NOT NULL,
    sexo TEXT NOT NULL,
    setor TEXT NOT NULL,
    data_internacao TEXT NOT NULL,
    diagnostico TEXT NOT NULL,
    protocolo TEXT NOT NULL,
    nota_admissao TEXT NOT NULL,
    evolucao TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alergias (
    paciente_id TEXT NOT NULL REFERENCES pacientes(id),
    substancia TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS medicacoes (
    paciente_id TEXT NOT NULL REFERENCES pacientes(id),
    descricao TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id),
    codigo TEXT,
    nome TEXT NOT NULL,
    valor REAL,
    unidade TEXT,
    laudo TEXT,
    status TEXT NOT NULL CHECK (status IN ('resultado', 'pendente')),
    data TEXT,
    data_solicitacao TEXT
);
CREATE INDEX IF NOT EXISTS idx_exames_paciente ON exames(paciente_id);
CREATE TABLE IF NOT EXISTS alertas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id),
    trace_id TEXT NOT NULL,
    mensagem TEXT NOT NULL,
    criado_em TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS condutas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id),
    trace_id TEXT NOT NULL,
    sugestao TEXT NOT NULL,
    fontes TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('aprovada', 'rejeitada', 'bloqueada')),
    medico TEXT,
    comentario TEXT,
    registrado_em TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS auditoria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    fluxo TEXT NOT NULL,
    etapa TEXT NOT NULL,
    paciente_id TEXT,
    dados TEXT NOT NULL,
    duracao_ms INTEGER NOT NULL,
    criado_em TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auditoria_trace ON auditoria(trace_id);
"""


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def conectar(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def popular(conn: sqlite3.Connection, prontuarios: list[dict]) -> None:
    conn.executescript(SCHEMA)
    for p in prontuarios:
        conn.execute(
            "INSERT INTO pacientes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                p["id"],
                p["idade"],
                p["sexo"],
                p["setor"],
                p["data_internacao"],
                p["diagnostico"],
                p["protocolo"],
                p["nota_admissao"],
                p["evolucao"],
            ),
        )
        conn.executemany(
            "INSERT INTO alergias VALUES (?, ?)", [(p["id"], a) for a in p["alergias"]]
        )
        conn.executemany(
            "INSERT INTO medicacoes VALUES (?, ?)", [(p["id"], m) for m in p["medicacoes_em_uso"]]
        )
        for e in p["exames"]:
            conn.execute(
                "INSERT INTO exames (paciente_id, codigo, nome, valor, unidade, laudo, status, data,"
                " data_solicitacao) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    p["id"],
                    e.get("codigo"),
                    e["nome"],
                    e.get("valor"),
                    e.get("unidade"),
                    e.get("laudo"),
                    e["status"],
                    e.get("data"),
                    e.get("data_solicitacao"),
                ),
            )
    conn.commit()


def inicializar(path: Path = DB_PATH) -> sqlite3.Connection:
    """Abre o banco; cria e popula a partir dos prontuários anonimizados se ainda não existir."""
    novo = not path.exists()
    conn = conectar(path)
    conn.executescript(SCHEMA)  # cria tabelas novas em bancos antigos
    if novo:
        with (PROCESSED_DIR / "prontuarios.jsonl").open(encoding="utf-8") as f:
            popular(conn, [json.loads(linha) for linha in f if linha.strip()])
    return conn


def listar_pacientes(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, idade, sexo, setor, diagnostico FROM pacientes ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def buscar_paciente(conn: sqlite3.Connection, paciente_id: str) -> dict | None:
    """Prontuário completo, no mesmo formato usado no treino (ver prompts.format_patient_context)."""
    row = conn.execute("SELECT * FROM pacientes WHERE id = ?", (paciente_id,)).fetchone()
    if row is None:
        return None
    paciente = dict(row)
    paciente["alergias"] = [
        r["substancia"]
        for r in conn.execute(
            "SELECT substancia FROM alergias WHERE paciente_id = ?", (paciente_id,)
        )
    ]
    paciente["medicacoes_em_uso"] = [
        r["descricao"]
        for r in conn.execute(
            "SELECT descricao FROM medicacoes WHERE paciente_id = ?", (paciente_id,)
        )
    ]
    paciente["exames"] = [
        {k: v for k, v in dict(r).items() if v is not None and k != "paciente_id"}
        for r in conn.execute(
            "SELECT * FROM exames WHERE paciente_id = ? ORDER BY id", (paciente_id,)
        )
    ]
    return paciente


def registrar_alerta(
    conn: sqlite3.Connection, paciente_id: str, trace_id: str, mensagem: str
) -> None:
    conn.execute(
        "INSERT INTO alertas (paciente_id, trace_id, mensagem, criado_em) VALUES (?, ?, ?, ?)",
        (paciente_id, trace_id, mensagem, _agora()),
    )
    conn.commit()


def registrar_conduta(
    conn: sqlite3.Connection,
    paciente_id: str,
    trace_id: str,
    sugestao: str,
    fontes: list[str],
    status: str,
    medico: str | None = None,
    comentario: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO condutas (paciente_id, trace_id, sugestao, fontes, status, medico, comentario,"
        " registrado_em) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            paciente_id,
            trace_id,
            sugestao,
            json.dumps(fontes, ensure_ascii=False),
            status,
            medico,
            comentario,
            _agora(),
        ),
    )
    conn.commit()


def listar_registros(conn: sqlite3.Connection, tabela: str, trace_id: str) -> list[dict]:
    """Alertas ou condutas de uma execução do fluxo."""
    if tabela not in {"alertas", "condutas"}:
        raise ValueError(tabela)
    rows = conn.execute(f"SELECT * FROM {tabela} WHERE trace_id = ? ORDER BY id", (trace_id,))
    return [dict(r) for r in rows]


def registrar_resultado(
    conn: sqlite3.Connection, paciente_id: str, codigo: str, valor: float
) -> None:
    """Simula o laboratório liberando um novo resultado (o prontuário passa a refleti-lo)."""
    exame = EXAMES_LAB[codigo]
    conn.execute(
        "INSERT INTO exames (paciente_id, codigo, nome, valor, unidade, status, data)"
        " VALUES (?, ?, ?, ?, ?, 'resultado', ?)",
        (paciente_id, codigo, exame.nome, valor, exame.unidade, _agora()),
    )
    conn.commit()
