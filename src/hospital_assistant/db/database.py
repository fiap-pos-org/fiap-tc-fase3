"""Banco estruturado do prontuário (SQLite), populado com os prontuários anonimizados.

O assistente só lê este banco, sempre com consultas parametrizadas: o modelo nunca
escreve SQL.
"""

import json
import sqlite3
from pathlib import Path

from hospital_assistant.config import DB_PATH, PROCESSED_DIR

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
"""


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
