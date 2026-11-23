"""Ferramentas LangChain de consulta ao prontuário e aos protocolos.

São chamadas pelos nós dos fluxos (LangGraph) em ordem fixa, não escolhidas pelo modelo:
num fluxo clínico a sequência de verificações precisa ser garantida, e um modelo de 1,5B não
faz tool-calling de forma confiável. Todas são somente leitura.
"""

import sqlite3

from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import InMemoryVectorStore

from hospital_assistant.data.exames import EXAMES_LAB
from hospital_assistant.db.database import buscar_paciente
from hospital_assistant.prompts import fmt_limite_critico, fmt_num
from hospital_assistant.rag.indice import buscar


def criar_ferramentas(conn: sqlite3.Connection, indice: InMemoryVectorStore) -> dict[str, BaseTool]:
    @tool
    def consultar_prontuario(paciente_id: str) -> dict | None:
        """Retorna o prontuário do paciente: dados, alergias, medicações e exames."""
        return buscar_paciente(conn, paciente_id)

    @tool
    def listar_exames_pendentes(paciente_id: str) -> list[str]:
        """Lista os exames solicitados e ainda sem resultado."""
        paciente = buscar_paciente(conn, paciente_id) or {"exames": []}
        return [e["nome"] for e in paciente["exames"] if e["status"] == "pendente"]

    @tool
    def verificar_valores_criticos(paciente_id: str) -> list[dict]:
        """Compara os resultados com os limites de valor crítico do PROT-VCRIT-01 §2."""
        paciente = buscar_paciente(conn, paciente_id) or {"exames": []}
        return [
            {
                "exame": e["nome"],
                "valor": f"{fmt_num(e['valor'])} {e['unidade']}".strip(),
                "limite": fmt_limite_critico(e["codigo"]),
            }
            for e in paciente["exames"]
            if e.get("codigo") in EXAMES_LAB and EXAMES_LAB[e["codigo"]].is_critico(e["valor"])
        ]

    @tool
    def buscar_protocolos(consulta: str, k: int = 3) -> list[tuple[str, str]]:
        """Busca semântica nos protocolos e modelos internos; retorna (referência, trecho)."""
        return buscar(indice, consulta, k=k)

    ferramentas = [
        consultar_prontuario,
        listar_exames_pendentes,
        verificar_valores_criticos,
        buscar_protocolos,
    ]
    return {f.name: f for f in ferramentas}
