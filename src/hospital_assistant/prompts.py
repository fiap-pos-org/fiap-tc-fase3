"""Prompt de sistema e formato das mensagens. O mesmo formato é usado no treino e na inferência."""

from hospital_assistant.data.exames import EXAMES_LAB

SYSTEM_PROMPT = (
    "Você é o assistente virtual médico do Hospital Santa Aurora e apoia médicos com base nos "
    "protocolos internos e no prontuário do paciente. Regras: (1) você não prescreve, não altera "
    "prescrições e não decide condutas: toda sugestão fica pendente de validação médica; "
    "(2) responda só com base nos protocolos e no prontuário; sem base, diga que não encontrou "
    "a informação; (3) destaque valores críticos e conflitos com alergias; (4) recuse perguntas "
    "fora do escopo clínico-assistencial; (5) termine sempre com a linha 'Fontes:' citando "
    "[ID §seção] e [PRONTUÁRIO: campo]."
)


def fmt_num(valor: float) -> str:
    """Formata número no padrão pt-BR (2,8 / 203.000)."""
    if float(valor).is_integer():
        return f"{int(valor):,}".replace(",", ".")
    return f"{valor:.2f}".rstrip("0").replace(".", ",")


def fmt_limite_critico(codigo: str) -> str:
    """Limite de valor crítico do exame (PROT-VCRIT-01 §2), ex.: '< 2,8 ou > 6,2'."""
    e = EXAMES_LAB[codigo]
    partes = []
    if e.critico_abaixo is not None:
        partes.append(f"< {fmt_num(e.critico_abaixo)}")
    if e.critico_acima is not None:
        partes.append(f"{'≥' if e.critico_acima_inclusivo else '>'} {fmt_num(e.critico_acima)}")
    return " ou ".join(partes)


def fmt_exame(exame: dict) -> str:
    if "valor" in exame:
        ref = EXAMES_LAB[exame["codigo"]].referencia
        unidade = f" {exame['unidade']}" if exame["unidade"] else ""
        return (
            f"{exame['nome']}: {fmt_num(exame['valor'])}{unidade} "
            f"(ref. {fmt_num(ref[0])} a {fmt_num(ref[1])})"
        )
    return f"{exame['nome']}: {exame['laudo']}"


def format_patient_context(p: dict) -> str:
    resultados = [e for e in p["exames"] if e["status"] == "resultado"]
    pendentes = [e for e in p["exames"] if e["status"] == "pendente"]
    linhas = [
        f"[PRONTUÁRIO {p['id']}]",
        f"Idade/sexo: {p['idade']} anos, {p['sexo']}",
        f"Setor: {p['setor']} | Internação: {p['data_internacao'].replace('T', ' ')}",
        f"Diagnóstico: {p['diagnostico']}",
        f"Alergias: {', '.join(p['alergias']) or 'nenhuma registrada'}",
        f"Medicações em uso: {', '.join(p['medicacoes_em_uso']) or 'nenhuma'}",
        "Exames com resultado:",
        *[f"- {fmt_exame(e)}" for e in resultados],
        "Exames pendentes:",
        *([f"- {e['nome']}" for e in pendentes] or ["- nenhum"]),
    ]
    return "\n".join(linhas)


def format_user_message(
    pergunta: str, paciente: dict | None = None, trechos: list[tuple[str, str]] | None = None
) -> str:
    """Monta a mensagem do usuário. `trechos` são pares (referência, texto) recuperados."""
    partes = []
    if paciente:
        partes.append(format_patient_context(paciente))
    if trechos:
        partes.append(
            "Trechos de protocolos:\n" + "\n".join(f"[{ref}] {txt}" for ref, txt in trechos)
        )
    partes.append(f"Pergunta: {pergunta}")
    return "\n\n".join(partes)
