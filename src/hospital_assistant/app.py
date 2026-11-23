"""Interface web do assistente (Gradio).

- Dúvida clínica: perguntas com o modelo ajustado, contextualizadas com o prontuário do
  paciente (SQLite) e com trechos de protocolo (busca semântica). Fluxo B do LangGraph.
- Triagem automática: fluxo A do LangGraph (pendências, valores críticos, alertas, sugestão,
  validação de segurança) com pausa para aprovação do médico.

Uso: uv run python -m hospital_assistant.app (ou via docker compose)
"""

import html
import sqlite3
import uuid

import gradio as gr
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from hospital_assistant.assistant.qa import criar_cadeia
from hospital_assistant.config import (
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_THREAD,
    STORAGE_DIR,
)
from hospital_assistant.db.database import (
    buscar_paciente,
    inicializar,
    listar_pacientes,
    listar_registros,
)
from hospital_assistant.graph.duvida import criar_fluxo_duvida
from hospital_assistant.graph.triagem import criar_fluxo_triagem
from hospital_assistant.prompts import format_patient_context
from hospital_assistant.rag.indice import criar_indice

EXEMPLOS_GERAIS = [
    "Qual o prazo para iniciar o antibiótico na sepse?",
    "Quais os critérios diagnósticos de cetoacidose diabética?",
    "Quais valores de potássio são críticos?",
    "Prescreva ceftriaxona 1 g EV para o paciente.",
    "Qual o protocolo do hospital para grandes queimados?",
]
EXEMPLOS_PACIENTE = [
    "Quais exames estão pendentes para este paciente?",
    "Qual a conduta inicial prevista para este caso?",
    "Resuma o caso deste paciente.",
]

ETAPAS = {
    "carregar_prontuario": "Carregar prontuário",
    "verificar_pendencias": "Verificar exames pendentes",
    "checar_valores_criticos": "Checar valores críticos",
    "emitir_alertas": "Emitir alertas à equipe",
    "buscar_protocolos": "Buscar protocolos",
    "gerar_sugestao": "Gerar sugestão (modelo ajustado)",
    "validar_seguranca": "Validar segurança",
    "aprovacao_humana": "Aguardar aprovação do médico",
    "registrar_conduta": "Registrar conduta",
}

LIMITES = """
**Limites de atuação**

- Não prescreve nem altera prescrições
- Toda sugestão depende de validação médica
- Responde só com base nos protocolos e no prontuário
- Cita a fonte de cada resposta
- Recusa perguntas fora do escopo clínico
"""

CSS = """
.cabecalho { padding: 20px 24px; border-radius: 12px;
  background: linear-gradient(135deg, #0f4c5c 0%, #1b6f7f 100%); color: #fff; }
.cabecalho h1 { margin: 0; font-size: 1.5rem; font-weight: 600; color: #fff; }
.cabecalho p { margin: 4px 0 0; opacity: .85; font-size: .95rem; color: #fff; }
.selo { display: inline-block; margin-top: 10px; padding: 2px 10px; border-radius: 999px;
  background: rgba(255,255,255,.15); font-size: .8rem; }
.painel { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 16px; }
.rodape { font-size: .8rem; opacity: .7; text-align: center; }
.fontes { margin-top: 8px; padding: 8px 12px; border-left: 3px solid #1b6f7f;
  background: rgba(27,111,127,.08); font-size: .85rem; border-radius: 4px; }
.aviso { margin-top: 8px; padding: 8px 12px; border-left: 3px solid #b45309;
  background: rgba(180,83,9,.08); font-size: .85rem; border-radius: 4px; }
.etapas { list-style: none; padding: 0; margin: 0; }
.etapas li { padding: 6px 10px; margin: 4px 0; border-radius: 6px; font-size: .9rem;
  border: 1px solid var(--border-color-primary); }
.etapas li.feita { border-left: 4px solid #1b6f7f; }
.etapas li.pausa { border-left: 4px solid #b45309; font-weight: 600; }
.alerta { padding: 10px 12px; margin: 6px 0; border-radius: 6px; font-size: .9rem;
  border-left: 4px solid #b91c1c; background: rgba(185,28,28,.08); }
.status { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; }
.status.aprovada { background: rgba(21,128,61,.12); color: #15803d; }
.status.rejeitada, .status.bloqueada { background: rgba(185,28,28,.12); color: #b91c1c; }
.status.aguardando { background: rgba(180,83,9,.12); color: #b45309; }
"""

TEMA = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

SEM_PACIENTE = "Nenhum (pergunta geral)"


def formatar(resposta: str) -> str:
    """Destaca a linha de fontes num bloco próprio."""
    if "Fontes:" not in resposta:
        return resposta
    texto, fontes = resposta.rsplit("Fontes:", 1)
    return f'{texto.strip()}\n\n<div class="fontes"><b>Fontes:</b> {fontes.strip()}</div>'


def formatar_trechos(trechos: list[tuple[str, str]]) -> str:
    if not trechos:
        return "Nenhum trecho consultado ainda."
    return "\n\n".join(f"**[{ref}]** {texto}" for ref, texto in trechos)


def formatar_etapas(feitas: list[str], pausado: bool = False) -> str:
    itens = [f'<li class="feita">{ETAPAS[n]}: concluída</li>' for n in feitas if n in ETAPAS]
    if pausado:
        itens.append(f'<li class="pausa">{ETAPAS["aprovacao_humana"]}</li>')
    return f'<ul class="etapas">{"".join(itens)}</ul>' if itens else ""


def formatar_resultado(estado: dict, status: str) -> str:
    partes = [f'<span class="status {status}">{status.replace("_", " ").capitalize()}</span>']
    if estado.get("alertas"):
        partes.append("**Alertas emitidos**")
        partes += [f'<div class="alerta">{html.escape(a)}</div>' for a in estado["alertas"]]
    if estado.get("pendentes") is not None:
        pend = ", ".join(estado["pendentes"]) or "nenhum"
        partes.append(f"**Exames pendentes:** {pend}")
    if estado.get("sugestao"):
        partes.append("**Sugestão do assistente**")
        partes.append(formatar(estado["sugestao"]))
    validacao = estado.get("validacao")
    if validacao and not validacao["aprovada"]:
        motivos = "; ".join(validacao["motivos"])
        partes.append(f'<div class="aviso"><b>Bloqueada na validação:</b> {motivos}</div>')
    return "\n\n".join(partes)


def criar_interface(conn, indice, cadeia, checkpointer) -> gr.Blocks:
    fluxo_duvida = criar_fluxo_duvida(conn, indice, cadeia)
    fluxo_triagem = criar_fluxo_triagem(conn, indice, cadeia, checkpointer)
    opcoes = [SEM_PACIENTE] + [
        f"{p['id']} | {p['idade']} anos, {p['sexo']} | {p['diagnostico']}"
        for p in listar_pacientes(conn)
    ]

    def _paciente_id(opcao: str) -> str | None:
        return None if opcao in (None, SEM_PACIENTE) else opcao.split(" | ")[0]

    def mostrar_paciente(opcao: str) -> str:
        pid = _paciente_id(opcao)
        if pid is None:
            return "Sem paciente selecionado: as respostas usam só os protocolos."
        return f"```\n{format_patient_context(buscar_paciente(conn, pid))}\n```"

    # ------------------------------------------------------------ dúvida clínica
    def perguntar(mensagem: str, historico: list, opcao: str):
        if not mensagem.strip():
            yield "", historico, gr.skip()
            return
        historico = historico + [{"role": "user", "content": mensagem}]
        entrada = {
            "pergunta": mensagem,
            "paciente_id": _paciente_id(opcao),
            "trace_id": uuid.uuid4().hex[:12],
        }
        resposta, estado = "", {}
        for modo, dado in fluxo_duvida.stream(entrada, stream_mode=["messages", "values"]):
            if modo == "messages":
                resposta += dado[0].content
                yield "", historico + [{"role": "assistant", "content": resposta}], gr.skip()
            else:
                estado = dado
        final = formatar(estado["resposta"])
        if not estado["validacao"]["aprovada"]:
            motivos = "; ".join(estado["validacao"]["motivos"])
            final += f'\n\n<div class="aviso"><b>Atenção:</b> {motivos}</div>'
        trechos = formatar_trechos(estado["trechos"])
        yield "", historico + [{"role": "assistant", "content": final}], trechos

    # ------------------------------------------------------------ triagem
    def executar_triagem(opcao: str):
        pid = _paciente_id(opcao)
        if pid is None:
            yield "", "Selecione um paciente.", None, gr.update(visible=False)
            return
        thread = uuid.uuid4().hex[:12]
        cfg = {"configurable": {"thread_id": thread}}
        feitas: list[str] = []
        for passo in fluxo_triagem.stream(
            {"paciente_id": pid, "trace_id": thread}, cfg, stream_mode="updates"
        ):
            feitas += [n for n in passo if n in ETAPAS]
            estado = fluxo_triagem.get_state(cfg).values
            yield (
                formatar_etapas(feitas),
                formatar_resultado(estado, "em andamento"),
                None,
                gr.skip(),
            )
        snapshot = fluxo_triagem.get_state(cfg)
        aguardando = bool(snapshot.next)
        status = "aguardando" if aguardando else snapshot.values["status"]
        yield (
            formatar_etapas(feitas, pausado=aguardando),
            formatar_resultado(snapshot.values, status),
            thread if aguardando else None,
            gr.update(visible=aguardando),
        )

    def decidir(aprovar: bool, thread: str | None, medico: str, comentario: str):
        if not thread:
            return gr.skip(), gr.skip(), None, gr.update(visible=False)
        if not medico.strip():
            raise gr.Error("Informe o médico responsável (nome e CRM).")
        cfg = {"configurable": {"thread_id": thread}}
        decisao = {"aprovar": aprovar, "medico": medico.strip(), "comentario": comentario.strip()}
        final = fluxo_triagem.invoke(Command(resume=decisao), cfg)
        conduta = listar_registros(conn, "condutas", thread)[-1]
        registro = (
            f"\n\n**Registro:** conduta {conduta['status']} por {conduta['medico']} "
            f"em {conduta['registrado_em'].replace('T', ' ')} (execução `{thread}`)."
        )
        feitas = [n for n in ETAPAS if n != "emitir_alertas" or final.get("alertas")]
        return (
            formatar_etapas(feitas),
            formatar_resultado(final, final["status"]) + registro,
            None,
            gr.update(visible=False),
        )

    with gr.Blocks(title="Assistente Médico | Hospital Santa Aurora") as demo:
        gr.HTML(
            f"""<div class="cabecalho">
            <h1>Hospital Santa Aurora</h1>
            <p>Assistente clínico virtual de apoio à decisão</p>
            <span class="selo">Modelo: {MODEL_NAME}</span></div>"""
        )
        with gr.Row():
            with gr.Column(scale=3):
                paciente = gr.Dropdown(opcoes, value=SEM_PACIENTE, label="Paciente")
                with gr.Tabs():
                    with gr.Tab("Dúvida clínica"):
                        chat = gr.Chatbot(
                            height=440, show_label=False, placeholder="Faça uma pergunta clínica."
                        )
                        with gr.Row():
                            entrada = gr.Textbox(
                                placeholder="Digite sua pergunta e pressione Enter",
                                show_label=False,
                                scale=5,
                            )
                            botao_enviar = gr.Button("Enviar", variant="primary", scale=1)
                        limpar = gr.Button("Limpar conversa", variant="secondary", size="sm")
                        with gr.Accordion("Trechos de protocolo consultados", open=False):
                            trechos = gr.Markdown(formatar_trechos([]))
                    with gr.Tab("Triagem automática"):
                        gr.Markdown(
                            "Executa o fluxo de triagem do paciente selecionado. A sugestão "
                            "só é registrada após a decisão do médico."
                        )
                        botao_triagem = gr.Button("Executar triagem", variant="primary")
                        with gr.Row():
                            etapas = gr.HTML()
                            resultado = gr.Markdown()
                        thread = gr.State(None)
                        with gr.Group(visible=False) as painel_decisao:
                            medico = gr.Textbox(label="Médico responsável (nome e CRM)")
                            comentario = gr.Textbox(label="Comentário (opcional)")
                            with gr.Row():
                                aprovar = gr.Button("Aprovar sugestão", variant="primary")
                                rejeitar = gr.Button("Rejeitar sugestão", variant="stop")
            with gr.Column(scale=2, elem_classes="painel"):
                with gr.Accordion("Prontuário do paciente", open=True):
                    resumo = gr.Markdown(mostrar_paciente(SEM_PACIENTE))
                gr.Markdown("**Perguntas gerais**")
                botoes = [
                    (gr.Button(e, size="sm", variant="secondary"), e) for e in EXEMPLOS_GERAIS
                ]
                gr.Markdown("**Sobre o paciente selecionado**")
                botoes += [
                    (gr.Button(e, size="sm", variant="secondary"), e) for e in EXEMPLOS_PACIENTE
                ]
                gr.Markdown(LIMITES)
        gr.Markdown(
            "Projeto acadêmico (FIAP Pós Tech). Hospital e dados fictícios: "
            "não usar na prática clínica.",
            elem_classes="rodape",
        )

        paciente.change(mostrar_paciente, paciente, resumo)
        entrada.submit(perguntar, [entrada, chat, paciente], [entrada, chat, trechos])
        botao_enviar.click(perguntar, [entrada, chat, paciente], [entrada, chat, trechos])
        limpar.click(lambda: [], None, chat)
        for botao, exemplo in botoes:
            botao.click(lambda e=exemplo: e, None, entrada)
        saidas_triagem = [etapas, resultado, thread, painel_decisao]
        botao_triagem.click(executar_triagem, paciente, saidas_triagem)
        aprovar.click(
            lambda t, m, c: decidir(True, t, m, c), [thread, medico, comentario], saidas_triagem
        )
        rejeitar.click(
            lambda t, m, c: decidir(False, t, m, c), [thread, medico, comentario], saidas_triagem
        )
    return demo


def main() -> None:
    conn = inicializar()
    indice = criar_indice()
    llm = ChatOllama(
        model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.2, num_thread=OLLAMA_NUM_THREAD
    )
    checkpointer = SqliteSaver(
        sqlite3.connect(STORAGE_DIR / "checkpoints.db", check_same_thread=False)
    )
    criar_interface(conn, indice, criar_cadeia(llm), checkpointer).launch(
        server_name="0.0.0.0", server_port=7860, theme=TEMA, css=CSS
    )


if __name__ == "__main__":
    main()
