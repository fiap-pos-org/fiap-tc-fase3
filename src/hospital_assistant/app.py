"""Interface web do assistente (Gradio).

- Dúvida clínica: perguntas com o modelo ajustado, contextualizadas com o prontuário do
  paciente (SQLite) e com trechos de protocolo (busca semântica). Fluxo B do LangGraph.
- Triagem automática: fluxo A do LangGraph (pendências, valores críticos, alertas, sugestão,
  validação de segurança) com pausa para aprovação do médico.

Uso: uv run python -m hospital_assistant.app (ou via docker compose)
"""

import html
import json
import logging
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
from hospital_assistant.observability.auditoria import listar_execucoes, trilha
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
.execucao { font-size: .75rem; opacity: .6; font-family: monospace; }
.evento { padding: 8px 12px; margin: 6px 0; border-radius: 6px;
  border-left: 4px solid #1b6f7f; border: 1px solid var(--border-color-primary); }
.evento .hora { font-family: monospace; font-size: .8rem; opacity: .7; }
"""

TEMA = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

SEM_PACIENTE = "Nenhum (pergunta geral)"


def formatar(resposta: str) -> str:
    """Escapa o texto do modelo (ex.: "pH < 7,30" não pode virar HTML) e destaca as fontes."""
    if "Fontes:" not in resposta:
        return html.escape(resposta)
    texto, fontes = resposta.rsplit("Fontes:", 1)
    texto, fontes = html.escape(texto.strip()), html.escape(fontes.strip())
    return f'{texto}\n\n<div class="fontes"><b>Fontes:</b> {fontes}</div>'


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
        # A resposta só aparece depois de validada: mostrar o texto enquanto é gerado
        # exporia conteúdo que a validação ainda pode reprovar.
        aguarde = "<em>Analisando a pergunta e consultando protocolos...</em>"
        yield "", historico + [{"role": "assistant", "content": aguarde}], gr.skip()
        estado = fluxo_duvida.invoke(entrada)
        final = formatar(estado["resposta"])
        if estado["categoria"] != "clinica":
            final += '\n\n<div class="aviso">Respondido pela trava de segurança, sem consultar o modelo.</div>'
        elif not estado["validacao"]["aprovada"]:
            final += '\n\n<div class="aviso">A resposta do modelo foi reprovada na validação e substituída.</div>'
        final += f'\n\n<span class="execucao">Execução {entrada["trace_id"]}</span>'
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
            formatar_resultado(snapshot.values, status)
            + f'\n\n<span class="execucao">Execução {thread}</span>',
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

    # ------------------------------------------------------------ auditoria
    def tabela_execucoes():
        return [
            [
                e["inicio"].replace("T", " ")[:19],
                e["trace_id"],
                e["fluxo"],
                e["paciente_id"] or "-",
                e["etapas"],
            ]
            for e in listar_execucoes(conn)
        ]

    def mostrar_trilha(trace_id: str) -> str:
        eventos = trilha(conn, trace_id.strip())
        if not eventos:
            return "Execução não encontrada."
        blocos = []
        for e in eventos:
            dados = html.escape(json.dumps(e["dados"], ensure_ascii=False, indent=2))
            blocos.append(
                f'<div class="evento"><span class="hora">{e["criado_em"][11:23]}</span> '
                f"<b>{html.escape(e['etapa'])}</b> ({e['duracao_ms']} ms)"
                f"<details><summary>dados</summary><pre>{dados}</pre></details></div>"
            )
        return "".join(blocos)

    def selecionar_execucao(evt: gr.SelectData, tabela):
        trace_id = tabela.iloc[evt.index[0], 1]
        return trace_id, mostrar_trilha(trace_id)

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
                    with gr.Tab("Auditoria") as aba_auditoria:
                        gr.Markdown(
                            "Trilha de cada execução dos fluxos: entradas e saídas de cada etapa, "
                            "texto enviado ao modelo, fontes, validação e decisão do médico."
                        )
                        execucoes = gr.Dataframe(
                            headers=["Início", "Execução", "Fluxo", "Paciente", "Etapas"],
                            interactive=False,
                            wrap=True,
                        )
                        with gr.Row():
                            trace_busca = gr.Textbox(label="Execução (trace_id)", scale=4)
                            botao_trilha = gr.Button("Ver trilha", scale=1)
                        trilha_html = gr.HTML()
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
        aba_auditoria.select(tabela_execucoes, None, execucoes)
        execucoes.select(selecionar_execucao, execucoes, [trace_busca, trilha_html])
        botao_trilha.click(mostrar_trilha, trace_busca, trilha_html)
        trace_busca.submit(mostrar_trilha, trace_busca, trilha_html)
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # sem ruído de cada chamada HTTP
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
