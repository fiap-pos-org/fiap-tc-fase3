"""Interface web do assistente (Gradio).

Perguntas clínicas com o modelo ajustado, contextualizadas com o prontuário do paciente
(banco SQLite) e com trechos dos protocolos internos recuperados por busca semântica.
Uso: uv run python -m hospital_assistant.app (ou via docker compose)
"""

import gradio as gr
from langchain_ollama import ChatOllama

from hospital_assistant.assistant.qa import montar_contexto, responder
from hospital_assistant.config import MODEL_NAME, OLLAMA_BASE_URL, OLLAMA_NUM_THREAD
from hospital_assistant.db.database import buscar_paciente, inicializar, listar_pacientes
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
    "Há algum valor crítico nos exames?",
    "Resuma o caso deste paciente.",
]

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


def criar_interface(conn, indice, llm) -> gr.Blocks:
    pacientes = listar_pacientes(conn)
    opcoes = [SEM_PACIENTE] + [
        f"{p['id']} | {p['idade']} anos, {p['sexo']} | {p['diagnostico']}" for p in pacientes
    ]

    def _paciente_id(opcao: str) -> str | None:
        return None if opcao == SEM_PACIENTE else opcao.split(" | ")[0]

    def mostrar_paciente(opcao: str) -> str:
        pid = _paciente_id(opcao)
        if pid is None:
            return "Sem paciente selecionado: as respostas usam só os protocolos."
        return f"```\n{format_patient_context(buscar_paciente(conn, pid))}\n```"

    def enviar(mensagem: str, historico: list, opcao: str):
        if not mensagem.strip():
            yield "", historico, gr.skip()
            return
        historico = historico + [{"role": "user", "content": mensagem}]
        contexto = montar_contexto(mensagem, indice, conn, _paciente_id(opcao))
        trechos = formatar_trechos(contexto.trechos)
        resposta = ""
        for resposta in responder(contexto, llm):
            yield "", historico + [{"role": "assistant", "content": resposta}], trechos
        yield "", historico + [{"role": "assistant", "content": formatar(resposta)}], trechos

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
                chat = gr.Chatbot(
                    height=480, show_label=False, placeholder="Faça uma pergunta clínica."
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
        entrada.submit(enviar, [entrada, chat, paciente], [entrada, chat, trechos])
        botao_enviar.click(enviar, [entrada, chat, paciente], [entrada, chat, trechos])
        limpar.click(lambda: [], None, chat)
        for botao, exemplo in botoes:
            botao.click(lambda e=exemplo: e, None, entrada)
    return demo


def main() -> None:
    conn = inicializar()
    indice = criar_indice()
    llm = ChatOllama(
        model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.2, num_thread=OLLAMA_NUM_THREAD
    )
    criar_interface(conn, indice, llm).launch(
        server_name="0.0.0.0", server_port=7860, theme=TEMA, css=CSS
    )


if __name__ == "__main__":
    main()
