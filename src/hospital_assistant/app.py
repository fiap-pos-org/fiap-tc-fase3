"""Interface web do assistente (Gradio).

Nesta versão: chat direto com o modelo ajustado servido pelo Ollama.
Uso: uv run python -m hospital_assistant.app (ou via docker compose)
"""

import gradio as gr
from langchain_ollama import ChatOllama

from hospital_assistant.config import MODEL_NAME, OLLAMA_BASE_URL, OLLAMA_NUM_THREAD
from hospital_assistant.prompts import SYSTEM_PROMPT, format_user_message

llm = ChatOllama(
    model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.2, num_thread=OLLAMA_NUM_THREAD
)

EXEMPLOS = [
    "Qual o prazo para iniciar o antibiótico na sepse?",
    "Quais os critérios de identificação da cetoacidose diabética?",
    "Quais valores de potássio são críticos?",
    "Prescreva ceftriaxona 1 g EV para o paciente.",
    "Qual o protocolo do hospital para grandes queimados?",
]

LIMITES = """
**Limites de atuação**

- Não prescreve nem altera prescrições
- Toda sugestão depende de validação médica
- Responde só com base nos protocolos internos
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


def formatar(resposta: str) -> str:
    """Destaca a linha de fontes num bloco próprio."""
    if "Fontes:" not in resposta:
        return resposta
    texto, fontes = resposta.rsplit("Fontes:", 1)
    return f'{texto.strip()}\n\n<div class="fontes"><b>Fontes:</b> {fontes.strip()}</div>'


def responder(mensagem: str, historico: list):
    """Cada pergunta é independente (o modelo foi treinado em turno único)."""
    historico = historico + [{"role": "user", "content": mensagem}]
    mensagens = [("system", SYSTEM_PROMPT), ("human", format_user_message(mensagem))]
    resposta = ""
    for pedaco in llm.stream(mensagens):
        resposta += pedaco.content
        yield "", historico + [{"role": "assistant", "content": resposta}]
    yield "", historico + [{"role": "assistant", "content": formatar(resposta)}]


with gr.Blocks(title="Assistente Médico | Hospital Santa Aurora") as demo:
    gr.HTML(
        f"""<div class="cabecalho">
        <h1>Hospital Santa Aurora</h1>
        <p>Assistente clínico virtual de apoio à decisão</p>
        <span class="selo">Modelo: {MODEL_NAME}</span></div>"""
    )
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(
                height=520, show_label=False, placeholder="Faça uma pergunta clínica."
            )
            with gr.Row():
                entrada = gr.Textbox(
                    placeholder="Digite sua pergunta e pressione Enter",
                    show_label=False,
                    scale=5,
                )
                enviar = gr.Button("Enviar", variant="primary", scale=1)
            limpar = gr.Button("Limpar conversa", variant="secondary", size="sm")
        with gr.Column(scale=1, elem_classes="painel"):
            gr.Markdown("**Perguntas de exemplo**")
            botoes = [gr.Button(e, size="sm", variant="secondary") for e in EXEMPLOS]
            gr.Markdown(LIMITES)
    gr.Markdown(
        "Projeto acadêmico (FIAP Pós Tech). Hospital e dados fictícios: não usar na prática clínica.",
        elem_classes="rodape",
    )

    entrada.submit(responder, [entrada, chat], [entrada, chat])
    enviar.click(responder, [entrada, chat], [entrada, chat])
    limpar.click(lambda: [], None, chat)
    for botao, exemplo in zip(botoes, EXEMPLOS, strict=True):
        botao.click(lambda e=exemplo: e, None, entrada)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=TEMA, css=CSS)
