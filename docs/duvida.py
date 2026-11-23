# Fluxo de dúvida clínica no LangGraph + cadeia LangChain (spec da skill architecture-diagrams):
#   <skill>/.venv/bin/python <skill>/scripts/diagram.py docs/duvida.py docs/duvida.gif
W, H = 1280, 580
A, B = 150, 440                  # linha 1: entrada e travas; linha 2: LangChain, modelo e validação
XS = [140, 400, 700, 1000]
YS = [140, 400, 680, 940, 1160]

GROUPS = [
    dict(style='tier_edge', x=40, y=40, w=1200, h=240, label='Entrada e travas (sem chamar o modelo)'),
    dict(style='tier_app', x=40, y=360, w=520, h=200, label='LangChain + modelo ajustado', label_right=True),
    dict(style='tier_data', x=600, y=360, w=640, h=200, label='Validação de saída'),
]

NODES = [
    dict(icon='User', x=XS[0], y=A, label=['Pergunta do médico', 'com ou sem paciente']),
    dict(icon='guard-path', x=XS[1], y=A, size=52, label=['Trava de entrada', 'prescrição · injeção · escopo']),
    dict(icon='guard-events', x=XS[2], y=A, size=52, label=['Resposta padrão', 'recusa, sem modelo']),
    dict(icon='check-tests', x=XS[3], y=A, size=52, label=['Valores críticos', 'ferramenta em código']),

    dict(icon='langchain', x=YS[0], y=B, label=['Montar contexto', 'prontuário + busca bge-m3']),
    dict(icon='qwen', x=YS[1], y=B, label=['Gerar resposta', 'LCEL · ChatOllama · QLoRA']),
    dict(icon='check-scan', x=YS[2], y=B, size=52, label=['Validar resposta', 'fontes · alergia · dose']),
    dict(icon='User', x=YS[3], y=B, label=['Resposta com fontes', '[PROT-SEPSE-01 §4]']),
    dict(icon='guard-role', x=YS[4], y=B, size=52, label=['Resposta segura', 'original só na auditoria']),
]

EDGES = [
    dict(points=[(XS[0] + 30, A), (XS[1] - 30, A)]),
    dict(points=[(XS[1] + 30, A), (XS[2] - 30, A)], label='bloqueada', label_at=(550, A)),
    dict(points=[(XS[1], A - 32), (XS[1], 80), (XS[3], 80), (XS[3], A - 32)],
         label='críticos do paciente', label_at=(850, 80)),
    dict(points=[(XS[1], A + 70), (XS[1], 320), (YS[0], 320), (YS[0], B - 32)], label='segue', label_at=(270, 320)),
    dict(points=[(YS[0] + 30, B), (YS[1] - 30, B)], label='trechos', label_at=(270, B)),
    dict(points=[(YS[1] + 30, B), (YS[2] - 30, B)], label='texto', label_at=(540, B)),
    dict(points=[(YS[2] + 30, B), (YS[3] - 30, B)], label='aprovada', label_at=(810, B)),
    dict(points=[(YS[2], B - 32), (YS[2], 390), (YS[4], 390), (YS[4], B - 32)],
         label='reprovada', label_at=(1050, 390), color='#E7157B'),
]
