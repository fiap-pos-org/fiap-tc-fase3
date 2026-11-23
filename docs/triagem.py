# Fluxo de triagem no LangGraph (spec da skill architecture-diagrams):
#   <skill>/.venv/bin/python <skill>/scripts/diagram.py docs/triagem.py docs/triagem.gif
W, H = 1280, 580
A, B = 150, 440                  # linha 1: verificações em código; linha 2: modelo, segurança e decisão
XS = [140, 360, 580, 800]
YS = [140, 360, 680, 900, 1120]

GROUPS = [
    dict(style='tier_data', x=40, y=40, w=900, h=240, label='Verificações em código'),
    dict(style='tier_app', x=40, y=360, w=440, h=200, label='Modelo ajustado', label_right=True),
    dict(style='tier_edge', x=560, y=360, w=680, h=200, label='Segurança e decisão humana'),
]

NODES = [
    dict(icon='sqlite', x=XS[0], y=A, label=['Carregar prontuário', 'consulta no banco']),
    dict(icon='check-tests', x=XS[1], y=A, size=52, label=['Exames pendentes', 'sem resultado']),
    dict(icon='guard-path', x=XS[2], y=A, size=52, label=['Valores críticos', 'regra do PROT-VCRIT-01']),
    dict(icon='guard-events', x=XS[3], y=A, size=52, label=['Alertas à equipe', 'gravados no banco']),

    dict(icon='ollama', x=YS[0], y=B, label=['Protocolo do caso', 'prontuário + busca bge-m3']),
    dict(icon='qwen', x=YS[1], y=B, label=['Sugestão', 'Qwen2.5 + QLoRA redige']),
    dict(icon='check-scan', x=YS[2], y=B, size=52, label=['Validação', 'fontes · alergia · dose']),
    dict(icon='User', x=YS[3], y=B, label=['Aprovação do médico', 'nome e CRM · interrupt']),
    dict(icon='sqlite', x=YS[4], y=B, label=['Conduta registrada', 'aprovada ou rejeitada']),
]

EDGES = [
    dict(points=[(XS[0] + 30, A), (XS[1] - 30, A)]),
    dict(points=[(XS[1] + 30, A), (XS[2] - 30, A)]),
    dict(points=[(XS[2] + 30, A), (XS[3] - 30, A)], label='se crítico', label_at=(690, A)),
    dict(points=[(XS[2], A + 70), (XS[2], 320), (YS[0], 320), (YS[0], B - 32)], label='segue', label_at=(360, 320)),
    dict(points=[(YS[0] + 30, B), (YS[1] - 30, B)], label='trechos', label_at=(250, B)),
    dict(points=[(YS[1] + 30, B), (YS[2] - 30, B)], label='texto', label_at=(520, B)),
    dict(points=[(YS[2] + 30, B), (YS[3] - 30, B)], label='aprovada', label_at=(790, B)),
    dict(points=[(YS[3] + 30, B), (YS[4] - 30, B)], label='decide', label_at=(1010, B)),
    dict(points=[(YS[2], B - 32), (YS[2], 390), (YS[4], 390), (YS[4], B - 32)],
         label='reprovada: bloqueada', label_at=(900, 390), color='#E7157B'),
]
