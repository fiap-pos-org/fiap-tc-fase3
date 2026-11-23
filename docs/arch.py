# Arquitetura do assistente médico (spec da skill architecture-diagrams):
#   <skill>/.venv/bin/python <skill>/scripts/diagram.py docs/arch.py docs/arch.gif
W, H = 1300, 880
A, B, T = 250, 450, 730          # linhas: requisição, dados e segurança, treino
X0, X1, X2, X3, X4 = 90, 320, 580, 840, 1100

GROUPS = [
    dict(style='plain', x=200, y=40, w=1060, h=540, label='VPS · aurora.williamdevops.com.br'),
    dict(style='compose', x=450, y=110, w=790, h=450, label='container assistente-medico'),
    dict(style='soft', x=200, y=620, w=1060, h=240, label='Treino e publicação do modelo (offline)'),
]

NODES = [
    dict(icon='Users', x=X0, y=A, label=['Médico', 'navegador']),
    dict(icon='traefikproxy', x=X1, y=A, label=['Traefik', "HTTPS · Let's Encrypt"]),
    dict(icon='fastapi', x=X2, y=A, label=['FastAPI', 'API + front Tabler · login'], chips=['python']),
    dict(icon='langchain', x=X3, y=A, label=['LangGraph + LangChain', 'triagem · dúvida clínica']),
    dict(icon='ollama', x=X4, y=A, label=['Ollama', 'Qwen2.5-1.5B + QLoRA · bge-m3'], chips=['qwen']),

    dict(icon='guard-role', x=X2, y=B, size=50, label=['Travas de segurança', 'entrada · fontes · aprovação']),
    dict(icon='sqlite', x=X3, y=B, label=['SQLite', 'prontuário · auditoria']),

    dict(icon='python', x=X1, y=T, size=50, label=['Pipeline de dados', 'sintéticos · anonimização']),
    dict(icon='googlecolab', x=X2, y=T, label=['Google Colab', 'fine-tuning QLoRA'], chips=['nvidia']),
    dict(icon='huggingface', x=X3, y=T, label=['Hugging Face', 'modelo GGUF Q4_K_M']),
]

EDGES = [
    dict(points=[(X0 + 30, A), (X1 - 30, A)], label='HTTPS', label_at=(205, A)),
    dict(points=[(X1 + 30, A), (X2 - 30, A)], label='proxy', label_at=(450, A)),
    dict(points=[(X2 + 30, A), (X3 - 30, A)], label='invoca', label_at=(710, A)),
    dict(points=[(X3 + 30, A), (X4 - 30, A)], label='LCEL', label_at=(970, A)),
    dict(points=[(X2, A + 100), (X2, B - 30)], label='valida', label_at=(X2, 385)),
    dict(points=[(X3, A + 70), (X3, B - 32)], label='SQL', label_at=(X3, 375)),
    dict(points=[(X2 + 30, B), (X3 - 32, B)], label='auditoria', label_at=(710, B)),
    dict(points=[(X1 + 30, T), (X2 - 30, T)], label='dataset', label_at=(450, T)),
    dict(points=[(X2 + 30, T), (X3 - 30, T)], label='GGUF', label_at=(710, T)),
    dict(points=[(X3 + 30, T), (1190, T), (1190, A), (X4 + 30, A)], label='ollama create', label_at=(1190, 520)),
]
