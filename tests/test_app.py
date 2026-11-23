from hospital_assistant.app import formatar
from hospital_assistant.finetune.export import GGUF_NOME, modelfile
from hospital_assistant.prompts import SYSTEM_PROMPT


def test_formatar_destaca_fontes():
    saida = formatar("Antibiótico em até 1 hora.\n\nFontes: [PROT-SEPSE-01 §4]")
    assert saida.startswith("Antibiótico em até 1 hora.")
    assert '<div class="fontes"><b>Fontes:</b> [PROT-SEPSE-01 §4]</div>' in saida


def test_formatar_sem_fontes_mantem_texto():
    assert formatar("Resposta sem fonte.") == "Resposta sem fonte."


def test_modelfile_usa_gguf_prompt_do_treino_e_stop_tokens():
    mf = modelfile()
    assert mf.startswith(f"FROM /models/gguf/{GGUF_NOME}")
    assert SYSTEM_PROMPT in mf
    assert 'PARAMETER stop "<|im_end|>"' in mf
