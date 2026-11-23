from hospital_assistant.data.curate import curate, split_por_grupo
from hospital_assistant.data.exames import EXAMES_LAB
from hospital_assistant.prompts import fmt_num

REFS = {"PROT-SEPSE-01 §4"}


def _ex(user: str, resposta: str, grupo: str = "g1", tipo: str = "faq") -> dict:
    return {
        "id": user,
        "grupo": grupo,
        "tipo": tipo,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": resposta},
        ],
    }


RESPOSTA_OK = "Antibiótico em até 1 hora do reconhecimento.\n\nFontes: [PROT-SEPSE-01 §4]"


def test_curadoria_remove_duplicata_fonte_invalida_e_dado_pessoal():
    exemplos = [
        _ex("Prazo do antibiótico?", RESPOSTA_OK),
        _ex("Prazo do antibiótico?", RESPOSTA_OK),  # duplicado
        _ex("Outra?", "Resposta longa o bastante sem fonte nenhuma aqui."),  # sem Fontes
        _ex("E essa?", "Resposta longa o bastante.\n\nFontes: [PROT-INEXISTENTE §9]"),
        _ex("Ligar para 11 98765-4321?", RESPOSTA_OK),  # telefone
        _ex("Caso da Ana Teste?", RESPOSTA_OK),  # nome real conhecido
    ]
    aprovados, descartes = curate(exemplos, REFS, nomes_reais=["Ana Teste"])
    assert len(aprovados) == 1
    assert descartes == {"duplicado": 1, "fonte_invalida": 2, "dado_pessoal": 2}


def test_split_mantem_grupo_inteiro_no_mesmo_split():
    exemplos = [_ex(f"q{g}-{i}", RESPOSTA_OK, grupo=f"g{g}") for g in range(30) for i in range(3)]
    splits = split_por_grupo(exemplos)
    grupos = {nome: {ex["grupo"] for ex in linhas} for nome, linhas in splits.items()}
    assert not grupos["train"] & grupos["val"]
    assert not grupos["train"] & grupos["test"]
    assert not grupos["val"] & grupos["test"]
    assert sum(len(v) for v in splits.values()) == len(exemplos)


def test_limites_de_valor_critico_seguem_o_protocolo():
    assert not EXAMES_LAB["potassio"].is_critico(6.2)  # > 6,2
    assert EXAMES_LAB["potassio"].is_critico(6.3)
    assert EXAMES_LAB["potassio"].is_critico(2.7)  # < 2,8
    assert EXAMES_LAB["lactato"].is_critico(4.0)  # ≥ 4,0
    assert not EXAMES_LAB["creatinina"].is_critico(9.0)  # sem valor crítico definido


def test_formatacao_numerica_pt_br():
    assert fmt_num(2.8) == "2,8"
    assert fmt_num(203000.0) == "203.000"
    assert fmt_num(7.25) == "7,25"
