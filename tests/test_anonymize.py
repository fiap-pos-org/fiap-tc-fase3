from collections import Counter

from hospital_assistant.data.anonymize import anonymize_patient, anonymize_text
from hospital_assistant.data.generate_patients import generate_patients


def test_remove_documentos_e_contatos():
    texto = (
        "CPF 123.456.789-09, CNS 712 3456 7890 1234, tel (11) 98765-4321, "
        "+55 21 91234-5678, email fulano.tal@gmail.com, CEP 13010-000."
    )
    anon = anonymize_text(texto)
    for marcador in ("[CPF]", "[CNS]", "[TELEFONE]", "[EMAIL]", "[CEP]"):
        assert marcador in anon
    assert "123.456" not in anon and "98765" not in anon and "@" not in anon


def test_identificador_conhecido_vira_pseudonimo():
    anon = anonymize_text("Paciente Maria Souza Lima refere dor.", {"Maria Souza Lima": "PAC-0001"})
    assert anon == "Paciente PAC-0001 refere dor."


def test_ner_pega_nome_e_local_desconhecidos():
    anon = anonymize_text("Conversei com o Sr. Roberto Carlos da Silva, que mora em Campinas.")
    assert "Roberto" not in anon and "Campinas" not in anon
    assert "[NOME]" in anon and "[LOCAL]" in anon


def test_ner_nao_apaga_termos_clinicos_no_inicio_de_frase():
    texto = "Hipótese diagnóstica: sepse. Familiar orientado."
    assert anonymize_text(texto) == texto


def test_prontuario_anonimizado_nao_tem_campos_identificadores():
    bruto = generate_patients(n=1)[0]
    anon = anonymize_patient(bruto, "PAC-0001", Counter())
    for campo in ("nome", "cpf", "cns", "telefone", "email", "endereco", "data_nascimento"):
        assert campo not in anon
    assert anon["id"] == "PAC-0001"
    assert isinstance(anon["idade"], int)
    texto = anon["nota_admissao"] + anon["evolucao"]
    assert (
        bruto["nome"] not in texto and bruto["cpf"] not in texto and bruto["telefone"] not in texto
    )
