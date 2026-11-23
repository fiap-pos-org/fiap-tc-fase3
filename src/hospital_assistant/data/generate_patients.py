"""Gera prontuários sintéticos BRUTOS, com dados pessoais fictícios de propósito.

Os dados pessoais (nome, CPF, telefone, endereço etc.) existem para que o pipeline de
anonimização tenha o que remover. Nada aqui corresponde a pessoas reais.
"""

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

from faker import Faker

from hospital_assistant.data.exames import EXAMES_LAB

# Cada cenário: diagnóstico, protocolo de referência, faixas (min, max) dos exames
# laboratoriais (cruzam o limite crítico em parte dos pacientes) e exames pendentes.
CENARIOS = [
    {
        "diagnostico": "Sepse de foco urinário",
        "protocolo": "PROT-SEPSE-01",
        "setor": "Pronto-socorro",
        "queixa": "febre, calafrios e confusão mental há 1 dia",
        "labs": {
            "lactato": (1.8, 6.5),
            "leucocitos": (12_000, 28_000),
            "creatinina": (1.0, 2.8),
            "plaquetas": (60_000, 220_000),
            "pcr": (40, 280),
        },
        "pendentes": ["Hemocultura (2 pares)", "Urocultura"],
    },
    {
        "diagnostico": "Dor torácica a esclarecer",
        "protocolo": "PROT-DORTORAC-01",
        "setor": "Pronto-socorro",
        "queixa": "dor torácica em aperto irradiando para o braço esquerdo há 2 horas",
        "labs": {"troponina": (5, 180), "potassio": (3.4, 5.2), "creatinina": (0.7, 1.4)},
        "pendentes": ["Troponina ultrassensível (2ª amostra)", "Radiografia de tórax"],
        "laudos": {
            "Eletrocardiograma": [
                "Ritmo sinusal, sem alterações agudas do segmento ST.",
                "Ritmo sinusal. Infradesnivelamento do ST de V4 a V6.",
            ]
        },
    },
    {
        "diagnostico": "AVC isquêmico agudo",
        "protocolo": "PROT-AVC-01",
        "setor": "Unidade de AVC",
        "queixa": "fraqueza súbita no braço direito e fala arrastada há 90 minutos",
        "labs": {"glicose": (90, 220), "plaquetas": (80_000, 300_000), "inr": (0.9, 2.4)},
        "pendentes": ["Angiotomografia de crânio e pescoço"],
        "laudos": {
            "Tomografia de crânio sem contraste": ["Sem sinais de hemorragia intracraniana."]
        },
    },
    {
        "diagnostico": "Pneumonia adquirida na comunidade",
        "protocolo": "PROT-PAC-01",
        "setor": "Enfermaria de clínica médica",
        "queixa": "tosse produtiva, febre e dispneia há 4 dias",
        "labs": {
            "leucocitos": (9_000, 22_000),
            "ureia": (25, 95),
            "pcr": (30, 250),
            "sodio": (128, 142),
        },
        "pendentes": ["Radiografia de tórax (controle)"],
        "laudos": {
            "Radiografia de tórax": [
                "Opacidade em base direita, compatível com processo consolidativo."
            ]
        },
    },
    {
        "diagnostico": "Cetoacidose diabética",
        "protocolo": "PROT-CAD-01",
        "setor": "UTI adulto",
        "queixa": "vômitos, dor abdominal e muita sede há 2 dias",
        "labs": {
            "glicose": (280, 620),
            "potassio": (2.6, 6.4),
            "ph": (6.95, 7.28),
            "bicarbonato": (6, 17),
        },
        "pendentes": ["Cetonemia", "Urina tipo 1"],
    },
    {
        "diagnostico": "Emergência hipertensiva",
        "protocolo": "PROT-HAS-01",
        "setor": "Pronto-socorro",
        "queixa": "cefaleia intensa e visão turva, PA 220/130 mmHg na triagem",
        "labs": {"creatinina": (0.8, 2.2), "potassio": (3.3, 5.0)},
        "pendentes": ["Troponina ultrassensível", "Fundoscopia"],
    },
    {
        "diagnostico": "Pós-operatório de artroplastia de quadril",
        "protocolo": "PROT-TEV-01",
        "setor": "Enfermaria cirúrgica",
        "queixa": "internação eletiva para artroplastia total de quadril",
        "labs": {
            "plaquetas": (45_000, 260_000),
            "hemoglobina": (6.5, 12.5),
            "creatinina": (0.6, 1.8),
        },
        "pendentes": ["Hemograma de controle"],
    },
    {
        "diagnostico": "Hipoglicemia em diabético insulinizado",
        "protocolo": "PROT-HIPOGLI-01",
        "setor": "Enfermaria de clínica médica",
        "queixa": "sudorese e confusão após dose de insulina, glicemia capilar de 48 mg/dL",
        "labs": {"glicose": (38, 85), "creatinina": (0.8, 2.6)},
        "pendentes": ["Hemoglobina glicada"],
    },
    {
        "diagnostico": "Lesão renal aguda",
        "protocolo": "PROT-IRA-01",
        "setor": "Enfermaria de clínica médica",
        "queixa": "redução do volume urinário e inchaço nas pernas há 3 dias",
        "labs": {
            "creatinina": (1.8, 6.0),
            "ureia": (80, 220),
            "potassio": (4.8, 7.2),
            "ph": (7.10, 7.36),
        },
        "pendentes": ["Ultrassonografia de rins e vias urinárias"],
    },
    {
        "diagnostico": "Anemia a esclarecer",
        "protocolo": "PROT-VCRIT-01",
        "setor": "Enfermaria de clínica médica",
        "queixa": "cansaço progressivo e fezes escurecidas há 1 semana",
        "labs": {"hemoglobina": (5.5, 9.5), "plaquetas": (15_000, 180_000), "inr": (1.0, 6.5)},
        "pendentes": ["Endoscopia digestiva alta"],
    },
]

ALERGIAS = [
    "penicilina",
    "dipirona",
    "sulfa",
    "ácido acetilsalicílico",
    "contraste iodado",
    "látex",
]
MEDICACOES_USO = [
    "losartana 50 mg 12/12h",
    "metformina 850 mg 12/12h",
    "anlodipino 5 mg 1x/dia",
    "sinvastatina 20 mg à noite",
    "insulina NPH 20 UI pela manhã",
    "varfarina 5 mg 1x/dia",
    "omeprazol 20 mg em jejum",
    "glibenclamida 5 mg 1x/dia",
    "levotiroxina 50 mcg em jejum",
]
PARENTESCOS = ["filha", "filho", "esposa", "marido", "irmã", "neto"]


def _arredonda(codigo: str, valor: float) -> float:
    if codigo in {"plaquetas", "leucocitos"}:
        return float(round(valor, -3))
    if codigo == "ph":
        return round(valor, 2)
    return round(valor, 1) if valor < 100 else float(round(valor))


def _gera_paciente(fake: Faker, rng: random.Random, n: int, hoje: date) -> dict:
    cenario = CENARIOS[n % len(CENARIOS)]
    sexo = rng.choice(["F", "M"])
    primeiro = fake.first_name_female() if sexo == "F" else fake.first_name_male()
    nome = f"{primeiro} {fake.last_name()} {fake.last_name()}"
    nascimento = fake.date_of_birth(minimum_age=25, maximum_age=90)
    idade = (hoje - nascimento).days // 365
    internacao = datetime.combine(hoje - timedelta(days=rng.randint(0, 5)), datetime.min.time())
    internacao += timedelta(hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
    telefone = fake.cellphone_number()
    acompanhante = f"{fake.first_name()} {fake.last_name()}"
    parentesco = rng.choice(PARENTESCOS)
    endereco = fake.street_address()
    cidade = fake.city()
    cpf = fake.cpf()

    exames = []
    for codigo, (lo, hi) in cenario["labs"].items():
        exame = EXAMES_LAB[codigo]
        exames.append(
            {
                "codigo": codigo,
                "nome": exame.nome,
                "valor": _arredonda(codigo, rng.uniform(lo, hi)),
                "unidade": exame.unidade,
                "data": (internacao + timedelta(hours=1)).isoformat(timespec="minutes"),
                "status": "resultado",
            }
        )
    for nome_exame, laudos in cenario.get("laudos", {}).items():
        exames.append(
            {
                "nome": nome_exame,
                "laudo": rng.choice(laudos),
                "data": (internacao + timedelta(hours=2)).isoformat(timespec="minutes"),
                "status": "resultado",
            }
        )
    for nome_exame in cenario["pendentes"]:
        exames.append(
            {
                "nome": nome_exame,
                "data_solicitacao": internacao.isoformat(timespec="minutes"),
                "status": "pendente",
            }
        )

    nota = (
        f"Paciente {nome}, {idade} anos, CPF {cpf}, deu entrada em "
        f"{internacao:%d/%m/%Y às %H:%M} com {cenario['queixa']}. "
        f"Acompanhado(a) por {parentesco} {acompanhante}, contato {telefone}. "
        f"Reside em {endereco}, {cidade}."
    )
    evolucao = (
        f"Evolução: {nome.split()[0]} segue em observação no setor {cenario['setor']}. "
        f"Hipótese diagnóstica: {cenario['diagnostico']}. "
        f"Familiar ({acompanhante}) orientado(a) sobre o quadro por telefone ({telefone})."
    )
    return {
        "atendimento": f"AT-{hoje.year}-{10000 + n}",
        "nome": nome,
        "cpf": cpf,
        "cns": fake.numerify("7## #### #### ####"),
        "data_nascimento": nascimento.isoformat(),
        "sexo": sexo,
        "telefone": telefone,
        "email": fake.free_email(),
        "endereco": f"{endereco}, {cidade}",
        "data_internacao": internacao.isoformat(timespec="minutes"),
        "setor": cenario["setor"],
        "diagnostico": cenario["diagnostico"],
        "protocolo": cenario["protocolo"],
        "alergias": rng.sample(ALERGIAS, k=rng.choice([0, 0, 1, 1, 2])),
        "medicacoes_em_uso": rng.sample(MEDICACOES_USO, k=rng.randint(0, 3)),
        "exames": exames,
        "nota_admissao": nota,
        "evolucao": evolucao,
    }


def generate_patients(n: int = 200, seed: int = 42, hoje: date = date(2026, 11, 20)) -> list[dict]:
    fake = Faker("pt_BR")
    fake.seed_instance(seed)
    rng = random.Random(seed)
    return [_gera_paciente(fake, rng, i, hoje) for i in range(n)]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
