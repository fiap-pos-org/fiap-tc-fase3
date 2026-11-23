"""Trava de entrada: classifica a pergunta antes de chamar o modelo (PROT-SEGMED-01).

Pedidos de prescrição e tentativas de burlar as regras recebem a recusa padrão sem passar
pelo modelo: a garantia não depende de o modelo obedecer. Perguntas muito distantes de
qualquer trecho dos protocolos recebem "não encontrei". O que passar segue para o modelo,
que também foi treinado para recusar, e para a validação de saída.
"""

import re

# Similaridade (cosseno, bge-m3) mínima entre a pergunta e o trecho mais próximo. Calibrado
# com perguntas do treino: 0,42 bloqueia 1,2% das perguntas legítimas (FAQ) e pega 7 de 22
# perguntas sem base/fora de escopo. Conservador de propósito: o restante é coberto pelo
# modelo (treinado para recusar) e pela validação de fontes na saída.
LIMIAR_SEM_BASE = 0.42

_INJECAO = re.compile(
    r"ignore|ignora|esque[cç]a|desconsidere|voc[eê] agora [eé]|finja|finge|modo (?:admin|desenvolvedor)"
    r"|sem (?:passar por )?valida[cç][aã]o|valida[cç][aã]o (?:humana )?desativada|system\s*:"
    r"|sem avisos|j[aá] (?:foi )?aprovad|autorizo voc[eê]",
    re.IGNORECASE,
)
# Pedido de execução: verbo no imperativo (ou "deixar prescrito") + alvo de prescrição.
# Perguntas sobre o assunto ("o que deve constar numa receita?") não têm verbo de ação.
_ACAO = re.compile(
    r"\b(?:prescrev[aeo]|deix[ae]r? prescrit\w*|emit[ae]|fa[cçz]a?|libera|liber[ae]|coloc[ae]"
    r"|lan[cç]a|lance|inclu[ai]|suspend[ae]|troc[ae]|alter[ae]|aument[ae]|reduz[ae]?|mand[ae]"
    r"|aplic[ae]|apliqu[ae]|administr[ae]|j[aá] coloca)\b",
    re.IGNORECASE,
)
_ALVO = re.compile(
    r"prescri[cçt]|receita|\bdose\b|ampola|na veia|bomba de infus|no sistema|"
    r"\b\d+(?:[.,]\d+)?\s?(?:mg|g|mcg|UI|mL)\b",
    re.IGNORECASE,
)
_PRESCREVA = re.compile(r"\bprescrev[aeo]\b", re.IGNORECASE)

RESPOSTA_PRESCRICAO = (
    "Não posso prescrever, emitir receita nem alterar prescrições: prescrição é ato médico "
    "privativo. Posso apoiar sua decisão com o que os protocolos recomendam, e qualquer sugestão "
    "fica pendente de validação médica. A prescrição deve ser feita por você no sistema, após "
    "conferir alergias, medicações em uso e função renal.\n\n"
    "Fontes: [PROT-SEGMED-01 §2]; [PROT-SEGMED-01 §4]"
)
RESPOSTA_SEM_BASE = (
    "Não encontrei essa informação nos protocolos internos disponíveis, então não vou responder "
    "por suposição. Recomendo consultar o especialista de plantão ou a coordenação médica.\n\n"
    "Fontes: [PROT-SEGMED-01 §4]"
)


def classificar_entrada(pergunta: str, similaridade_max: float | None = None) -> str:
    """Retorna 'injecao', 'prescricao', 'sem_base' ou 'clinica'."""
    if _INJECAO.search(pergunta):
        return "injecao"
    if _PRESCREVA.search(pergunta) or (_ACAO.search(pergunta) and _ALVO.search(pergunta)):
        return "prescricao"
    if similaridade_max is not None and similaridade_max < LIMIAR_SEM_BASE:
        return "sem_base"
    return "clinica"


def resposta_bloqueio(categoria: str) -> str:
    return RESPOSTA_SEM_BASE if categoria == "sem_base" else RESPOSTA_PRESCRICAO
