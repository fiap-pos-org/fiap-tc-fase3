# Tech Challenge: Fase 3

Projeto da Fase 3 da Pós Tech IA para Devs (FIAP): assistente virtual médico de um hospital fictício, com fine-tuning de LLM em dados internos sintéticos e anonimizados, orquestrado com LangChain e LangGraph.

O enunciado completo está em [`desafio/`](desafio/).

## Objetivo

- Ajustar (fine-tuning) uma LLM aberta com protocolos, perguntas frequentes e modelos de documentos do hospital.
- Responder dúvidas clínicas contextualizadas com o prontuário do paciente, citando a fonte.
- Executar fluxos automatizados (exames pendentes, alertas, sugestões) com **aprovação humana obrigatória** antes de qualquer conduta.
- Registrar logs detalhados para auditoria.

## Estrutura

```
├── desafio/                      # enunciado do Tech Challenge
├── data/
│   ├── raw/                      # fontes internas do hospital fictício
│   │   ├── protocolos/           # 11 protocolos clínicos (Markdown, seções numeradas)
│   │   ├── modelos/              # 6 modelos de laudos, receitas e procedimentos
│   │   ├── faq.jsonl             # 56 perguntas frequentes de médicos (3 variações cada)
│   │   └── prontuarios.jsonl     # 200 prontuários brutos, COM dados pessoais fictícios
│   ├── processed/                # dados anonimizados e dataset de instruções
│   │   ├── prontuarios.jsonl     # prontuários anonimizados
│   │   ├── train/val/test.jsonl  # pares de instrução (formato chat)
│   │   └── relatorio_dados.json  # estatísticas de anonimização, curadoria e splits
│   └── eval/                     # conjuntos de avaliação (fora do treino)
│       ├── ataques.jsonl         # 30 pedidos de prescrição, injeções, fora de escopo e sem base
│       └── pubmedqa_200.jsonl    # amostra estratificada do PubMedQA
├── src/hospital_assistant/
│   ├── config.py                 # caminhos centralizados
│   ├── prompts.py                # prompt de sistema e formato das mensagens (treino = inferência)
│   └── data/                     # pipeline de dados
└── tests/                        # testes automatizados
```

A estrutura cresce a cada fase (fine-tuning, RAG, grafos, segurança e avaliação).

## Requisitos

- [uv](https://docs.astral.sh/uv/) (instala o Python 3.12 automaticamente)

## Como rodar

```bash
uv sync            # cria o ambiente e instala as dependências
uv run pytest      # roda os testes
uv run ruff check  # lint
```

## Dados

Todos os dados são **sintéticos**: o "Hospital Santa Aurora" é fictício e nenhum registro corresponde a pessoas reais. Os protocolos foram escritos com base em diretrizes clínicas públicas, apenas para fins acadêmicos, e **não devem ser usados na prática clínica**.

Para regenerar o dataset (determinístico, semente fixa):

```bash
uv run python -m hospital_assistant.data.pipeline   # prontuários, anonimização, dataset e splits
uv run python -m hospital_assistant.data.pubmedqa   # amostra de avaliação do PubMedQA (baixa da internet)
```

O pipeline faz, em ordem:

1. **Geração**: 200 prontuários brutos em 10 cenários clínicos, com nome, CPF, CNS, telefone, e-mail e endereço fictícios (gerados com Faker) para exercitar a anonimização.
2. **Pré-processamento**: normalização Unicode, remoção de caracteres de controle, padronização de unidades (mg/dL, mEq/L) e espaços.
3. **Anonimização** em três camadas:
   - identificadores conhecidos do próprio registro viram pseudônimos (`PAC-0001`, `[ACOMPANHANTE]`);
   - expressões regulares para CPF, CNS, telefone, e-mail e CEP;
   - reconhecimento de entidades do spaCy (`pt_core_news_md`) para nomes e locais que sobrarem.

   Campos identificadores são descartados, a data de nascimento vira idade e o mapeamento pseudônimo para pessoa não é salvo.
4. **Montagem** dos pares de instrução: perguntas frequentes, perguntas por seção de protocolo e modelo, perguntas contextualizadas com o prontuário (exames pendentes, valores críticos, alergias, resumo SBAR) e exemplos de segurança (recusa de prescrição, injeção de prompt, fora de escopo e "não encontrei nos protocolos"). Parte dos exemplos traz trechos de protocolo recuperados, inclusive um irrelevante, para o modelo aprender a citar a fonte certa.
5. **Curadoria**: deduplicação, limites de tamanho, validação das fontes citadas e nova busca por dados pessoais (exemplo com qualquer resíduo é descartado).
6. **Divisão** 80/10/10 por grupo e estratificada por tipo: variações da mesma pergunta e exemplos do mesmo paciente nunca ficam em splits diferentes.

O PubMedQA ([licença MIT](https://github.com/pubmedqa/pubmedqa)) é usado só na avaliação e é mantido em inglês: é um benchmark padrão, então mantê-lo no original permite comparar com a literatura e medir se o fine-tuning degradou o conhecimento geral do modelo.

## Status

Em desenvolvimento.
