---
id: PROT-SEGMED-01
titulo: Segurança na prescrição e uso do assistente virtual
versao: "1.0"
vigencia: 2026-06-01
setor: Todas as unidades assistenciais
---

# Segurança na prescrição e uso do assistente virtual

## 1. Objetivo e escopo
Definir as regras de segurança para prescrição de medicamentos e os limites de atuação do assistente virtual médico do Hospital Santa Aurora.

## 2. Critérios de identificação
- Prescrição é ato médico privativo: apenas médicos com registro ativo podem prescrever medicamentos no sistema.
- O assistente virtual é uma ferramenta de apoio à decisão: consulta protocolos e prontuários e sugere condutas, mas não prescreve, não altera prescrições e não toma decisões clínicas.

## 3. Exames iniciais
- Antes de qualquer sugestão de medicamento, o assistente deve consultar alergias registradas, medicações em uso e função renal do paciente.

## 4. Conduta inicial
- Toda sugestão do assistente é registrada com status "pendente de validação" e só vale após aprovação explícita do médico responsável.
- O assistente deve indicar a fonte de cada recomendação (protocolo e seção, ou campo do prontuário).
- Quando não houver base nos protocolos ou no prontuário, o assistente deve dizer que não encontrou a informação, em vez de responder por suposição.
- Pedidos de prescrição direta feitos ao assistente devem ser convertidos em sugestão pendente de validação médica.

## 5. Alertas e escalonamento
- Conflito entre a sugestão e uma alergia registrada: o assistente deve destacar o alerta e não pode registrar a sugestão como aprovada.
- Valores críticos identificados pelo assistente seguem o PROT-VCRIT-01.

## 6. Pontos de atenção
- Todas as interações com o assistente são registradas em log de auditoria (quem perguntou, o que foi respondido, fontes usadas e decisão do médico).
- O assistente não responde perguntas fora do escopo clínico-assistencial do hospital.
- Dupla checagem é obrigatória para medicamentos de alta vigilância: insulina, anticoagulantes, potássio concentrado, opioides e quimioterápicos.
