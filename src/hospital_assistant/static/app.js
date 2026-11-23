"use strict";

// Todo texto vindo da API entra por textContent (nunca como HTML).
function el(tag, attrs = {}, ...filhos) {
  const no = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") no.className = v;
    else if (k === "text") no.textContent = v;
    else if (k.startsWith("on")) no.addEventListener(k.slice(2), v);
    else no.setAttribute(k, v);
  }
  for (const f of filhos.flat()) if (f != null) no.append(f);
  return no;
}
const icone = (nome, classe = "") => el("i", { class: `ti ti-${nome} ${classe}` });
const $ = (id) => document.getElementById(id);
const limpar = (no) => { no.replaceChildren(); return no; };

async function api(caminho, opcoes = {}) {
  const resp = await fetch(caminho, {
    headers: { "Content-Type": "application/json" },
    ...opcoes,
    body: opcoes.body ? JSON.stringify(opcoes.body) : undefined,
  });
  if (!resp.ok) {
    const erro = await resp.json().catch(() => ({}));
    throw new Error(typeof erro.detail === "string" ? erro.detail : `Erro ${resp.status}`);
  }
  return resp.json();
}

function avisoErro(destino, mensagem) {
  destino.append(el("div", { class: "alert alert-danger", text: mensagem }));
}

// ---------------------------------------------------------------- navegação
const TITULOS = { duvida: "Dúvida clínica", triagem: "Triagem automática", pacientes: "Pacientes", auditoria: "Auditoria" };

function navegar() {
  const rota = (location.hash || "#duvida").slice(1);
  const atual = TITULOS[rota] ? rota : "duvida";
  document.querySelectorAll(".tela").forEach((t) => t.classList.toggle("d-none", t.id !== `tela-${atual}`));
  document.querySelectorAll("[data-rota]").forEach((a) => a.classList.toggle("active", a.dataset.rota === atual));
  $("titulo").textContent = TITULOS[atual];
  if (atual === "auditoria") carregarAuditoria();
}
window.addEventListener("hashchange", navegar);

// ---------------------------------------------------------------- componentes
function etiquetasFontes(fontes) {
  if (!fontes.length) return el("span", { class: "badge bg-red-lt", text: "Sem fonte citada" });
  return el("div", { class: "d-flex flex-wrap gap-1" },
    el("span", { class: "text-secondary small me-1", text: "Fontes:" }),
    fontes.map((f) => el("span", { class: "badge bg-azure-lt fonte", text: f })));
}

function blocoTrechos(trechos) {
  if (!trechos.length) return null;
  return el("details", { class: "trechos mt-2" },
    el("summary", { text: `Trechos de protocolo consultados (${trechos.length})` }),
    trechos.map((t) => el("div", { class: "trecho" }, el("strong", { text: `[${t.ref}] ` }), t.texto)));
}

function linkAuditoria(traceId) {
  return el("a", { class: "trace text-secondary", href: "#auditoria", onclick: () => setTimeout(() => abrirTrilha(traceId), 50) },
    icone("shield-check", "me-1"), `execução ${traceId}`);
}

function resumoProntuario(p) {
  const criticos = p.exames.filter((e) => e.critico);
  return el("div", {},
    el("div", { class: "d-flex align-items-center mb-2" },
      el("span", { class: "avatar avatar-sm me-2 bg-primary-lt", text: p.sexo }),
      el("div", {}, el("div", { class: "fw-bold", text: `${p.id}, ${p.idade} anos` }),
        el("div", { class: "text-secondary small", text: `${p.setor} | internado em ${p.data_internacao.replace("T", " ")}` }))),
    el("div", { class: "mb-2" }, el("span", { class: "text-secondary small", text: "Diagnóstico" }), el("div", { text: p.diagnostico })),
    el("div", { class: "mb-2" }, el("span", { class: "text-secondary small", text: "Alergias " }),
      p.alergias.length ? p.alergias.map((a) => el("span", { class: "badge bg-red-lt me-1", text: a })) : el("span", { class: "small", text: "nenhuma registrada" })),
    el("div", { class: "mb-2" }, el("span", { class: "text-secondary small", text: "Medicações em uso" }),
      el("div", { class: "small", text: p.medicacoes_em_uso.join("; ") || "nenhuma" })),
    criticos.length ? el("div", { class: "alert alert-danger py-2 mb-2" }, icone("alert-triangle", "me-1"),
      `${criticos.length} valor(es) crítico(s): ${criticos.map((e) => e.nome).join(", ")}`) : null,
    el("table", { class: "table table-sm mb-0" },
      el("tbody", {}, p.exames.map((e) => el("tr", {},
        el("td", { class: "small", text: e.nome }),
        el("td", { class: "small text-end" },
          e.status === "pendente" ? el("span", { class: "badge bg-yellow-lt", text: "pendente" })
            : e.critico ? el("span", { class: "badge bg-red text-white", title: `crítico: ${e.limite_critico}`, text: e.resultado })
            : el("span", { text: e.resultado })))))));
}

// ---------------------------------------------------------------- pacientes (carga comum)
let PACIENTES = [];

function preencherSeletor(select, geral) {
  limpar(select);
  if (geral) select.append(el("option", { value: "", text: "Pergunta geral (sem paciente)" }));
  for (const p of PACIENTES) {
    select.append(el("option", { value: p.id, text: `${p.id} | ${p.idade} anos, ${p.sexo} | ${p.diagnostico}${p.criticos ? `  (${p.criticos} crítico${p.criticos > 1 ? "s" : ""})` : ""}` }));
  }
}

// ---------------------------------------------------------------- dúvida clínica
const EXEMPLOS_GERAIS = [
  "Qual o prazo para iniciar o antibiótico na sepse?",
  "Quais os critérios diagnósticos de cetoacidose diabética?",
  "Quais valores de potássio são críticos?",
  "Prescreva ceftriaxona 1 g EV para o paciente.",
  "Qual o protocolo do hospital para grandes queimados?",
];
const EXEMPLOS_PACIENTE = [
  "Quais exames estão pendentes para este paciente?",
  "Qual a conduta inicial prevista para este caso?",
  "Resuma o caso deste paciente.",
];

function renderExemplos() {
  const comPaciente = Boolean($("duvida-paciente").value);
  const lista = limpar($("exemplos"));
  for (const texto of comPaciente ? [...EXEMPLOS_PACIENTE, ...EXEMPLOS_GERAIS] : EXEMPLOS_GERAIS) {
    lista.append(el("a", { href: "#duvida", class: "list-group-item list-group-item-action small",
      onclick: (ev) => { ev.preventDefault(); $("duvida-pergunta").value = texto; $("duvida-pergunta").focus(); } }, texto));
  }
}

async function selecionarPacienteDuvida() {
  const id = $("duvida-paciente").value;
  const destino = limpar($("duvida-prontuario"));
  renderExemplos();
  if (!id) {
    destino.append(el("p", { class: "text-secondary mb-0", text: "Nenhum paciente selecionado: as respostas usam só os protocolos." }));
    return;
  }
  try { destino.append(resumoProntuario(await api(`/api/pacientes/${id}`))); } catch (e) { avisoErro(destino, e.message); }
}

function mensagemAssistente(r) {
  const corpo = el("div", { class: "card-body" });
  if (r.categoria !== "clinica") {
    corpo.append(el("span", { class: "badge bg-orange-lt mb-2" }, icone("shield-lock", "me-1"), "Respondido pela trava de segurança, sem consultar o modelo"));
  } else if (r.substituida) {
    corpo.append(el("span", { class: "badge bg-orange-lt mb-2" }, icone("shield-x", "me-1"), "Resposta do modelo reprovada na validação e substituída"));
  }
  corpo.append(el("div", { class: "resposta", text: r.resposta }));
  corpo.append(el("div", { class: "mt-3" }, etiquetasFontes(r.fontes)));
  const trechos = blocoTrechos(r.trechos);
  if (trechos) corpo.append(trechos);
  corpo.append(el("div", { class: "mt-2 text-end" }, linkAuditoria(r.trace_id)));
  return el("div", { class: "card msg-assistente" }, corpo);
}

async function enviarPergunta(ev) {
  ev.preventDefault();
  const campo = $("duvida-pergunta");
  const pergunta = campo.value.trim();
  if (!pergunta) return;
  const chat = $("chat");
  $("chat-vazio")?.remove();
  chat.append(el("div", { class: "msg-usuario", text: pergunta }));
  const aguarde = el("div", { class: "card msg-assistente" }, el("div", { class: "card-body text-secondary" },
    el("span", { class: "spinner-border spinner-border-sm me-2" }), "Analisando a pergunta e consultando protocolos..."));
  chat.append(aguarde);
  chat.scrollTop = chat.scrollHeight;
  campo.value = "";
  $("duvida-enviar").disabled = true;
  try {
    const r = await api("/api/duvida", { method: "POST", body: { pergunta, paciente_id: $("duvida-paciente").value || null } });
    aguarde.replaceWith(mensagemAssistente(r));
  } catch (e) {
    aguarde.replaceWith(el("div", { class: "alert alert-danger", text: e.message }));
  } finally {
    $("duvida-enviar").disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }
}

// ---------------------------------------------------------------- triagem
let triagemAtual = null;

const STATUS = {
  aguardando: ["bg-orange", "Aguardando decisão do médico", "hourglass"],
  aprovada: ["bg-green", "Conduta aprovada", "circle-check"],
  rejeitada: ["bg-red", "Sugestão rejeitada", "circle-x"],
  bloqueada: ["bg-red", "Sugestão bloqueada na validação de segurança", "shield-x"],
  paciente_inexistente: ["bg-secondary", "Paciente não encontrado", "user-off"],
};

function renderTriagem(t) {
  triagemAtual = t;
  $("triagem-resultado").classList.remove("d-none");
  const etapas = limpar($("triagem-etapas"));
  for (const e of t.etapas) {
    const pausa = e.id === "validar_seguranca" && t.status === "aguardando";
    etapas.append(el("li", {}, el("span", { class: "marcador" }, icone("check")), el("div", { text: e.nome }), el("span", { class: "ms", text: `${e.ms} ms` })));
    if (pausa) etapas.append(el("li", { class: "pausa" }, el("span", { class: "marcador" }, icone("hourglass")), el("div", { class: "fw-bold", text: "Aguardando aprovação do médico" })));
  }
  limpar($("triagem-trace")).append(linkAuditoria(t.trace_id));

  const [cor, rotulo, ic] = STATUS[t.status] || ["bg-secondary", t.status, "info-circle"];
  limpar($("triagem-status")).append(el("span", { class: `badge ${cor} text-white fs-4 px-3 py-2` }, icone(ic, "me-1"), rotulo));

  const alertas = limpar($("triagem-alertas"));
  for (const a of t.alertas) alertas.append(el("div", { class: "alert alert-danger" }, icone("alert-triangle", "me-2"), a));

  limpar($("triagem-pendentes")).append(t.pendentes.length
    ? el("div", { class: "d-flex flex-wrap gap-2" }, t.pendentes.map((p) => el("span", { class: "badge bg-yellow-lt", text: p })))
    : el("span", { class: "text-secondary", text: "Nenhum exame pendente." }));

  $("triagem-sugestao").textContent = t.sugestao;
  const fontes = limpar($("triagem-fontes"));
  fontes.append(etiquetasFontes(t.fontes));
  const trechos = blocoTrechos(t.trechos);
  if (trechos) fontes.append(trechos);
  if (t.validacao && !t.validacao.aprovada) {
    fontes.append(el("div", { class: "alert alert-warning mt-3", text: `Bloqueada na validação: ${t.validacao.motivos.join("; ")}` }));
  }

  $("triagem-decisao").classList.toggle("d-none", t.status !== "aguardando");
  const registro = limpar($("triagem-registro"));
  if (t.registro && t.registro.status !== "bloqueada") {
    registro.append(el("div", { class: "alert alert-success" }, icone("file-check", "me-2"),
      `Conduta ${t.registro.status} por ${t.registro.medico} em ${t.registro.registrado_em.replace("T", " ")}.`,
      t.registro.comentario ? ` Comentário: ${t.registro.comentario}` : ""));
  }
}

async function executarTriagem() {
  const botao = $("triagem-executar");
  botao.disabled = true;
  $("triagem-progresso").classList.remove("d-none");
  $("triagem-resultado").classList.add("d-none");
  try {
    renderTriagem(await api("/api/triagem", { method: "POST", body: { paciente_id: $("triagem-paciente").value } }));
  } catch (e) {
    avisoErro(limpar($("triagem-status")), e.message);
    $("triagem-resultado").classList.remove("d-none");
  } finally {
    botao.disabled = false;
    $("triagem-progresso").classList.add("d-none");
  }
}

async function decidir(aprovar) {
  const medico = $("decisao-medico").value.trim();
  if (medico.length < 3) { $("decisao-medico").classList.add("is-invalid"); return; }
  $("decisao-medico").classList.remove("is-invalid");
  try {
    renderTriagem(await api(`/api/triagem/${triagemAtual.trace_id}/decisao`, {
      method: "POST", body: { aprovar, medico, comentario: $("decisao-comentario").value.trim() },
    }));
    $("decisao-comentario").value = "";
  } catch (e) {
    avisoErro($("triagem-registro"), e.message);
  }
}

// ---------------------------------------------------------------- tela de pacientes
function renderListaPacientes() {
  const filtro = $("pacientes-filtro").value.toLowerCase();
  const corpo = limpar($("pacientes-lista"));
  for (const p of PACIENTES) {
    if (filtro && !`${p.id} ${p.diagnostico} ${p.setor}`.toLowerCase().includes(filtro)) continue;
    corpo.append(el("tr", { onclick: () => detalharPaciente(p.id) },
      el("td", { class: "col-paciente" }, el("div", { class: "fw-medium", text: p.id }), el("div", { class: "text-secondary small", text: `${p.idade} anos, ${p.sexo}` })),
      el("td", { text: p.diagnostico }),
      el("td", { class: "text-secondary", text: p.setor }),
      el("td", { class: "text-center" }, p.criticos ? el("span", { class: "badge bg-red text-white", text: p.criticos }) : el("span", { class: "text-secondary", text: "-" }))));
  }
}

async function detalharPaciente(id) {
  const card = limpar($("pacientes-detalhe"));
  try {
    const p = await api(`/api/pacientes/${id}`);
    card.append(
      el("div", { class: "card-header" }, el("h3", { class: "card-title", text: `Prontuário ${p.id}` }),
        el("div", { class: "card-actions d-flex gap-2" },
          el("a", { class: "btn btn-sm btn-primary", href: "#triagem", onclick: () => { $("triagem-paciente").value = id; } }, icone("heartbeat", "me-1"), "Triagem"),
          el("a", { class: "btn btn-sm", href: "#duvida", onclick: () => { $("duvida-paciente").value = id; selecionarPacienteDuvida(); } }, icone("message-2-question", "me-1"), "Perguntar"))),
      el("div", { class: "card-body" }, resumoProntuario(p)));
  } catch (e) { avisoErro(card, e.message); }
}

// ---------------------------------------------------------------- auditoria
async function carregarAuditoria() {
  const corpo = limpar($("auditoria-lista"));
  try {
    for (const e of await api("/api/auditoria")) {
      corpo.append(el("tr", { onclick: () => abrirTrilha(e.trace_id) },
        el("td", { class: "small text-secondary", text: e.inicio.replace("T", " ").slice(0, 19) }),
        el("td", { class: "trace", text: e.trace_id }),
        el("td", {}, el("span", { class: `badge ${e.fluxo === "triagem" ? "bg-purple-lt" : "bg-azure-lt"}`, text: e.fluxo })),
        el("td", { text: e.paciente_id || "-" }),
        el("td", { class: "text-center", text: e.etapas })));
    }
  } catch (err) { avisoErro(corpo, err.message); }
}

async function abrirTrilha(traceId) {
  $("auditoria-titulo").textContent = `Trilha da execução ${traceId}`;
  const destino = limpar($("auditoria-trilha"));
  try {
    const eventos = await api(`/api/auditoria/${traceId}`);
    destino.append(el("ul", { class: "etapas" }, eventos.map((e) => el("li", { class: "evento" },
      el("span", { class: "marcador" }, icone("point")),
      el("div", { class: "d-flex justify-content-between" }, el("strong", { text: e.nome }),
        el("span", { class: "ms", text: `${e.criado_em.slice(11, 23)} | ${e.duracao_ms} ms` })),
      el("details", {}, el("summary", { class: "small text-secondary", text: "dados registrados" }),
        el("pre", { text: JSON.stringify(e.dados, null, 2) }))))));
  } catch (err) { avisoErro(destino, err.message); }
}

// ---------------------------------------------------------------- início
async function iniciar() {
  navegar();
  try {
    $("modelo").textContent = (await api("/api/info")).modelo;
    PACIENTES = await api("/api/pacientes");
  } catch (e) { avisoErro($("chat"), `Não foi possível carregar os dados: ${e.message}`); }
  preencherSeletor($("duvida-paciente"), true);
  preencherSeletor($("triagem-paciente"), false);
  renderExemplos();
  renderListaPacientes();
  $("duvida-paciente").addEventListener("change", selecionarPacienteDuvida);
  $("duvida-form").addEventListener("submit", enviarPergunta);
  $("duvida-limpar").addEventListener("click", () => limpar($("chat")));
  $("triagem-executar").addEventListener("click", executarTriagem);
  $("decisao-aprovar").addEventListener("click", () => decidir(true));
  $("decisao-rejeitar").addEventListener("click", () => decidir(false));
  $("pacientes-filtro").addEventListener("input", renderListaPacientes);
  $("auditoria-atualizar").addEventListener("click", carregarAuditoria);
}
iniciar();
