"""Gera os diagramas SVG do README (versões clara e escura).

Uso: python docs/diagramas/gerar.py

Os PNGs (docs/diagramas/*.png, usados no relatório) são o render da versão clara em escala 2x,
feito no navegador (ex.: Chrome headless com --force-device-scale-factor=2).
"""

from pathlib import Path

SAIDA = Path(__file__).parent

TEMAS = {
    "claro": {
        "texto": "#1f2328", "sub": "#59636e", "linha": "#818b98", "caixa": "#ffffff",
        "borda": "#d1d9e0", "grupo": "#f6f8fa", "modelo": "#e3f2f4", "modelo_borda": "#0f6674",
        "humano": "#fff4e5", "humano_borda": "#bc4c00", "alerta": "#ffebe9", "alerta_borda": "#cf222e",
    },
    "escuro": {
        "texto": "#e6edf3", "sub": "#9198a1", "linha": "#7d8590", "caixa": "#161b22",
        "borda": "#3d444d", "grupo": "#0d1117", "modelo": "#0f2e33", "modelo_borda": "#2a9aab",
        "humano": "#2e1e0c", "humano_borda": "#f0883e", "alerta": "#2d1214", "alerta_borda": "#f85149",
    },
}
FONTE = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"


def svg(largura: int, altura: int, rotulo: str, corpo: list[str], c: dict) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {largura} {altura}' "
        f"width='{largura}' role='img' aria-label='{rotulo}' {FONTE}>"
        f"<defs><marker id='seta' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' "
        f"markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' "
        f"fill='{c['linha']}'/></marker>"
        f"<marker id='seta-humano' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' "
        f"markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' "
        f"fill='{c['humano_borda']}'/></marker></defs>" + "".join(corpo) + "</svg>"
    )


def caixa(c, x, y, w, h, titulo, sub="", tipo="caixa") -> str:
    fundo = {"caixa": c["caixa"], "modelo": c["modelo"], "humano": c["humano"], "alerta": c["alerta"]}[tipo]
    borda = {"caixa": c["borda"], "modelo": c["modelo_borda"], "humano": c["humano_borda"],
             "alerta": c["alerta_borda"]}[tipo]
    cx = x + w / 2
    ty = y + h / 2 + (-4 if sub else 5)
    partes = [
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fundo}' stroke='{borda}' stroke-width='1.5'/>",
        f"<text x='{cx}' y='{ty}' text-anchor='middle' font-size='13' font-weight='600' fill='{c['texto']}'>{titulo}</text>",
    ]
    if sub:
        partes.append(
            f"<text x='{cx}' y='{ty + 17}' text-anchor='middle' font-size='11' fill='{c['sub']}'>{sub}</text>"
        )
    return "".join(partes)


def seta(c, pontos, rotulo="", rx=None, ry=None, humano=False, tracejada=False, ancora="middle") -> str:
    cor = c["humano_borda"] if humano else c["linha"]
    marcador = "seta-humano" if humano else "seta"
    traco = " stroke-dasharray='5 4'" if tracejada else ""
    pts = " ".join(f"{x},{y}" for x, y in pontos)
    partes = [
        f"<polyline points='{pts}' fill='none' stroke='{cor}' stroke-width='1.5'{traco} "
        f"marker-end='url(#{marcador})'/>"
    ]
    if rotulo:
        cor_texto = c["humano_borda"] if humano else c["sub"]
        partes.append(
            f"<text x='{rx}' y='{ry}' text-anchor='{ancora}' font-size='11' fill='{cor_texto}'>{rotulo}</text>"
        )
    return "".join(partes)


def arquitetura(c: dict) -> str:
    # grade: colunas x=20 (médico), 190 (API), 400..630 (orquestração), 700 (dados e modelo)
    lojas = [(85, "Prontuário", "SQLite, consultas fixas", "consulta", "caixa"),
             (165, "Protocolos", "busca semântica (bge-m3)", "trechos", "caixa"),
             (245, "Modelo ajustado", "Qwen2.5-1.5B + QLoRA", "só redige", "modelo"),
             (325, "Auditoria", "um evento por etapa", "registra", "caixa")]
    corpo = [
        caixa(c, 20, 177, 110, 56, "Médico"),
        caixa(c, 190, 169, 150, 72, "Interface + API", "Tabler, FastAPI, login"),
        seta(c, [(130, 205), (188, 205)], "pergunta", 159, 197),
        f"<rect x='400' y='50' width='230' height='310' rx='10' fill='{c['grupo']}' "
        f"stroke='{c['borda']}' stroke-dasharray='4 4'/>",
        f"<text x='515' y='348' text-anchor='middle' font-size='11' fill='{c['sub']}'>LangGraph + LangChain</text>",
        caixa(c, 420, 80, 190, 64, "Fluxo A: triagem", "pendências, críticos, alertas"),
        caixa(c, 420, 250, 190, 64, "Fluxo B: dúvida clínica", "travas de entrada e saída"),
        seta(c, [(340, 192), (370, 192), (370, 112), (418, 112)], "paciente", 364, 150, ancora="end"),
        seta(c, [(340, 218), (370, 218), (370, 282), (418, 282)], "pergunta", 364, 256, ancora="end"),
        # pausa humana: o fluxo A devolve a decisão ao médico
        seta(c, [(460, 80), (460, 24), (75, 24), (75, 175)],
             "pausa: o médico aprova ou rejeita a sugestão", 268, 16, humano=True, tracejada=True),
    ]
    for y, titulo, sub, rotulo, tipo in lojas:
        corpo.append(caixa(c, 700, y - 28, 200, 56, titulo, sub, tipo=tipo))
        corpo.append(seta(c, [(630, y), (698, y)], rotulo, 664, y - 7))
    rotulo = ("Arquitetura: o modelo só redige; fatos vêm do prontuário e dos protocolos; "
              "a triagem pausa até o médico decidir; cada etapa vai para a auditoria")
    return svg(920, 372, rotulo, corpo, c)


def triagem(c: dict) -> str:
    y1, y2, h, w = 34, 154, 58, 150
    xs = [20, 215, 410, 605]  # vãos de 45 px entre as caixas
    m1, m2 = y1 + h // 2, y2 + h // 2
    corpo = [
        caixa(c, xs[0], y1, w, h, "Prontuário", "consulta no banco"),
        caixa(c, xs[1], y1, w, h, "Pendências", "exames sem resultado"),
        caixa(c, xs[2], y1, w, h, "Valores críticos", "regra do PROT-VCRIT-01"),
        caixa(c, xs[3], y1, w, h, "Protocolo do caso", "do prontuário + busca"),
        caixa(c, 800, y1, w, h, "Alertas à equipe", "gravados no banco", tipo="alerta"),
        seta(c, [(170, m1), (213, m1)]),
        seta(c, [(365, m1), (408, m1)]),
        seta(c, [(560, m1), (603, m1)]),
        seta(c, [(485, y1), (485, 20), (875, 20), (875, y1 - 2)], "se houver valor crítico", 680, 15),
        caixa(c, xs[0], y2, w, h, "Sugestão", "modelo ajustado redige", tipo="modelo"),
        caixa(c, xs[1], y2, w, h, "Validação", "fontes, alergia, dose"),
        caixa(c, xs[2], y2, w, h, "Aprovação", "médico com nome e CRM", tipo="humano"),
        caixa(c, xs[3], y2, w, h, "Conduta registrada", "aprovada ou rejeitada"),
        seta(c, [(680, y1 + h), (680, 122), (95, 122), (95, y2 - 2)]),
        seta(c, [(170, m2), (213, m2)]),
        seta(c, [(365, m2), (408, m2)]),
        seta(c, [(560, m2), (603, m2)], humano=True),
        seta(c, [(290, y2 + h), (290, 236), (680, 236), (680, y2 + h + 2)],
             "reprovada na validação: registrada como bloqueada, sem chegar ao médico", 485, 252),
        # legenda (a codificação por cor se repete nas caixas)
        f"<rect x='800' y='158' width='14' height='14' rx='3' fill='{c['caixa']}' stroke='{c['borda']}'/>",
        f"<text x='822' y='170' font-size='11' fill='{c['sub']}'>código</text>",
        f"<rect x='800' y='180' width='14' height='14' rx='3' fill='{c['modelo']}' stroke='{c['modelo_borda']}'/>",
        f"<text x='822' y='192' font-size='11' fill='{c['sub']}'>modelo ajustado</text>",
        f"<rect x='800' y='202' width='14' height='14' rx='3' fill='{c['humano']}' stroke='{c['humano_borda']}'/>",
        f"<text x='822' y='214' font-size='11' fill='{c['sub']}'>decisão humana</text>",
    ]
    rotulo = ("Fluxo de triagem: verificações em código, o modelo só redige a sugestão, a validação "
              "barra respostas inseguras e nada vira conduta sem a aprovação do médico")
    return svg(970, 266, rotulo, corpo, c)


if __name__ == "__main__":
    for nome_tema, cores in TEMAS.items():
        for nome, desenhar in [("arquitetura", arquitetura), ("triagem", triagem)]:
            (SAIDA / f"{nome}-{nome_tema}.svg").write_text(desenhar(cores), encoding="utf-8")
    print(f"SVGs gerados em {SAIDA}")
