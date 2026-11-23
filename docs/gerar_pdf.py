"""Gera docs/relatorio-entrega.pdf a partir de docs/relatorio.md (Markdown -> HTML -> PDF pelo Chrome).

Uso: uv run --with markdown python docs/gerar_pdf.py
"""

import re
import subprocess
import sys
from pathlib import Path

import markdown

DOCS = Path(__file__).parent
SAIDA = DOCS / "relatorio-entrega.pdf"
RODAPE = "Tech Challenge Fase 3 · Hospital Santa Aurora"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=block');
:root { --marca: #2e7480; --barra: #4a9eaa; --texto: #2b2f33; --suave: #6b7178; --zebra: #f0f7f8; --linha: #d7e3e5; }
@page { size: A4; margin: 18mm 17.5mm 20mm 17.5mm;
  @bottom-left { content: "RODAPE"; font-family: Poppins, sans-serif; font-size: 7.5pt; color: var(--suave); }
  @bottom-right { content: counter(page); font-family: Poppins, sans-serif; font-size: 7.5pt; color: var(--suave); } }
@page capa { margin: 0; @bottom-left { content: none; } @bottom-right { content: none; } }
body { font-family: Poppins, sans-serif; font-size: 9.6pt; line-height: 1.85; color: var(--texto); margin: 0; }
b, strong { font-weight: 600; color: #1a1d20; }
a { color: var(--marca); text-decoration: none; }

.capa { page: capa; height: 297mm; box-sizing: border-box; display: flex; flex-direction: column;
  border-left: 6mm solid var(--barra); padding: 26mm 18mm 22mm 16mm; break-after: page; }
.capa .rotulo { color: var(--marca); font-weight: 600; letter-spacing: .06em; font-size: 7.8pt; }
.capa h1 { font-size: 29.5pt; line-height: 1.22; font-weight: 700; margin: 7mm 0 4mm; color: #1a1d20; }
.capa .sub { font-size: 12pt; color: var(--suave); }
.capa .meta { margin-top: auto; border-top: 1px solid var(--linha); padding-top: 7mm;
  display: grid; grid-template-columns: 40mm 1fr; row-gap: 3.2mm; font-size: 9.8pt; line-height: 1.4; }
.capa .meta b { font-weight: 600; }

h2 { font-size: 18pt; font-weight: 700; color: #1a1d20; margin: 0 0 6mm; padding-top: 5mm;
  border-top: 1.3mm solid var(--barra); break-before: page; break-after: avoid; }
h2.sem-quebra { break-before: auto; margin-top: 12mm; }
h2 .num { color: var(--marca); }
h3 { font-size: 12pt; font-weight: 600; color: var(--marca); margin: 7mm 0 2mm; break-after: avoid; }
p { margin: 0 0 3.5mm; }
ul, ol { padding-left: 6mm; margin: 0 0 4mm; }
li { margin-bottom: 1.6mm; }
ul li::marker { color: var(--barra); }
ol li::marker { color: var(--marca); }
p, li { orphans: 3; widows: 3; }

table { border-collapse: collapse; width: 100%; margin: 4mm 0 6mm; font-size: 8.4pt; line-height: 1.45;
  break-inside: avoid; }
th { background: var(--marca); color: #fff; font-weight: 600; text-align: left; padding: 2.6mm 2.2mm; }
td { padding: 2.4mm 2.2mm; border-bottom: 1px solid var(--linha); vertical-align: top; }
tbody tr:nth-child(odd) td { background: var(--zebra); }
td:first-child { font-weight: 500; color: #1a1d20; }
th[align="right"], td[align="right"] { text-align: right; }

code { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.6pt; color: var(--marca);
  background: #e2f1f3; padding: .4mm 1.2mm; border-radius: 1.2mm; }

img { display: block; box-sizing: border-box; max-width: 100%; margin: 4mm auto; border: 1px solid #e3e6e8; border-radius: 2mm;
  break-inside: avoid; }
figure { margin: 4mm 0 6mm; break-inside: avoid; }
figure img { width: 92%; margin: 0 auto 2mm; }
figure.estreita img { width: 70%; }
figcaption { font-size: 8.4pt; color: var(--suave); }
""".replace("RODAPE", RODAPE)

CAPA = """
<div class="capa">
  <div class="rotulo">TECH CHALLENGE · FASE 3 · PÓS TECH IA PARA DEVS</div>
  <h1>Assistente médico virtual com fine&#8209;tuning, LangChain e LangGraph</h1>
  <div class="sub">Hospital Santa Aurora (hospital fictício) · Relatório técnico</div>
  <div class="meta">
    <b>Autores</b><span>Nicolas Gomes (RM 374793) e William Coelho (RM 374810)</span>
    <b>Repositório</b><a href="https://github.com/fiap-pos-org/fiap-tc-fase3">github.com/fiap-pos-org/fiap-tc-fase3</a>
    <b>App online</b><span><a href="https://aurora.williamdevops.com.br">aurora.williamdevops.com.br</a> (usuário <b>admin</b>, senha <b>admin123</b>)</span>
    <b>Modelo</b><a href="https://huggingface.co/william-devops/hospital-assistant-gguf">huggingface.co/william-devops/hospital-assistant-gguf</a>
  </div>
</div>
"""


def ajustar(html: str) -> str:
    # Número da seção em destaque; a conclusão continua na página da seção anterior.
    def h2(m):
        num, titulo = m.group(1), m.group(2)
        classe = ' class="sem-quebra"' if num == "10." else ""
        return f'<h2{classe}><span class="num">{num}</span>&nbsp;&nbsp;{titulo}</h2>'
    html = re.sub(r"<h2>(\d+\.)\s*(.*?)</h2>", h2, html)

    # Prints das telas viram figuras com legenda (a da triagem é mais alta, fica mais estreita).
    def figura(m):
        alt, src = m.group(1), m.group(2)
        classe = ' class="estreita"' if "triagem" in src else ""
        return f'<figure{classe}><img src="{src}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'
    html = re.sub(r'<p><img alt="([^"]*)" src="(telas/[^"]+)" ?/?></p>', figura, html)

    # Endereços soltos no texto viram links.
    return re.sub(r'(?<![">])(https?://[^\s<)]+?)(?=[.,;]?[\s<)])', r'<a href="\1">\1</a>', html)


def main() -> None:
    texto = (DOCS / "relatorio.md").read_text(encoding="utf-8")
    # A capa substitui o cabeçalho do Markdown (título, autores e links), até a seção 1.
    corpo = texto[texto.index("## 1. Resumo"):]
    html_corpo = ajustar(markdown.markdown(corpo, extensions=["tables", "fenced_code"]))
    html = (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
            f"<title>Relatório técnico, Tech Challenge Fase 3</title><style>{CSS}</style></head>"
            f"<body>{CAPA}{html_corpo}</body></html>")
    temporario = DOCS / "_relatorio.html"
    temporario.write_text(html, encoding="utf-8")
    try:
        subprocess.run(
            ["google-chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             "--allow-file-access-from-files", "--virtual-time-budget=15000",
             f"--print-to-pdf={SAIDA}", temporario.resolve().as_uri()],
            check=True, capture_output=True,
        )
    finally:
        temporario.unlink()
    print(f"gerado: {SAIDA}")


if __name__ == "__main__":
    sys.exit(main())
