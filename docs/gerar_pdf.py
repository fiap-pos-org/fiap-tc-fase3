"""Gera docs/relatorio.pdf a partir de docs/relatorio.md (Markdown -> HTML -> PDF pelo Chrome).

Uso: uv run --with markdown python docs/gerar_pdf.py
"""

import subprocess
import sys
from pathlib import Path

import markdown

DOCS = Path(__file__).parent
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5;
  color: #1f2328; }
.capa { height: 245mm; display: flex; flex-direction: column; justify-content: center;
  border-left: 8px solid #0f6674; padding-left: 14mm; page-break-after: always; }
.capa .rotulo { color: #0f6674; font-weight: 700; letter-spacing: .08em; font-size: 10pt; }
.capa h1 { font-size: 26pt; line-height: 1.2; margin: 6mm 0 4mm; border: none; }
.capa .sub { font-size: 13pt; color: #59636e; margin-bottom: 14mm; }
.capa .meta { font-size: 11pt; line-height: 1.9; }
.capa .meta a { color: #0f6674; }
h1 { font-size: 18pt; color: #0f6674; border-bottom: 2px solid #0f6674; padding-bottom: 2mm; }
h2 { font-size: 14pt; color: #0f6674; margin-top: 9mm; border-bottom: 1px solid #d1d9e0;
  padding-bottom: 1.5mm; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 6mm; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #d1d9e0; padding: 1.6mm 2.4mm; text-align: left; vertical-align: top; }
th { background: #eef6f7; }
img { max-width: 100%; display: block; margin: 4mm auto; page-break-inside: avoid;
  border: 1px solid #e5e7eb; border-radius: 4px; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f3f4f6;
  padding: .3mm 1mm; border-radius: 3px; }
a { color: #0f6674; text-decoration: none; }
p, li { orphans: 3; widows: 3; }
"""

CAPA = """
<div class="capa">
  <div class="rotulo">TECH CHALLENGE · FASE 3 · PÓS TECH IA PARA DEVS</div>
  <h1>Assistente médico virtual com fine-tuning, LangChain e LangGraph</h1>
  <div class="sub">Hospital Santa Aurora (hospital fictício) · Relatório técnico</div>
  <div class="meta">
    <b>Autores:</b> Nicolas Gomes (RM 374793) e William Coelho (RM 374810)<br>
    <b>Repositório:</b> <a href="https://github.com/fiap-pos-org/fiap-tc-fase3">github.com/fiap-pos-org/fiap-tc-fase3</a><br>
    <b>App online:</b> <a href="https://aurora.williamdevops.com.br">aurora.williamdevops.com.br</a> (usuário <b>admin</b>, senha <b>admin123</b>)<br>
    <b>Modelo:</b> <a href="https://huggingface.co/william-devops/hospital-assistant-gguf">huggingface.co/william-devops/hospital-assistant-gguf</a>
  </div>
</div>
"""


def main() -> None:
    texto = (DOCS / "relatorio.md").read_text(encoding="utf-8")
    # A capa substitui o cabeçalho do Markdown (título, autores e links), até a seção 1.
    corpo = texto[texto.index("## 1. Resumo"):]
    html_corpo = markdown.markdown(corpo, extensions=["tables", "fenced_code"])
    html = (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
            f"<title>Relatório técnico, Tech Challenge Fase 3</title><style>{CSS}</style></head>"
            f"<body>{CAPA}{html_corpo}</body></html>")
    temporario = DOCS / "_relatorio.html"
    temporario.write_text(html, encoding="utf-8")
    try:
        subprocess.run(
            ["google-chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             "--allow-file-access-from-files", f"--print-to-pdf={DOCS / 'relatorio.pdf'}",
             temporario.resolve().as_uri()],
            check=True, capture_output=True,
        )
    finally:
        temporario.unlink()
    print(f"gerado: {DOCS / 'relatorio.pdf'}")


if __name__ == "__main__":
    sys.exit(main())
