"""Gráficos do relatório a partir de docs/avaliacao/treino.json e metricas.json.

Uso: uv run python -m hospital_assistant.evaluation.graficos
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hospital_assistant.config import ROOT_DIR  # noqa: E402

SAIDA = ROOT_DIR / "docs" / "avaliacao"
# Paleta categórica validada (ordem fixa) e tinta de texto neutra.
COR = {"original": "#2a78d6", "v1": "#eb6834", "v2": "#1baf7a", "sistema": "#eda100"}
NOME = {
    "original": "Original",
    "v1": "Ajustado v1",
    "v2": "Ajustado v2",
    "sistema": "Sistema completo (v2)",
}
TEXTO, TEXTO_2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _estilo(ax, titulo: str, rotulo_y: str) -> None:
    ax.set_title(titulo, loc="left", fontsize=12, color=TEXTO, pad=12)
    ax.set_ylabel(rotulo_y, color=TEXTO_2)
    ax.set_facecolor(FUNDO)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRADE)
    ax.tick_params(colors=TEXTO_2, length=0)


def _barras(ax, grupos: list[str], series: dict[str, list[float]]) -> None:
    largura = 0.8 / len(series)
    for i, (nome, valores) in enumerate(series.items()):
        xs = [g + (i - (len(series) - 1) / 2) * largura for g in range(len(grupos))]
        barras = ax.bar(xs, valores, largura - 0.03, color=COR[nome], label=NOME[nome])
        ax.bar_label(
            barras, labels=[f"{v:.0%}" for v in valores], fontsize=7.5, color=TEXTO_2, padding=2
        )
    ax.set_xticks(range(len(grupos)), grupos)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0, -0.12),
        ncol=len(series),
        labelcolor=TEXTO_2,
    )


def curvas_treino() -> None:
    historico = json.loads((SAIDA / "treino.json").read_text())["log_history"]
    treino = [(h["step"], h["loss"]) for h in historico if "loss" in h]
    validacao = [(h["step"], h["eval_loss"]) for h in historico if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=FUNDO)
    ax.plot(*zip(*treino, strict=True), color=COR["original"], linewidth=2, label="Treino")
    ax.plot(
        *zip(*validacao, strict=True),
        color=COR["v1"],
        linewidth=2,
        marker="o",
        markersize=5,
        label="Validação",
    )
    ax.annotate(
        f"{validacao[-1][1]:.2f}".replace(".", ","),
        validacao[-1],
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=8,
        color=TEXTO_2,
    )
    _estilo(ax, "Loss do fine-tuning v2 (QLoRA, 2 épocas)", "loss")
    ax.set_xlabel("passo", color=TEXTO_2)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    ax.legend(frameon=False, labelcolor=TEXTO_2)
    fig.tight_layout()
    fig.savefig(SAIDA / "curvas_treino.png", dpi=150)


def comparacao_teste(metricas: dict) -> None:
    rotulos = {
        "rouge_l": "ROUGE-L",
        "similaridade_semantica": "Similaridade\nsemântica",
        "fontes_exatas": "Fontes\nexatas",
        "com_linha_fontes": "Cita\nfontes",
    }
    series = {
        m: [metricas["teste"][m]["todos"][k] for k in rotulos] for m in ("original", "v1", "v2")
    }
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=FUNDO)
    _barras(ax, list(rotulos.values()), series)
    _estilo(
        ax,
        f"Conjunto de teste ({metricas['teste']['v2']['todos']['n']} exemplos não vistos no treino)",
        "",
    )
    fig.tight_layout()
    fig.savefig(SAIDA / "comparacao_teste.png", dpi=150)


def ataques(metricas: dict) -> None:
    categorias = {
        "prescricao": "Prescrição",
        "injecao": "Injeção de\nprompt",
        "fora_escopo": "Fora de\nescopo",
        "sem_base": "Sem base",
    }
    series = {
        m: [metricas["ataques"][m]["por_categoria"].get(c, 0) for c in categorias]
        for m in ("original", "v1", "v2", "sistema")
    }
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=FUNDO)
    _barras(ax, list(categorias.values()), series)
    _estilo(ax, "Respostas seguras em 30 ataques (conjunto separado)", "")
    fig.tight_layout()
    fig.savefig(SAIDA / "ataques.png", dpi=150)


if __name__ == "__main__":
    curvas_treino()
    metricas_path = SAIDA / "metricas.json"
    if metricas_path.exists():
        metricas = json.loads(metricas_path.read_text())
        comparacao_teste(metricas)
        ataques(metricas)
    print(f"gráficos em {SAIDA}")
