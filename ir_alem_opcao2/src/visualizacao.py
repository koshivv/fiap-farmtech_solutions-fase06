"""Figuras do Ir Além: grades de segmentação, curvas, matrizes de confusão, CAM e diagrama do pipeline."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from . import config

CORES = {"cnn": "tab:green", "tl": "tab:blue", "original": "tab:gray", "segmentada": "tab:orange"}


def aplicar_estilo() -> None:
    """Mesmo visual (fundo cinza com grade) das figuras das Entregas 1 e 2."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 120, "savefig.bbox": "tight"})


def salvar(fig, nome: str) -> Path:
    """Salva em assets/ (versionado) e devolve o caminho."""
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    caminho = config.ASSETS_DIR / nome
    fig.savefig(caminho)
    return caminho


# ---------------- Segmentação ----------------
def grade_segmentacao(itens: list[dict], titulo: str = "") -> plt.Figure:
    """Grade original | máscara | imagem recortada (exigida pelo enunciado).

    Cada item: {'titulo', 'original' (H,W,3), 'mascara' (H,W bool ou None), 'recortada' (H,W,3)}.
    Máscara None indica falha da segmentação: a imagem original segue para o classificador.
    """
    fig, eixos = plt.subplots(len(itens), 3, figsize=(12, 3.2 * len(itens)), squeeze=False)
    for linha, item in zip(eixos, itens):
        linha[0].imshow(item["original"])
        linha[0].set_ylabel(item["titulo"], fontsize=9)
        if item["mascara"] is None:
            linha[1].imshow(np.zeros(item["original"].shape[:2]), cmap="gray", vmin=0, vmax=1)
            linha[1].text(0.5, 0.5, "falha\n(fallback: original)", color="white", ha="center", va="center",
                          transform=linha[1].transAxes, fontsize=11)
        else:
            linha[1].imshow(item["mascara"], cmap="gray", vmin=0, vmax=1)
        linha[2].imshow(item["recortada"])
        for eixo in linha:
            eixo.set_xticks([])
            eixo.set_yticks([])
    for eixo, nome in zip(eixos[0], ["Original", "Máscara (automática)", "Recortada (fundo preto)"]):
        eixo.set_title(nome, fontsize=12, fontweight="bold")
    if titulo:
        fig.suptitle(titulo, fontsize=14, y=1.0)
    fig.tight_layout()
    return fig


# ---------------- Treino ----------------
def curvas_treino(historico: pd.DataFrame, titulo: str, inicio_fase2: int | None = None) -> plt.Figure:
    """Entropia cruzada e acurácia (treino x validação) por época; linhas finas = sementes, grossa = média."""
    fig, (ax_perda, ax_acc) = plt.subplots(1, 2, figsize=(14, 4.2))
    pares = [(ax_perda, "entropia", "Entropia cruzada (↓ melhor)"), (ax_acc, "accuracy", "Acurácia (↑ melhor)")]
    for ax, coluna, rotulo in pares:
        for prefixo, cor, nome in [("", "tab:blue", "treino"), ("val_", "tab:orange", "validação")]:
            for _, g in historico.groupby("seed"):
                ax.plot(g.epoca, g[prefixo + coluna], color=cor, alpha=0.2, linewidth=0.8)
            media = historico.groupby("epoca")[prefixo + coluna].mean()
            ax.plot(media.index, media.values, color=cor, linewidth=2.2, label=f"{nome} (média)")
        if inicio_fase2:
            ax.axvline(inicio_fase2 + 0.5, color="black", linestyle="--", linewidth=1.2,
                       label="início do fine tuning")
        ax.set_title(rotulo)
        ax.set_xlabel("Época")
        ax.legend(fontsize=9)
    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    return fig


def matrizes_confusao(paineis: list[tuple[str, np.ndarray]], colunas: int = 4) -> plt.Figure:
    """Matrizes de confusão lado a lado (linhas = real, colunas = previsto)."""
    linhas = int(np.ceil(len(paineis) / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(3.6 * colunas, 3.4 * linhas), squeeze=False)
    for ax, (titulo, matriz) in zip(eixos.flat, paineis):
        ax.imshow(matriz, cmap="Blues", vmin=0)
        for (i, j), valor in np.ndenumerate(matriz):
            ax.text(j, i, f"{valor:g}", ha="center", va="center", fontsize=13,
                    color="white" if valor > matriz.max() / 2 else "black")
        ax.set_xticks([0, 1], config.CLASSES)
        ax.set_yticks([0, 1], config.CLASSES)
        ax.set_xlabel("previsto")
        ax.set_ylabel("real")
        ax.set_title(titulo, fontsize=10)
        ax.grid(False)
    for ax in list(eixos.flat)[len(paineis):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------- CAM ----------------
def calcular_cam(submodelo, imagem: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """Class Activation Map da classe prevista, normalizado em [0, 1] e no tamanho da imagem."""
    mapas, logits = submodelo.predict(imagem[None].astype("float32"), verbose=0)
    pesos = submodelo.get_layer("logits").get_weights()[0]           # (canais, 2)
    classe = int(np.argmax(logits[0]))
    cam = np.maximum(mapas[0] @ pesos[:, classe], 0)
    cam = cam / (cam.max() + 1e-8)
    cam = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(imagem.shape[1::-1], Image.BILINEAR)) / 255.0
    prob = np.exp(logits[0] - logits[0].max())
    return cam, classe, prob / prob.sum()


def sobrepor_cam(ax, imagem: np.ndarray, cam: np.ndarray, titulo: str) -> None:
    ax.imshow(imagem.astype(np.uint8))
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(titulo, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------- Diagrama do pipeline (figura autoral) ----------------
def _caixa(ax, x0, y0, x1, y1, texto, cor, fonte=10, tracejada=False):
    """Caixa arredondada ocupando exatamente [x0, x1] x [y0, y1] (coordenadas da figura)."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.004,rounding_size=0.015",
                                facecolor=cor, edgecolor="black", linewidth=1.2,
                                linestyle="--" if tracejada else "-"))
    if texto:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, texto, ha="center", va="center", fontsize=fonte)


def _seta(ax, inicio, fim, cor="black", tracejada=False, curva=0.0):
    ax.add_patch(FancyArrowPatch(inicio, fim, arrowstyle="-|>", mutation_scale=15, color=cor, linewidth=1.5,
                                 linestyle="--" if tracejada else "-", connectionstyle=f"arc3,rad={curva}"))


ROTULO_SEGMENTADOR = {"yolo11s-seg": "YOLO11s-seg", "yolo26n-seg": "YOLO26n-seg",
                      "yolo26n-seg+sam2.1_b": "YOLO26n-seg (caixas)\n→ SAM 2.1-b"}


def figura_pipeline(exemplo: dict | None = None, segmentador: str = config.SEGMENTADOR_PADRAO,
                    classes: str = "estritas") -> plt.Figure:
    """Diagrama do pipeline: imagem -> [segmentação -> máscara -> recorte] -> classificador -> predição.

    `exemplo` (opcional): {'original', 'mascara', 'recortada'} para miniaturas reais sob as etapas.
    `segmentador` e `classes`: a escolha feita no notebook 02, escrita na caixa da segmentação.
    """
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.975, "Ir Além — Opção 2: segmentação + classificação (FarmTech)", ha="center", va="top",
            fontsize=15, fontweight="bold")

    y0, y1 = 0.56, 0.70                                         # linha principal
    _caixa(ax, 0.015, y0, 0.135, y1, "Imagem COCO\nval2017\n(gato ou carro)", "#e8e8e8")
    _caixa(ax, 0.185, y0, 0.335, y1, f"Segmentação: {ROTULO_SEGMENTADOR.get(segmentador, segmentador)}\n"
                                     f"(COCO, classes {classes})\nclasse prevista descartada", "#ffd8a8", fonte=9)
    _caixa(ax, 0.385, y0, 0.515, y1, "Máscara binária\n(união das\ninstâncias)", "#ffe8cc")
    _caixa(ax, 0.565, y0, 0.685, y1, "Aplicar máscara\n(fundo preto)", "#ffe8cc")
    _caixa(ax, 0.735, 0.47, 0.895, 0.79, "", "#f7f7f7", tracejada=True)        # grupo dos classificadores
    ax.text(0.815, 0.765, "Classificador", ha="center", va="center", fontsize=10, fontweight="bold")
    _caixa(ax, 0.75, 0.64, 0.88, 0.735, "CNN do zero\n(242 mil parâmetros)", "#c3e6cb", fonte=9.5)
    _caixa(ax, 0.75, 0.495, 0.88, 0.605, "EfficientNetB0\n(ImageNet + fine tuning)", "#b8daff", fonte=9.5)
    _caixa(ax, 0.93, 0.575, 0.99, 0.685, "Predição\ngato /\ncarro", "#e8e8e8", fonte=9.5)
    _caixa(ax, 0.30, 0.80, 0.70, 0.90, "Falha (sem detecção ou máscara < 1% da imagem):\n"
                                       "retenta com confiança 0,10; se falhar de novo, segue a imagem original",
           "#f8d7da", fonte=9)

    yc = (y0 + y1) / 2
    for xa, xb in ((0.135, 0.185), (0.335, 0.385), (0.515, 0.565), (0.685, 0.735), (0.895, 0.93)):
        _seta(ax, (xa, yc), (xb, yc))
    ax.text(0.71, yc + 0.015, "S", ha="center", fontsize=11, fontweight="bold", color="tab:orange")
    _seta(ax, (0.26, y1), (0.33, 0.80), cor="tab:red", tracejada=True)
    _seta(ax, (0.70, 0.85), (0.80, 0.79), cor="tab:red", tracejada=True, curva=-0.3)
    # Condição O: a imagem original vai direto para o classificador
    _seta(ax, (0.075, y0), (0.80, 0.47), cor="tab:gray", tracejada=True, curva=0.22)
    ax.text(0.40, 0.385, "O: imagem original, sem segmentação", color="dimgray", fontsize=10, style="italic",
            ha="center")
    ax.text(0.99, 0.36, "O = imagem original · S = imagem segmentada\n"
                        "Mesma entrada (224×224), mesma augmentation e mesma cabeça nos dois classificadores.\n"
                        "Treino e teste sempre na MESMA condição.", fontsize=9, va="top", ha="right")

    if exemplo is not None:                                     # miniaturas reais sob as etapas
        for x, chave, rotulo in ((0.015, "original", "original"), (0.385, "mascara", "máscara"),
                                 (0.565, "recortada", "recortada (entrada S)")):
            sub = fig.add_axes([x, 0.04, 0.13, 0.26])
            sub.imshow(exemplo[chave], cmap="gray" if chave == "mascara" else None)
            sub.set_title(rotulo, fontsize=9)
            sub.axis("off")
    return fig
