"""Data augmentation idêntica à da Entrega 2, implementada com as mesmas operações PIL do torchvision.

A Entrega 2 aplicava, em imagens PIL:
    RandomResizedCrop(scale=(0.6, 1.0)) -> RandomHorizontalFlip -> RandomRotation(10) -> ColorJitter(0.3, 0.3, 0.3)
Para imagens PIL o torchvision usa internamente Image.crop/resize, Image.rotate e ImageEnhance; reproduzimos
essas chamadas e os mesmos sorteios. Vantagens sobre camadas de augmentation do Keras: semântica exatamente
igual à da Entrega 2, independência de backend e custo desprezível em CPU (o recorte afim do Keras é lento
no backend JAX em CPU).
"""
import math

import numpy as np
from PIL import Image, ImageEnhance

from . import config


def _parametros_recorte(largura: int, altura: int, gerador: np.random.Generator,
                        escala=(0.6, 1.0), proporcao=(3 / 4, 4 / 3)) -> tuple[int, int, int, int]:
    """Mesmo algoritmo de RandomResizedCrop.get_params: 10 tentativas e, se falhar, recorte central."""
    area = largura * altura
    log_r = (math.log(proporcao[0]), math.log(proporcao[1]))
    for _ in range(10):
        area_alvo = area * gerador.uniform(*escala)
        razao = math.exp(gerador.uniform(*log_r))
        w = int(round(math.sqrt(area_alvo * razao)))
        h = int(round(math.sqrt(area_alvo / razao)))
        if 0 < w <= largura and 0 < h <= altura:
            topo = int(gerador.integers(0, altura - h + 1))
            esquerda = int(gerador.integers(0, largura - w + 1))
            return topo, esquerda, h, w
    razao_img = largura / altura                                # fallback: recorte central
    if razao_img < min(proporcao):
        w, h = largura, int(round(largura / min(proporcao)))
    elif razao_img > max(proporcao):
        w, h = int(round(altura * max(proporcao))), altura
    else:
        w, h = largura, altura
    return (altura - h) // 2, (largura - w) // 2, h, w


def aumentar(img: Image.Image, gerador: np.random.Generator, tamanho: int = config.IMG_SIZE) -> np.ndarray:
    """Aplica a augmentation da Entrega 2 a uma imagem PIL e devolve um array uint8 (tamanho, tamanho, 3)."""
    topo, esquerda, h, w = _parametros_recorte(img.width, img.height, gerador)
    img = img.crop((esquerda, topo, esquerda + w, topo + h)).resize((tamanho, tamanho), Image.BILINEAR)
    if gerador.uniform() < 0.5:                                 # RandomHorizontalFlip(p=0.5)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angulo = gerador.uniform(-10, 10)                           # RandomRotation(10): NEAREST, fundo preto
    img = img.rotate(angulo, resample=Image.NEAREST, expand=False, fillcolor=(0, 0, 0))
    # ColorJitter(0.3, 0.3, 0.3): fatores em [0,7; 1,3] aplicados em ordem aleatória (hue = 0 não é aplicado)
    fatores = {"brilho": gerador.uniform(0.7, 1.3), "contraste": gerador.uniform(0.7, 1.3),
               "saturacao": gerador.uniform(0.7, 1.3)}
    ajustes = {"brilho": ImageEnhance.Brightness, "contraste": ImageEnhance.Contrast, "saturacao": ImageEnhance.Color}
    for indice in gerador.permutation(4):
        if indice < 3:
            nome = ("brilho", "contraste", "saturacao")[indice]
            img = ajustes[nome](img).enhance(fatores[nome])
    return np.asarray(img, dtype=np.uint8)
