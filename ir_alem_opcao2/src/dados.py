"""Dataset do Ir Além: manifesto das Entregas 1/2, teste ampliado e carregamento das imagens.

As imagens vêm do COCO 2017 val2017, como nas Entregas 1 e 2. Nada é rotulado à mão aqui:
para classificação basta o rótulo da imagem (grupo), que vem do próprio COCO.
"""
import http.client
import random
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image

from . import config

CONJUNTOS = ("treino", "validacao", "teste_original", "teste_ampliado")
MAPA_DIVISAO = {"treino": "treino", "validacao": "validacao", "teste": "teste_original"}


# ---------------- Downloads ----------------
def baixar_arquivo(url: str, destino: Path, tentativas: int = 3) -> bool:
    """Baixa `url` para `destino` aceitando apenas o host oficial do COCO (mesma regra da Entrega 1).

    O arquivo só é movido para o destino se o tamanho bater com o Content-Length: uma conexão que cai
    no meio não gera erro no urllib e deixaria um arquivo truncado que nunca seria baixado de novo.
    """
    if urlparse(url).hostname != config.COCO_HOST:
        return False
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(destino.suffix + ".parcial")
    for _ in range(tentativas):
        try:
            with urllib.request.urlopen(url, timeout=60) as resposta, open(temporario, "wb") as f:
                esperado = int(resposta.headers.get("Content-Length", -1))
                shutil.copyfileobj(resposta, f)
            if esperado < 0 or temporario.stat().st_size == esperado:
                temporario.replace(destino)                     # só aparece completo no destino
                return True
        except (OSError, http.client.HTTPException):
            pass
        temporario.unlink(missing_ok=True)
        time.sleep(2)                                           # espera antes de tentar de novo
    return False


def baixar_anotacoes_coco(pasta: Path = config.COCO_DIR) -> Path:
    """Garante o instances_val2017.json (~20 MB) extraído do zip oficial (~241 MB)."""
    json_path = Path(pasta) / "instances_val2017.json"
    if json_path.exists():
        return json_path
    zip_path = Path(pasta) / "annotations_trainval2017.zip"
    for _ in range(2):                                          # um zip corrompido é baixado de novo
        if not zip_path.exists() and not baixar_arquivo(config.COCO_ANOTACOES_URL, zip_path):
            raise RuntimeError("Falha ao baixar as anotações do COCO.")
        try:
            with zipfile.ZipFile(zip_path) as z:               # extrai só o JSON necessário
                with z.open("annotations/instances_val2017.json") as origem, open(json_path, "wb") as f:
                    shutil.copyfileobj(origem, f)
            zip_path.unlink()                                   # libera ~241 MB
            return json_path
        except zipfile.BadZipFile:
            zip_path.unlink()
    raise RuntimeError("O zip de anotações do COCO veio corrompido duas vezes.")


def carregar_coco(json_path: Path | None = None):
    """Carrega as anotações com a API oficial (pycocotools)."""
    from pycocotools.coco import COCO
    return COCO(str(json_path or baixar_anotacoes_coco()))


# ---------------- Seleção (idêntica à CELL 8 da Entrega 1) ----------------
def _objetos_validos(coco, img_id: int, categoria: int, n_min: int, n_max: int, area_min: float) -> bool:
    # Quantidade de objetos da categoria no intervalo e maior objeto (área da MÁSCARA) >= area_min
    info = coco.loadImgs(img_id)[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[categoria], iscrowd=False))
    if not (n_min <= len(anns) <= n_max):
        return False
    return max(a["area"] for a in anns) / (info["width"] * info["height"]) >= area_min


def candidatos_filtro_f0(coco) -> dict[str, list[int]]:
    """Candidatos por grupo com os MESMOS filtros da Entrega 1 (ids ordenados)."""
    gato, carro = config.COCO_ID["gato"], config.COCO_ID["carro"]
    ids_gato, ids_carro = set(coco.getImgIds(catIds=[gato])), set(coco.getImgIds(catIds=[carro]))
    return {
        "gato": [i for i in sorted(ids_gato - ids_carro) if _objetos_validos(coco, i, gato, 1, 2, 0.10)],
        "carro": [i for i in sorted(ids_carro - ids_gato) if _objetos_validos(coco, i, carro, 1, 3, 0.05)],
        "mista": [i for i in sorted(ids_gato & ids_carro) if _objetos_validos(coco, i, gato, 1, 2, 0.03)
                  and _objetos_validos(coco, i, carro, 1, 3, 0.03)],
    }


def reproduzir_manifesto(coco) -> pd.DataFrame:
    """Refaz o sorteio da Entrega 1 (seed 42) — serve para provar que o manifesto é reprodutível."""
    random.seed(config.SEED)                                    # mesmo estado global da CELL 8
    splits = {"gato": (32, 4, 4), "carro": (32, 4, 4), "mista": (3, 1, 1)}
    cand, linhas = candidatos_filtro_f0(coco), []
    for grupo, (n_tr, n_va, n_te) in splits.items():
        escolhidas = random.sample(cand[grupo], n_tr + n_va + n_te)
        divisoes = ["treino"] * n_tr + ["validacao"] * n_va + ["teste"] * n_te
        for img_id, divisao in zip(escolhidas, divisoes):
            info = coco.loadImgs(img_id)[0]
            linhas.append([info["file_name"], grupo, divisao, info["coco_url"], info["flickr_url"]])
    return pd.DataFrame(linhas, columns=["arquivo", "grupo", "divisao", "coco_url", "flickr_url"])


def selecionar_teste_ampliado(coco, manifesto: pd.DataFrame, n_por_classe: int = config.N_TESTE_AMPLIADO,
                              seed: int = config.SEED) -> pd.DataFrame:
    """Sorteia o teste ampliado entre os candidatos F0 que NÃO estão no manifesto original.

    Mesmos filtros da Entrega 1, logo mesma distribuição de imagens. Os 26 carros são todos os
    que sobram no filtro F0; 26 gatos são sorteados entre os 70 restantes para balancear.
    """
    usados = set(manifesto.arquivo)
    gerador = random.Random(seed)                               # gerador próprio: não depende do estado global
    linhas = []
    for grupo in ("gato", "carro"):
        restantes = [i for i in candidatos_filtro_f0(coco)[grupo]
                     if coco.loadImgs(i)[0]["file_name"] not in usados]
        if len(restantes) < n_por_classe:
            raise ValueError(f"Só há {len(restantes)} candidatos novos para '{grupo}'.")
        for img_id in gerador.sample(restantes, n_por_classe):
            info = coco.loadImgs(img_id)[0]
            linhas.append([info["file_name"], grupo, "teste_ampliado", info["coco_url"], info["flickr_url"]])
    return pd.DataFrame(linhas, columns=["arquivo", "grupo", "divisao", "coco_url", "flickr_url"])


# ---------------- Metadados ----------------
def anotar_metadados(df: pd.DataFrame, coco) -> pd.DataFrame:
    """Acrescenta id COCO, licença, presença de pessoa e fração da imagem coberta pelo alvo (GT)."""
    licencas = {l["id"]: l["name"] for l in coco.dataset["licenses"]}
    por_arquivo = {img["file_name"]: img for img in coco.dataset["images"]}
    df = df.copy()
    df["coco_id"] = [por_arquivo[a]["id"] for a in df.arquivo]
    df["licenca"] = [licencas[por_arquivo[a]["license"]] for a in df.arquivo]
    df["permite_derivados"] = ~df.licenca.str.contains("NoDerivs")    # recortes publicáveis?
    df["tem_pessoa"] = [len(coco.getAnnIds(imgIds=i, catIds=[config.COCO_ID["pessoa"]])) > 0
                        for i in df.coco_id]
    df["area_alvo_gt"] = [float(mascara_gt(coco, i).mean()) for i in df.coco_id]
    df["y"] = df.grupo.map({"gato": 0, "carro": 1}).fillna(-1).astype(int)   # mista = -1
    return df


def mascara_gt(coco, img_id: int) -> np.ndarray:
    """União das máscaras GT de gato e carro (não-crowd). Usada SÓ para avaliar e no oráculo."""
    info = coco.loadImgs(img_id)[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False,
                                        catIds=[config.COCO_ID["gato"], config.COCO_ID["carro"]]))
    mascara = np.zeros((info["height"], info["width"]), dtype=bool)
    for ann in anns:
        mascara |= coco.annToMask(ann).astype(bool)
    return mascara


def tabela_imagens(coco, manifesto: pd.DataFrame | None = None,
                   ampliado: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabela única com as 85 imagens das Entregas 1/2 e as do teste ampliado."""
    manifesto = ler_manifesto() if manifesto is None else manifesto
    ampliado = pd.read_csv(config.MANIFESTO_AMPLIADO) if ampliado is None else ampliado
    df = pd.concat([manifesto, ampliado[manifesto.columns]], ignore_index=True)
    df["conjunto"] = df.divisao.map(MAPA_DIVISAO).fillna(df.divisao)
    return anotar_metadados(df, coco)


def ler_manifesto(caminho: Path = config.MANIFESTO_ORIGINAL) -> pd.DataFrame:
    """Manifesto versionado no repositório (85 imagens; não é regravado aqui)."""
    df = pd.read_csv(caminho)
    assert len(df) == 85 and df.arquivo.is_unique, "Manifesto inesperado."
    return df


# ---------------- Imagens ----------------
def baixar_imagens(df: pd.DataFrame, pasta: Path = config.IMAGENS_DIR) -> list[str]:
    """Baixa só as imagens que faltam; devolve a lista de falhas (deve ficar vazia)."""
    falhas = []
    for arquivo, url in zip(df.arquivo, df.coco_url):
        destino = Path(pasta) / arquivo
        if destino.exists() and not _jpeg_integro(destino):     # truncado em execução anterior
            destino.unlink()
        if not destino.exists() and not baixar_arquivo(url, destino):
            falhas.append(arquivo)
    return falhas


def _jpeg_integro(caminho: Path) -> bool:
    try:
        with Image.open(caminho) as img:
            img.load()                                          # decodifica tudo: acusa arquivo truncado
        return True
    except OSError:
        return False


def abrir_imagem(caminho: Path) -> Image.Image:
    return Image.open(caminho).convert("RGB")


def redimensionar(img: Image.Image | np.ndarray, tamanho: int = config.IMG_SIZE) -> np.ndarray:
    """Redimensiona para tamanho x tamanho SEM preservar a proporção (igual ao Resize da Entrega 2)."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return np.asarray(img.convert("RGB").resize((tamanho, tamanho), Image.BILINEAR), dtype=np.uint8)


def carregar_lote(caminhos, tamanho: int = config.IMG_SIZE) -> np.ndarray:
    """Matriz (N, tamanho, tamanho, 3) em float32 no intervalo [0, 255]."""
    return np.stack([redimensionar(abrir_imagem(c), tamanho) for c in caminhos]).astype("float32")


# Condições experimentais: O = original; S = segmentada (fundo preto); C = recorte pela máscara; G = oráculo (GT)
PASTAS_CONDICAO = {"S": "segmentada", "C": "recorte", "G": "oraculo"}


def caminho_condicao(arquivo: str, condicao: str) -> Path:
    if condicao == "O":
        return config.IMAGENS_DIR / arquivo
    return config.CONDICOES_DIR / PASTAS_CONDICAO[condicao] / Path(arquivo).with_suffix(".png").name


def carregar_condicao(tabelas: dict[str, pd.DataFrame], condicao: str) -> dict[str, np.ndarray]:
    """Matrizes de entrada (224x224) de todos os conjuntos em uma condição (S/C/G vêm do notebook 02)."""
    return {c: carregar_lote([caminho_condicao(a, condicao) for a in df.arquivo]) for c, df in tabelas.items()}


def imagens_treino(tabela_treino: pd.DataFrame, condicao: str) -> list[Image.Image]:
    """Imagens de treino em resolução original: a augmentation recorta antes de redimensionar."""
    imagens = []
    for arquivo in tabela_treino.arquivo:
        img = abrir_imagem(caminho_condicao(arquivo, condicao))
        img.load()                                              # lê já (o arquivo é fechado em seguida)
        imagens.append(img)
    return imagens


# ---------------- Checagem de quase-duplicatas ----------------
def dhash(img: Image.Image, lado: int = 8) -> int:
    """Hash perceptual por diferença (dHash, 64 bits): robusto a redimensionamento e compressão."""
    cinza = np.asarray(img.convert("L").resize((lado + 1, lado), Image.BILINEAR), dtype=np.int16)
    bits = (cinza[:, 1:] > cinza[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def menor_distancia_hash(caminhos_a, caminhos_b) -> tuple[int, str, str]:
    """Menor distância de Hamming entre dois grupos de imagens (<= 8 bits indica quase-duplicata)."""
    ha = {Path(c).name: dhash(abrir_imagem(c)) for c in caminhos_a}
    hb = {Path(c).name: dhash(abrir_imagem(c)) for c in caminhos_b}
    return min(((bin(x ^ y).count("1"), a, b) for a, x in ha.items() for b, y in hb.items()))
