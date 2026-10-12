"""Segmentação automática do objeto de interesse e utilidades de máscara.

Regra anti-vazamento: o segmentador escolhe QUAIS PIXELS ficam, mas a classe que ele prevê é
descartada. Ela só é registrada para o diagnóstico "o segmentador sozinho já classifica?".
A política de falha é a mesma para gatos e carros e foi fixada antes de olhar os testes.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

# nome -> (pesos do detector/segmentador YOLO, pesos do SAM ou None)
CANDIDATOS = {
    "yolo11s-seg": ("yolo11s-seg.pt", None),
    "yolo26n-seg": ("yolo26n-seg.pt", None),
    "yolo26n-seg+sam2.1_b": ("yolo26n-seg.pt", "sam2.1_b.pt"),
}
NOMES_ALVO = {config.YOLO80_ID["gato"]: "gato", config.YOLO80_ID["carro"]: "carro"}


@dataclass
class ResultadoSegmentacao:
    mascara: np.ndarray | None              # bool (H, W); None quando a segmentação falhou
    status: str                             # "ok", "ok_conf_reduzida" ou "falha"
    area: float                             # fração da imagem coberta pela máscara final
    classes_detectadas: list[str] = field(default_factory=list)   # só para diagnóstico
    n_deteccoes: int = 0


class Segmentador:
    """YOLO-seg pré-treinado no COCO (opcionalmente com as caixas refinadas pelo SAM 2.1).

    Os pesos são baixados automaticamente pelo pacote ultralytics para `pasta_pesos`.
    """

    def __init__(self, nome: str, classes=config.CLASSES_SEG_ESTRITAS, pasta_pesos: Path = config.MODELOS_DIR):
        from ultralytics import SAM, YOLO
        pesos_yolo, pesos_sam = CANDIDATOS[nome]
        Path(pasta_pesos).mkdir(parents=True, exist_ok=True)
        self.nome, self.classes = nome, list(classes)
        self.yolo = YOLO(str(Path(pasta_pesos) / pesos_yolo))
        self.sam = SAM(str(Path(pasta_pesos) / pesos_sam)) if pesos_sam else None

    def mascara(self, img: Image.Image, conf: float) -> tuple[np.ndarray, list[str], int]:
        """Máscara binária (união de todas as instâncias), classes-alvo detectadas e nº de detecções."""
        # PIL é interpretado como RGB pelo ultralytics (arrays numpy seriam lidos como BGR)
        r = self.yolo.predict(img, conf=conf, classes=self.classes, retina_masks=True, verbose=False)[0]
        vazia = np.zeros((img.height, img.width), dtype=bool)
        n = 0 if r.boxes is None else len(r.boxes)
        if n == 0:
            return vazia, [], 0
        nomes = sorted({NOMES_ALVO[c] for c in map(int, r.boxes.cls.tolist()) if c in NOMES_ALVO})
        if self.sam is None:
            mascara = r.masks.data.bool().any(0).cpu().numpy()  # retina_masks: já no tamanho original
        else:
            caixas = r.boxes.xyxy.cpu().numpy().tolist()
            # conf=0.0: o SAM descartaria máscaras com IoU previsto baixo e perderíamos caixas
            rs = self.sam(img, bboxes=caixas, conf=0.0, verbose=False)[0]
            mascara = vazia if rs.masks is None else rs.masks.data.bool().any(0).cpu().numpy()
        assert mascara.shape == vazia.shape, "Máscara fora do tamanho da imagem original."
        return mascara, nomes, n


def segmentar(segmentador, img: Image.Image, area_min: float = config.AREA_MIN_MASCARA,
              conf: float = config.CONF_SEG, conf_retry: float = config.CONF_SEG_RETRY) -> ResultadoSegmentacao:
    """Aplica a política de falha (igual para todas as imagens e classes).

    1) segmenta com conf 0,25; 2) se a máscara cobrir < 1% da imagem, tenta com conf 0,10;
    3) se ainda falhar, devolve mascara=None e a imagem original será usada (fallback).
    """
    mascara, nomes, n = segmentador.mascara(img, conf)
    if mascara.mean() >= area_min:
        return ResultadoSegmentacao(mascara, "ok", float(mascara.mean()), nomes, n)
    mascara, nomes, n = segmentador.mascara(img, conf_retry)
    if mascara.mean() >= area_min:
        return ResultadoSegmentacao(mascara, "ok_conf_reduzida", float(mascara.mean()), nomes, n)
    return ResultadoSegmentacao(None, "falha", float(mascara.mean()), nomes, n)


# ---------------- Operações com máscaras ----------------
def aplicar_mascara(img: np.ndarray, mascara: np.ndarray, fundo=config.FUNDO) -> np.ndarray:
    """Mantém só os pixels do objeto e pinta o resto com a cor de fundo (preto por padrão)."""
    saida = np.empty_like(img)
    saida[:] = fundo
    saida[mascara] = img[mascara]
    return saida


def recortar_pela_mascara(img: np.ndarray, mascara: np.ndarray, margem: float = config.MARGEM_RECORTE) -> np.ndarray:
    """Recorta a caixa envolvente da máscara com folga (ablação: amplia o objeto)."""
    linhas, colunas = np.where(mascara)
    y0, y1, x0, x1 = linhas.min(), linhas.max() + 1, colunas.min(), colunas.max() + 1
    dy, dx = int((y1 - y0) * margem), int((x1 - x0) * margem)
    altura, largura = mascara.shape
    return img[max(0, y0 - dy):min(altura, y1 + dy), max(0, x0 - dx):min(largura, x1 + dx)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersecção sobre união entre duas máscaras binárias."""
    uniao = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / uniao) if uniao else 1.0


def salvar_mascara(mascara: np.ndarray, caminho: Path) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mascara.astype(np.uint8) * 255).save(caminho)


def ler_mascara(caminho: Path) -> np.ndarray:
    return np.asarray(Image.open(caminho)) > 127


# ---------------- Ablação: detector da Entrega 1 -> SAM ----------------
class SegmentadorEntrega1:
    """Caixas da YOLOv5 customizada da Entrega 1 (repo clássico) como prompt do SAM 2.1.

    O best.pt do repo clássico só carrega com torch.hub (o ultralytics.YOLO() devolve lixo em
    silêncio quando o repo yolov5 está no sys.path). Como esse detector foi TREINADO no nosso split
    de treino, esperamos caixas melhores no treino que no teste: é isso que a ablação mede.
    """

    def __init__(self, pesos_e1: Path, repo_yolov5: Path, pasta_pesos: Path = config.MODELOS_DIR):
        import torch
        from ultralytics import SAM
        self.detector = torch.hub.load(str(repo_yolov5), "custom", path=str(pesos_e1), source="local",
                                       verbose=False)
        self.detector.conf = config.CONF_SEG
        self.sam = SAM(str(Path(pasta_pesos) / "sam2.1_b.pt"))

    def mascara(self, img: Image.Image, conf: float) -> tuple[np.ndarray, list[str], int]:
        self.detector.conf = conf
        det = self.detector(img, size=640).xyxy[0].cpu().numpy()       # x1, y1, x2, y2, conf, classe
        vazia = np.zeros((img.height, img.width), dtype=bool)
        if len(det) == 0:
            return vazia, [], 0
        nomes = sorted({config.CLASSES[int(c)] for c in det[:, 5]})
        rs = self.sam(img, bboxes=det[:, :4].tolist(), conf=0.0, verbose=False)[0]
        return (vazia if rs.masks is None else rs.masks.data.bool().any(0).cpu().numpy()), nomes, len(det)
