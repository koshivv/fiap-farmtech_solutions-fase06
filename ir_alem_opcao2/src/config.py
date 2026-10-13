"""Configuração central do Ir Além — Opção 2.

Todas as escolhas do experimento ficam aqui, fixadas ANTES de olhar os conjuntos de teste
(pré-registro). Alterar um valor deve ser decisão tomada com base apenas em treino/validação.
"""
import os
import sys
from pathlib import Path

# ---------------- Caminhos ----------------
IR_ALEM_DIR = Path(__file__).resolve().parents[1]          # .../ir_alem_opcao2
REPO_DIR = IR_ALEM_DIR.parent                                # raiz do repositório
MANIFESTO_ORIGINAL = REPO_DIR / "dataset" / "manifesto_dataset.csv"   # splits das Entregas 1 e 2
MANIFESTO_AMPLIADO = IR_ALEM_DIR / "dados" / "manifesto_teste_ampliado.csv"
RESULTADOS_REPO = IR_ALEM_DIR / "resultados"                 # CSVs pequenos, versionados
ASSETS_DIR = IR_ALEM_DIR / "assets"                          # figuras usadas no README


def _diretorio_dados() -> Path:
    """Pasta dos dados pesados (imagens, máscaras, pesos), fora do git.

    Ordem: variável FARMTECH_DADOS > Google Drive (Colab) > pasta local ignorada pelo git.
    """
    if os.environ.get("FARMTECH_DADOS"):
        return Path(os.environ["FARMTECH_DADOS"])
    if "google.colab" in sys.modules:
        drive = Path("/content/drive/MyDrive")
        return drive / "FarmTech_Fase6" / "ir_alem" if drive.exists() else Path("/content/ir_alem_dados")
    return IR_ALEM_DIR / "_dados"


DADOS_DIR = _diretorio_dados()
IMAGENS_DIR = DADOS_DIR / "imagens"            # JPEGs originais do COCO (todos os conjuntos)
COCO_DIR = DADOS_DIR / "coco"                  # instances_val2017.json
MASCARAS_DIR = DADOS_DIR / "mascaras"          # máscaras binárias (PNG) por segmentador
CONDICOES_DIR = DADOS_DIR / "condicoes"        # imagens por condição (segmentada, recorte, oráculo)
MODELOS_DIR = DADOS_DIR / "modelos"            # pesos .weights.h5 (não versionados)
RESULTADOS_DIR = DADOS_DIR / "resultados"      # CSVs da execução corrente

# ---------------- Reprodutibilidade ----------------
SEED = 42                                      # mesma semente das Entregas 1 e 2
SEEDS = (42, 43, 44, 45, 46)                   # 5 repetições por cenário: média ± desvio padrão

# ---------------- Dataset ----------------
CLASSES = ("gato", "carro")                    # índice 0 = gato, 1 = carro (igual à Entrega 2)
COCO_ANOTACOES_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_HOST = "images.cocodataset.org"           # único host aceito para downloads
COCO_ID = {"gato": 17, "carro": 3, "pessoa": 1}              # category_id originais do COCO
YOLO80_ID = {"gato": 15, "carro": 2}                         # índices na lista de 80 classes
N_TESTE_AMPLIADO = 26                          # por classe: todos os carros que sobram no filtro F0

# ---------------- Entrada dos classificadores ----------------
IMG_SIZE = 224                                 # resolução nativa da EfficientNetB0; CNN usa a mesma
BATCH = 16                                     # igual à Entrega 2

# ---------------- CNN do zero (port da Entrega 2) ----------------
EPOCAS_CNN = 100
LR_CNN = 1e-3
L2 = 5e-5                                      # = weight_decay 1e-4 acoplado do Adam do PyTorch (wd/2)

# ---------------- Transfer Learning (EfficientNetB0) ----------------
EPOCAS_TL_FASE1 = 30                           # fase 1: base congelada, só a cabeça treina
EPOCAS_TL_FASE2 = 30                           # fase 2: fine tuning do topo da base
LR_TL_FASE1 = 1e-3
LR_TL_FASE2 = 1e-5                             # LR baixa para não destruir os pesos da ImageNet
CAMADA_DESCONGELAR = "block7a_expand_conv"     # início do último bloco MBConv (corte por NOME)

# ---------------- Segmentação ----------------
CLASSES_SEG_ESTRITAS = (2, 15)                                       # car, cat (lista de 80)
CLASSES_SEG_AMPLAS = tuple(range(1, 9)) + tuple(range(14, 24))       # veículos ∪ animais do COCO
CONF_SEG = 0.25                                # mesmo limiar das Entregas 1 e 2
CONF_SEG_RETRY = 0.10                          # 2ª tentativa antes de declarar falha
AREA_MIN_MASCARA = 0.01                        # máscara < 1% da imagem = falha
FUNDO = (0, 0, 0)                              # fundo apagado em preto (sugestão do enunciado)
MARGEM_RECORTE = 0.10                          # folga do recorte pela caixa da máscara (ablação)
SEGMENTADOR_PADRAO = "yolo26n-seg"             # substituído pelo vencedor da seleção (notebook 02)


def garantir_pastas() -> None:
    """Cria as pastas de dados/resultados que ainda não existem."""
    for pasta in (IMAGENS_DIR, COCO_DIR, MASCARAS_DIR, CONDICOES_DIR, MODELOS_DIR,
                  RESULTADOS_DIR, RESULTADOS_REPO, ASSETS_DIR):
        pasta.mkdir(parents=True, exist_ok=True)
