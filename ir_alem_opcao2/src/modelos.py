"""Classificadores do Ir Além: CNN do zero (port Keras da Entrega 2) e EfficientNetB0 (Transfer Learning).

Os dois modelos recebem a MESMA entrada (224x224, pixels em [0, 255]), treinam com a MESMA augmentation
(src/aumento.py, aplicada fora do modelo) e têm a MESMA cabeça (GAP -> Dropout 0,3 -> Dense 2). Isso elimina
diferenças de pipeline; a comparação da H1 continua sendo "arquitetura grande pré-treinada x CNN pequena
treinada do zero", que é a pergunta do enunciado (isolar só o pré-treino exigiria treinar a própria
EfficientNetB0 com pesos aleatórios, o que `criar_modelo_tl(pesos=None)` permite).

Só a API do Keras 3 é usada, então o código roda nos backends tensorflow, torch e jax sem alteração.
"""
import math

import keras
from keras import layers

from . import config


# ---------------- Inicialização igual à do PyTorch ----------------
def _init_pytorch(fan_in: int):
    """Inicializadores equivalentes aos padrões de nn.Conv2d / nn.Linear do PyTorch.

    kaiming_uniform(a=√5) => U(-1/√fan_in, 1/√fan_in) = VarianceScaling(1/3, fan_in, uniform);
    o bias usa o mesmo limite. Mantém a CNN portada o mais próxima possível da Entrega 2.
    """
    limite = 1.0 / math.sqrt(fan_in)
    return (keras.initializers.VarianceScaling(1 / 3, "fan_in", "uniform"),
            keras.initializers.RandomUniform(-limite, limite))


def _cabeca(x, seed, l2, fan_in: int):
    """Cabeça comum aos dois modelos: GAP -> Dropout(0,3) -> Dense(2) com logits."""
    reg = keras.regularizers.L2(l2)
    kernel_init, bias_init = _init_pytorch(fan_in)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, seed=seed, name="dropout")(x)
    return layers.Dense(len(config.CLASSES), kernel_initializer=kernel_init, bias_initializer=bias_init,
                        kernel_regularizer=reg, bias_regularizer=reg, name="logits")(x)


# ---------------- CNN do zero ----------------
def criar_cnn_do_zero(tamanho: int = config.IMG_SIZE, seed=None, l2: float = config.L2) -> keras.Model:
    """Port fiel da CNN da Entrega 2 (241.794 parâmetros treináveis).

    Ajustes para paridade com o PyTorch: BN com momentum 0,9 / eps 1e-5 (padrão PyTorch 0,1 / 1e-5)
    e L2 = 5e-5 em todos os pesos, equivalente ao weight_decay=1e-4 acoplado do Adam do PyTorch.
    """
    reg = keras.regularizers.L2(l2)
    entrada = keras.Input((tamanho, tamanho, 3), name="imagem")
    x = layers.Rescaling(1 / 127.5, offset=-1.0, name="normalizacao")(entrada)   # = Normalize(0,5; 0,5)
    canais = 3
    for i, filtros in enumerate((32, 64, 128, 128), start=1):
        kernel_init, bias_init = _init_pytorch(3 * 3 * canais)
        x = layers.Conv2D(filtros, 3, padding="same", kernel_initializer=kernel_init, bias_initializer=bias_init,
                          kernel_regularizer=reg, bias_regularizer=reg, name=f"bloco{i}_conv")(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_regularizer=reg, beta_regularizer=reg,
                                      name=f"bloco{i}_bn")(x)
        x = layers.ReLU(name=f"bloco{i}_relu")(x)
        x = layers.MaxPooling2D(2, name=f"bloco{i}_pool")(x)
        canais = filtros
    return keras.Model(entrada, _cabeca(x, seed, l2, canais), name="cnn_do_zero")


# ---------------- Transfer Learning ----------------
def criar_modelo_tl(tamanho: int = config.IMG_SIZE, seed=None, l2: float = config.L2,
                    pesos: str | None = "imagenet") -> tuple[keras.Model, keras.Model]:
    """EfficientNetB0 pré-treinada na ImageNet com a base congelada (fase 1).

    A EfficientNet já traz a normalização dentro do modelo e espera pixels crus em [0, 255]:
    não chamamos preprocess_input (ele é identidade).

    A base deve se comportar como em inferência durante o treino (guia de transfer learning do Keras):
    - as BatchNormalization ficam congeladas (trainable=False) e, assim, usam as estatísticas da ImageNet;
    - o *stochastic depth* da EfficientNet (Dropout dentro dos blocos) é desligado (rate = 0). No Keras 3,
      o `training=False` passado à base aninhada é sobrescrito pelo `training=True` do fit, então esses
      Dropout ficariam ativos e a "feature extraction" da fase 1 veria features ruidosas.
    """
    entrada = keras.Input((tamanho, tamanho, 3), name="imagem")
    base = keras.applications.EfficientNetB0(include_top=False, weights=pesos,
                                             input_shape=(tamanho, tamanho, 3))
    base.trainable = False
    for camada in base.layers:                                  # desliga o stochastic depth
        if isinstance(camada, layers.Dropout):
            camada.rate = 0.0
    x = base(entrada, training=False)
    modelo = keras.Model(entrada, _cabeca(x, seed, l2, base.output.shape[-1]), name="tl_efficientnetb0")
    return modelo, base


def descongelar_topo(base: keras.Model, a_partir_de: str = config.CAMADA_DESCONGELAR) -> int:
    """Fase 2: libera o treino a partir da camada `a_partir_de` (por NOME), mantendo as BN congeladas.

    O corte é por nome porque o índice muda com os pesos (237 camadas sem pesos, 238 com
    'imagenet', que acrescenta uma Rescaling). Devolve o índice da primeira camada liberada.
    """
    nomes = [camada.name for camada in base.layers]
    inicio = nomes.index(a_partir_de)
    base.trainable = True
    for i, camada in enumerate(base.layers):
        camada.trainable = i >= inicio and not isinstance(camada, layers.BatchNormalization)
    return inicio


def contar_parametros(modelo: keras.Model) -> dict[str, int]:
    """Parâmetros treináveis e congelados (BN: média/variância móveis contam como congelados)."""
    def soma(pesos):
        return int(sum(math.prod(p.shape) for p in pesos))
    return {"treinaveis": soma(modelo.trainable_weights), "congelados": soma(modelo.non_trainable_weights)}


def modelo_cam(modelo: keras.Model) -> keras.Model:
    """Submodelo que devolve (mapas de ativação antes do GAP, logits), para o CAM.

    Como a cabeça é GAP -> Dense, o Class Activation Map (Zhou et al., 2016) é exato e não precisa
    de gradientes: CAM_c = soma_k W[k, c] * A_k. Funciona igual em qualquer backend.
    """
    return keras.Model(modelo.input, [modelo.get_layer("gap").input, modelo.output])
