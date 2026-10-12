"""Treino dos classificadores com a mesma política de checkpoint da Entrega 2.

Política: guarda os pesos da época com MENOR entropia cruzada na validação e os restaura ao fim.
A entropia cruzada é monitorada sem o termo L2 (como a perda de validação do PyTorch).
"""
import hashlib
import math
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from PIL import Image

from . import aumento, config, metricas, modelos


# ---------------- Dados de treino com augmentation ----------------
class LotesAumentados(keras.utils.PyDataset):
    """Lotes de treino com a augmentation da Entrega 2, sorteada a cada época.

    Recebe as imagens PIL em resolução original (o RandomResizedCrop recorta antes de redimensionar,
    como no torchvision). O sorteio de cada imagem depende só de (semente, época, índice), então o
    treino é reprodutível independentemente da ordem em que o Keras pede os lotes.
    """

    def __init__(self, imagens: list[Image.Image], y, seed: int, batch: int = config.BATCH):
        super().__init__(workers=1, use_multiprocessing=False)
        self.imagens, self.y, self.seed, self.batch = imagens, np.asarray(y), seed, batch
        self.epoca = 0
        self._sortear_ordem()

    def _sortear_ordem(self):
        self.ordem = np.random.default_rng([self.seed, self.epoca]).permutation(len(self.imagens))

    def __len__(self):
        return math.ceil(len(self.imagens) / self.batch)

    def __getitem__(self, indice):
        selecao = self.ordem[indice * self.batch:(indice + 1) * self.batch]
        lote = [aumento.aumentar(self.imagens[j], np.random.default_rng([self.seed, self.epoca, int(j)]))
                for j in selecao]
        return np.stack(lote).astype("float32"), self.y[selecao]

    def on_epoch_end(self):
        self.epoca += 1                                         # nova ordem e novos sorteios
        self._sortear_ordem()


# ---------------- Checkpoint ----------------
def capturar_pesos(modelo: keras.Model) -> dict[str, np.ndarray]:
    """Cópia dos pesos indexada pelo caminho da variável.

    Indexar por caminho (e não por posição) evita trocar pesos de lugar quando o atributo
    `trainable` muda entre as fases do fine tuning, o que reordena modelo.weights.
    """
    return {v.path: np.array(keras.ops.convert_to_numpy(v)) for v in modelo.weights}


def restaurar_pesos(modelo: keras.Model, pesos: dict[str, np.ndarray]) -> None:
    for v in modelo.weights:
        v.assign(pesos[v.path])


class MelhoresPesos(keras.callbacks.Callback):
    """Checkpoint em memória: restaura, ao fim do treino, a época com menor valor monitorado."""

    def __init__(self, monitor: str = "val_entropia"):
        super().__init__()
        self.monitor = monitor

    def on_train_begin(self, logs=None):
        self.melhor, self.melhor_epoca, self.pesos = np.inf, None, None

    def on_epoch_end(self, epoch, logs=None):
        valor = (logs or {})[self.monitor]
        if valor < self.melhor:
            self.melhor, self.melhor_epoca = float(valor), epoch + 1
            self.pesos = capturar_pesos(self.model)

    def on_train_end(self, logs=None):
        if self.pesos is not None:
            restaurar_pesos(self.model, self.pesos)


def _compilar(modelo: keras.Model, lr: float) -> None:
    # epsilon 1e-8 = padrão do Adam do PyTorch (o do Keras é 1e-7)
    modelo.compile(optimizer=keras.optimizers.Adam(lr, epsilon=1e-8),
                   loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=["accuracy", keras.metrics.SparseCategoricalCrossentropy(from_logits=True,
                                                                                    name="entropia")])


def _historico(hist: keras.callbacks.History, fase: str, epoca_inicial: int = 0) -> pd.DataFrame:
    df = pd.DataFrame(hist.history)
    df.insert(0, "epoca", np.arange(epoca_inicial + 1, epoca_inicial + len(df) + 1))
    df.insert(1, "fase", fase)
    return df


def prever_probabilidades(modelo: keras.Model, X: np.ndarray, batch: int = 32) -> np.ndarray:
    """Probabilidades (N, 2) via softmax dos logits."""
    logits = modelo.predict(X, batch_size=batch, verbose=0)
    logits = logits - logits.max(axis=1, keepdims=True)         # estabilidade numérica
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------- Laços de treino ----------------
def treinar_cnn(imagens_treino, y_tr, X_va, y_va, seed: int, epocas: int = config.EPOCAS_CNN) -> dict:
    """CNN do zero: Adam 1e-3, 100 épocas, batch 16, checkpoint pela menor entropia de validação."""
    keras.utils.set_random_seed(seed)                           # python, numpy e backend
    modelo = modelos.criar_cnn_do_zero(seed=seed)
    _compilar(modelo, config.LR_CNN)
    checkpoint = MelhoresPesos()
    inicio = time.perf_counter()
    hist = modelo.fit(LotesAumentados(imagens_treino, y_tr, seed), epochs=epocas, validation_data=(X_va, y_va),
                      callbacks=[checkpoint], verbose=0)
    return {"modelo": modelo, "historico": _historico(hist, "unica"), "melhor_epoca": checkpoint.melhor_epoca,
            "tempo_s": time.perf_counter() - inicio, "parametros": modelos.contar_parametros(modelo)}


def treinar_tl(imagens_treino, y_tr, X_va, y_va, seed: int, avaliar_em: dict[str, np.ndarray] | None = None,
               epocas_fase1: int = config.EPOCAS_TL_FASE1, epocas_fase2: int = config.EPOCAS_TL_FASE2) -> dict:
    """Transfer Learning em duas fases.

    Fase 1 (feature extraction): base congelada, só a cabeça treina (Adam 1e-3).
    Fase 2 (fine tuning): libera de `block7a_expand_conv` até `top_conv` (BN congeladas), recompila
    com Adam 1e-5 e continua a partir do melhor checkpoint da fase 1. O modelo final é o melhor
    checkpoint da fase 2. `avaliar_em` recebe conjuntos para prever ao fim da fase 1 (ablação
    feature extraction x fine tuning).
    """
    keras.utils.set_random_seed(seed)
    modelo, base = modelos.criar_modelo_tl(seed=seed)
    lotes = LotesAumentados(imagens_treino, y_tr, seed)        # a fase 2 continua a sequência de épocas
    inicio = time.perf_counter()

    _compilar(modelo, config.LR_TL_FASE1)
    cp1 = MelhoresPesos()
    h1 = modelo.fit(lotes, epochs=epocas_fase1, validation_data=(X_va, y_va), callbacks=[cp1], verbose=0)
    params_fase1 = modelos.contar_parametros(modelo)
    probs_fase1 = {nome: prever_probabilidades(modelo, X) for nome, X in (avaliar_em or {}).items()}

    camada_inicial = modelos.descongelar_topo(base)
    _compilar(modelo, config.LR_TL_FASE2)                       # obrigatório após mudar `trainable`
    cp2 = MelhoresPesos()
    h2 = modelo.fit(lotes, initial_epoch=epocas_fase1, epochs=epocas_fase1 + epocas_fase2,
                    validation_data=(X_va, y_va), callbacks=[cp2], verbose=0)

    historico = pd.concat([_historico(h1, "fase1"), _historico(h2, "fase2", epocas_fase1)], ignore_index=True)
    return {"modelo": modelo, "base": base, "historico": historico,
            "melhor_epoca_fase1": cp1.melhor_epoca, "melhor_epoca": cp2.melhor_epoca,
            "tempo_s": time.perf_counter() - inicio, "probs_fase1": probs_fase1,
            "parametros_fase1": params_fase1, "parametros": modelos.contar_parametros(modelo),
            "camada_inicial_fase2": camada_inicial}


# ---------------- Execuções do experimento (com cache) ----------------
ARQUIVOS_DO_TREINO = ("aumento.py", "config.py", "dados.py", "modelos.py", "treino.py")


def assinatura(condicao: str) -> str:
    """Impressão digital do que determina uma execução: código de treino, backend, versão do Keras e,
    nas condições S e C, o segmentador escolhido e as máscaras geradas no notebook 02.

    Se qualquer um muda, o cache antigo deixa de valer e o treino é refeito.
    """
    h = hashlib.sha256()
    for nome in ARQUIVOS_DO_TREINO:
        h.update((Path(__file__).parent / nome).read_bytes())
    h.update(f"{keras.config.backend()}|{keras.__version__}".encode())
    if condicao in ("S", "C"):
        for arquivo in ("segmentador_escolhido.json", "segmentacao_imagens.csv"):
            caminho = config.RESULTADOS_DIR / arquivo
            h.update(caminho.read_bytes() if caminho.exists() else b"ausente")
    return h.hexdigest()[:12]


def executar_com_cache(nome: str, condicao: str, funcao) -> dict[str, pd.DataFrame]:
    """Roda `funcao()` só se não houver uma execução `nome` com a MESMA assinatura e com os pesos salvos.

    Permite retomar o experimento depois de uma queda de sessão do Colab sem refazer o que já
    terminou (os CSVs ficam no Drive junto com os pesos), sem reaproveitar resultados de código antigo.
    """
    marca = assinatura(condicao)
    pasta = config.RESULTADOS_DIR / "execucoes" / marca
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos = {k: pasta / f"{nome}__{k}.csv" for k in ("predicoes", "historico", "execucao")}
    selo = caminho_pesos(nome).with_suffix(".assinatura")      # qual assinatura gerou os pesos
    if all(a.exists() for a in arquivos.values()) and selo.exists() and selo.read_text() == marca:
        return {k: pd.read_csv(a) for k, a in arquivos.items()}
    saida = funcao()
    selo.write_text(marca)
    for chave, df in saida.items():
        df.to_csv(arquivos[chave], index=False)
    return saida


def executar(tipo: str, condicao: str, seed: int, imagens_treino, X: dict, tabelas: dict,
             avaliar=("validacao", "teste_original", "teste_ampliado")) -> dict[str, pd.DataFrame]:
    """Treina um classificador ('cnn' ou 'tl') em uma condição e avalia nos conjuntos indicados.

    Treino e avaliação usam a MESMA condição (ex.: segmentada/segmentada), evitando mudança de
    distribuição entre treino e teste. Salva os pesos e devolve predições, histórico e resumo.
    """
    nome = f"{tipo}_{condicao}_s{seed}"

    def rodar():
        keras.backend.clear_session()                           # libera memória entre execuções
        y_tr, y_va = tabelas["treino"].y.to_numpy(), tabelas["validacao"].y.to_numpy()
        if tipo == "cnn":
            r = treinar_cnn(imagens_treino, y_tr, X["validacao"], y_va, seed)
        else:
            r = treinar_tl(imagens_treino, y_tr, X["validacao"], y_va, seed, avaliar_em={c: X[c] for c in avaliar})
        salvar_pesos(r["modelo"], nome)
        predicoes = []
        for conjunto in avaliar:
            meta = dict(modelo=tipo, condicao_treino=condicao, condicao_teste=condicao, seed=seed, conjunto=conjunto)
            probs = prever_probabilidades(r["modelo"], X[conjunto])
            predicoes.append(metricas.tabela_predicoes(probs, tabelas[conjunto], fase="final", **meta))
            if tipo == "tl":                                    # mesma rede antes do fine tuning
                predicoes.append(metricas.tabela_predicoes(r["probs_fase1"][conjunto], tabelas[conjunto],
                                                           fase="fase1", **meta))
        resumo = {"modelo": tipo, "condicao": condicao, "seed": seed, "melhor_epoca": r["melhor_epoca"],
                  "melhor_epoca_fase1": r.get("melhor_epoca_fase1"), "tempo_treino_s": round(r["tempo_s"], 1),
                  "parametros_treinaveis": r["parametros"]["treinaveis"],
                  "parametros_treinaveis_fase1": r.get("parametros_fase1", {}).get("treinaveis")}
        historico = r["historico"].assign(modelo=tipo, condicao=condicao, seed=seed)
        return {"predicoes": pd.concat(predicoes, ignore_index=True), "historico": historico,
                "execucao": pd.DataFrame([resumo])}

    return executar_com_cache(nome, condicao, rodar)


def avaliar_salvo(tipo: str, nome: str, X: dict, tabelas: dict, condicao_treino: str, condicao_teste: str,
                  seed: int, avaliar=("teste_original", "teste_ampliado")) -> pd.DataFrame:
    """Avalia um modelo já treinado em outra condição (teste de mudança de distribuição)."""
    keras.backend.clear_session()
    modelo = carregar_modelo(tipo, nome)
    return pd.concat([metricas.tabela_predicoes(prever_probabilidades(modelo, X[c]), tabelas[c], fase="final",
                                                modelo=tipo, condicao_treino=condicao_treino,
                                                condicao_teste=condicao_teste, seed=seed, conjunto=c)
                      for c in avaliar], ignore_index=True)


# ---------------- Pesos ----------------
def caminho_pesos(nome: str):
    return config.MODELOS_DIR / f"{nome}.weights.h5"


def salvar_pesos(modelo: keras.Model, nome: str) -> None:
    config.MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    modelo.save_weights(caminho_pesos(nome))


def carregar_modelo(tipo: str, nome: str) -> keras.Model:
    """Recria a arquitetura e carrega os pesos salvos (tipo: 'cnn' ou 'tl')."""
    modelo = modelos.criar_cnn_do_zero() if tipo == "cnn" else modelos.criar_modelo_tl()[0]
    modelo.load_weights(caminho_pesos(nome))
    return modelo
