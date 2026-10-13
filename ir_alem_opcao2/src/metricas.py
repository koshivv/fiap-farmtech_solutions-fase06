"""Métricas de classificação, testes estatísticos e medição de latência."""
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import config


def tabela_predicoes(probs: np.ndarray, df: pd.DataFrame, **meta) -> pd.DataFrame:
    """Uma linha por imagem: probabilidade de 'carro', classe prevista e acerto, mais metadados."""
    tabela = pd.DataFrame({"arquivo": df.arquivo.values, "grupo": df.grupo.values, "y": df.y.values,
                           "tem_pessoa": df.tem_pessoa.values, "prob_carro": probs[:, 1]})
    tabela["pred"] = (tabela.prob_carro >= 0.5).astype(int)
    tabela["correto"] = tabela.pred == tabela.y
    for chave, valor in meta.items():
        tabela.insert(0, chave, valor)
    return tabela


def metricas_basicas(df: pd.DataFrame) -> dict:
    """Acurácia, P/R/F1 por classe, F1 macro e métricas de probabilidade (contornam o efeito teto).

    log-loss, Brier e P(classe correta) distinguem modelos que acertam as mesmas imagens com
    confiança diferente, o que a acurácia não enxerga quando já está perto de 100%.
    """
    y, pred = df.y.to_numpy(), df.pred.to_numpy()
    p_carro = np.clip(df.prob_carro.to_numpy(), 1e-7, 1 - 1e-7)
    p_correta = np.where(y == 1, p_carro, 1 - p_carro)
    precisao, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    linha = {"acuracia": float((y == pred).mean()), "f1_macro": float(f1.mean()),
             "log_loss": float(-np.log(p_correta).mean()), "brier": float(((p_carro - y) ** 2).mean()),
             "p_correta": float(p_correta.mean()), "n": len(df)}
    for i, classe in enumerate(config.CLASSES):
        linha |= {f"precisao_{classe}": precisao[i], f"recall_{classe}": recall[i], f"f1_{classe}": f1[i]}
    return linha


def metricas_por_semente(predicoes: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    """Métricas de cada execução (combinação de `chaves` + seed)."""
    linhas = [{**dict(zip(chaves + ["seed"], k)), **metricas_basicas(g)}
              for k, g in predicoes.groupby(chaves + ["seed"], sort=False)]
    return pd.DataFrame(linhas)


def media_desvio(por_semente: pd.DataFrame, chaves: list[str], colunas: list[str], casas: int = 3) -> pd.DataFrame:
    """Tabela 'média ± desvio padrão' entre sementes."""
    agrupado = por_semente.groupby(chaves, sort=False)[colunas]
    media, desvio = agrupado.mean(), agrupado.std(ddof=1).fillna(0)
    return media.round(casas).astype(str) + " ± " + desvio.round(casas).astype(str)


def predicao_consenso(predicoes: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    """Consenso entre sementes (média das probabilidades): uma previsão por imagem e configuração.

    Usado no teste pareado: compara configurações imagem a imagem sem depender de uma semente.
    """
    consenso = (predicoes.groupby(chaves + ["arquivo", "grupo", "y", "tem_pessoa"], sort=False)
                .prob_carro.mean().reset_index())
    consenso["pred"] = (consenso.prob_carro >= 0.5).astype(int)
    consenso["correto"] = consenso.pred == consenso.y
    return consenso


def mcnemar_exato(correto_a, correto_b) -> dict:
    """Teste de McNemar exato (binomial) para dois classificadores avaliados nas MESMAS imagens.

    b = só A acerta; c = só B acerta. H0: b e c têm a mesma chance. Bilateral.
    """
    a, b_ = np.asarray(correto_a, bool), np.asarray(correto_b, bool)
    b, c = int((a & ~b_).sum()), int((~a & b_).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return {"so_A_acerta": b, "so_B_acerta": c, "p_valor": p}


def holm(p_valores) -> list[float]:
    """p-valores ajustados por Holm-Bonferroni (controla o erro de ter ao menos um falso positivo
    quando várias comparações respondem à mesma hipótese)."""
    p = np.asarray(p_valores, float)
    ajustados, acumulado = np.empty(len(p)), 0.0
    for posicao, i in enumerate(np.argsort(p)):
        acumulado = max(acumulado, min(1.0, (len(p) - posicao) * p[i]))
        ajustados[i] = acumulado
    return ajustados.round(4).tolist()


def ic_bootstrap(correto, n: int = 10_000, seed: int = config.SEED, alfa: float = 0.05) -> tuple[float, float]:
    """Intervalo de confiança (percentil) da acurácia por reamostragem das imagens."""
    correto = np.asarray(correto, float)
    gerador = np.random.default_rng(seed)
    amostras = gerador.choice(correto, size=(n, len(correto)), replace=True).mean(axis=1)
    return round(float(np.quantile(amostras, alfa / 2)), 3), round(float(np.quantile(amostras, 1 - alfa / 2)), 3)


def matriz_confusao(df: pd.DataFrame) -> np.ndarray:
    """Matriz 2x2 (linhas = real, colunas = previsto; ordem gato, carro)."""
    return confusion_matrix(df.y, df.pred, labels=[0, 1])


def medir_latencia(funcao, entradas, repeticoes: int = 5, aquecimento: int = 3) -> tuple[float, float]:
    """Média e desvio (ms) por chamada, uma entrada por vez, após aquecimento descartado.

    A função deve devolver um resultado já materializado (ex.: numpy), o que força a sincronização
    com a GPU em qualquer backend — mesmo papel do torch.cuda.synchronize da Entrega 2.
    """
    for entrada in entradas[:aquecimento]:
        funcao(entrada)
    tempos = []
    for _ in range(repeticoes):
        for entrada in entradas:
            inicio = time.perf_counter()
            funcao(entrada)
            tempos.append((time.perf_counter() - inicio) * 1000)
    return float(np.mean(tempos)), float(np.std(tempos))


def limiar_por_area(area_treino, y_treino) -> tuple[float, bool]:
    """Classificador trivial que só vê a fração visível da imagem segmentada: um único limiar.

    Mede o 'atalho da área': se ele acertar muito, parte do ganho da segmentação pode vir do tamanho
    da mancha, e não da aparência do objeto. O limiar (e o sentido) que mais acerta no TREINO é
    escolhido entre os pontos médios das áreas observadas. Devolve (limiar, carro_abaixo).
    """
    area, y = np.asarray(area_treino, float), np.asarray(y_treino, int)
    valores = np.unique(area)
    melhor = (-1.0, float(valores[0]), True)
    for limiar in (valores[:-1] + valores[1:]) / 2:
        for carro_abaixo in (True, False):
            acuracia = (prever_por_area(area, limiar, carro_abaixo) == y).mean()
            if acuracia > melhor[0]:
                melhor = (acuracia, float(limiar), carro_abaixo)
    return melhor[1], melhor[2]


def prever_por_area(area, limiar: float, carro_abaixo: bool) -> np.ndarray:
    """Aplica a regra aprendida por limiar_por_area (1 = carro, 0 = gato)."""
    area = np.asarray(area, float)
    return ((area < limiar) if carro_abaixo else (area >= limiar)).astype(int)
