"""Mixture model using EM"""

from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    n, d = X.shape  # num de pontos e dimensões
    K = mixture.mu.shape[0]  # num de componentes da mistura

    mu = mixture.mu  # mu[j] = média do componente j
    var = mixture.var  # var[j] = variância esférica do componente j
    p = mixture.p  # p[j] = peso do componente j

    post = np.zeros((n, K))  # Cria a matriz de responsabilidades (posterior)
    const = (2 * np.pi) ** (
        d / 2
    )  # Parte da constante da Gaussiana multivarada 1/ (2π)^(d/2)
    for i in range(n):
        for j in range(K):
            dist = X[i] - mu[j]  # Quadrado da distancia euclidiana por dimensão
            exponent = -np.dot(dist, dist) / (
                2 * var[j]
            )  # - ||x_i - mu_j||^2 / (2 σ_j^2)
            post[i, j] = (
                p[j] * np.exp(exponent) / (const * (var[j] ** (d / 2)))
            )  # Multiplica pela proporção do cluster e divide pela constante da gaussiana multivariada
        post[i] /= np.sum(
            post[i]
        )  # Normaliza para virar uma distribuição de probabilidade onde a soma das responsabilidades seja 1 | sum p(j|x_i) = 1

    # Calcula a log-Likelihood
    L = 0.0
    for i in range(n):
        s = 0.0
        for j in range(K):
            dist = X[i] - mu[j]
            exponent = -np.dot(dist, dist) / (2 * var[j])
            s += p[j] * np.exp(exponent) / (const * (var[j] ** (d / 2)))
        L += np.log(s)
    return post, L
    """E-step: Atribui de forma suave cada ponto de dados a um componente gaussiano

    Args:
                X: array (n, d) contendo os dados
                mixture: a mistura gaussiana atual

    Returns:
                np.ndarray: array (n, K) contendo as contagens suaves
                        para todos os componentes e todos os exemplos
                float: log-verossimilhança da atribuição
    """


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    n, d = X.shape  # num de pontos e dimensões
    K = post.shape[1]  # num de componentes

    nk = np.sum(
        post, axis=0
    )  # Soma das responsabilidades para cada cluster nk[j] = “quantos pontos pertencem ao cluster j” (suavemente)

    p = nk / n  # Peso do cluster j = fração de pontos atribuídos a ele.
    mu = (post.T @ X) / nk.reshape(
        -1, 1
    )  # Média do cluster j = média ponderada dos pontos atribuídos a ele (suavemente)
    var = np.zeros(K)  # Variância do cluster j = variância esférica
    for j in range(K):
        dist = X - mu[j]
        var[j] = np.sum(post[:, j] * np.sum(dist**2, axis=1)) / (nk[j] * d)
    return GaussianMixture(mu, var, p)
    """M-step: Atualiza a mistura gaussiana maximizando a log-verossimilhança
            do conjunto de dados ponderado

    Args:
                X: array (n, d) contendo os dados
                post: array (n, K) contendo as contagens suaves
                        para todos os componentes e todas as amostras

    Returns:
                GaussianMixture: a nova mistura gaussiana
    """


def run(
    X: np.ndarray, mixture: GaussianMixture, post: np.ndarray
) -> Tuple[GaussianMixture, np.ndarray, float]:
    old_L = None
    while True:
        post, L = estep(X, mixture)
        mixture = mstep(X, post)
        post, L = estep(X, mixture)
        if old_L is not None:
            if L - old_L <= 1e-6 * abs(L):
                break
        old_L = L
    return mixture, post, L
    """Runs the mixture model

    Args:
                X: array (n, d) contendo os dados
                post: array (n, K) contendo as contagens suaves
                        para todos os componentes e todas as amostras

    Returns:
                GaussianMixture: a nova mistura gaussiana
                np.ndarray: array (n, K) contendo as contagens suaves
                        para todos os componentes e todos os exemplos
                float: log-verossimilhança da atribuição atual
    """
    raise NotImplementedError
