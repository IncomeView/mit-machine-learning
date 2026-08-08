"""Mixture model based on kmeans"""

from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """E-step: Atribui cada ponto de dados ao componente gaussiano com a
            média mais próxima

    Args:
            X: array (n, d) contendo os dados
            mixture: a mistura gaussiana atual

    Returns:
            np.ndarray: array (n, K) contendo as contagens suaves
                    para todos os componentes e todos os exemplos
    """
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    for i in range(n):
        tiled_vector = np.tile(X[i, :], (K, 1))
        sse = ((tiled_vector - mixture.mu) ** 2).sum(axis=1)
        j = np.argmin(sse)
        post[i, j] = 1

    return post


def mstep(X: np.ndarray, post: np.ndarray) -> Tuple[GaussianMixture, float]:
    """M-step: Atualiza a mistura de gaussianas. Cada cluster
                gera uma média e uma variância de componente.

    Args: X:
                array (n, d) contendo os dados
                post: array (n, K) contendo as contagens suaves
                        para todos os componentes e todos os exemplos

    Returns:
                GaussianMixture: a nova mistura de gaussianas
                float: o custo de distorção para a atribuição atual
    """
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    cost = 0
    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((mu[j] - X) ** 2).sum(axis=1) @ post[:, j]
        cost += sse
        var[j] = sse / (d * n_hat[j])

    return GaussianMixture(mu, var, p), cost


def run(
    X: np.ndarray, mixture: GaussianMixture, post: np.ndarray
) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Executa o modelo de mistura

    Args:
                X: array (n, d) contendo os dados
                post: array (n, K) contendo as contagens suaves
                        para todos os componentes e todos os exemplos

    Returns:
                GaussianMixture: a nova mistura gaussiana
                np.ndarray: array (n, K) contendo as contagens suaves
                        para todos os componentes e todos os exemplos
                float: custo de distorção da atribuição atual
    """

    prev_cost = None
    cost = None
    while prev_cost is None or prev_cost - cost > 1e-4:
        prev_cost = cost
        post = estep(X, mixture)
        mixture, cost = mstep(X, post)

    return mixture, post, cost
