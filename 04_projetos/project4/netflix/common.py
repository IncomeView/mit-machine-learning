"""Mixture model para filtragem colaborativa"""

from typing import NamedTuple, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arc


class GaussianMixture(NamedTuple):
    """Tupla contendo uma gaussian mixture"""

    mu: (
        np.ndarray
    )  # (K, d) array - cada linha corresponde a uma média de componente gaussiana
    var: np.ndarray  # (K, ) array - cada linha corresponde à variância de um componente
    p: np.ndarray  # (K, ) array = cada linha corresponde ao peso de um componente


def init(X: np.ndarray, K: int, seed: int = 0) -> Tuple[GaussianMixture, np.ndarray]:
    """Inicializa o modelo de mistura com pontos aleatórios como ponto inicial
    atribuições de médias e uniformes

    Args:
        X: (n, d) array que armazena os dados
        K: número de componentes
        seed: semente aleatória

    Returns:
        mixture: a mistura gaussiana inicializada
        post: (n, K) array que armazena as contagens parciais
            para todos os componentes, para todos os exemplos
    """
    np.random.seed(seed)
    n, _ = X.shape
    p = np.ones(K) / K

    # select K random points as initial means
    mu = X[np.random.choice(n, K, replace=False)]
    var = np.zeros(K)
    # Compute variance
    for j in range(K):
        var[j] = ((X - mu[j]) ** 2).mean()

    mixture = GaussianMixture(mu, var, p)
    post = np.ones((n, K)) / K

    return mixture, post


def plot(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray, title: str):
    """Plots the mixture model for 2D data"""
    _, K = post.shape

    percent = post / post.sum(axis=1).reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.title.set_text(title)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    r = 0.25
    color = ["r", "b", "k", "y", "m", "c"]
    for i, point in enumerate(X):
        theta = 0
        for j in range(K):
            offset = percent[i, j] * 360
            arc = Arc(
                xy=point,
                width=r,
                height=r,
                angle=0,
                theta1=theta,
                theta2=theta + offset,
                edgecolor=color[j],
            )
            ax.add_patch(arc)
            theta += offset
    for j in range(K):
        mu = mixture.mu[j]
        sigma = np.sqrt(mixture.var[j])
        circle = Circle(mu, sigma, color=color[j], fill=False)
        ax.add_patch(circle)
        legend = "mu = ({:0.2f}, {:0.2f})\n stdv = {:0.2f}".format(mu[0], mu[1], sigma)
        ax.text(mu[0], mu[1], legend)
    plt.axis('equal')
    plt.show()


def rmse(X, Y):
    return np.sqrt(np.mean((X - Y) ** 2))


def bic(X: np.ndarray, mixture: GaussianMixture, log_likelihood: float) -> float:
    """Calcula o Critério de Informação Bayesiano para um
    mixture of gaussians

    Args:
        X: (n, d) array que armazena os dados
        mixture: uma mistura de gaussianas esféricas
        log_likelihood: a log-verossimilhança dos dados

    Returns:
        float: o BIC para esta mistura
    """
    raise NotImplementedError
