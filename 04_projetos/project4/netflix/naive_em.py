"""Mixture model using EM"""

from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Atribui de forma suave cada ponto de dados a um componente gaussiano

    Args:
                X: array (n, d) contendo os dados
                mixture: a mistura gaussiana atual

    Returns:
                np.ndarray: array (n, K) contendo as contagens suaves
                        para todos os componentes e todos os exemplos
                float: log-verossimilhança da atribuição
    """
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Atualiza a mistura gaussiana maximizando a log-verossimilhança
            do conjunto de dados ponderado

    Args:
                X: array (n, d) contendo os dados
                post: array (n, K) contendo as contagens suaves
                        para todos os componentes e todas as amostras

    Returns:
                GaussianMixture: a nova mistura gaussiana
    """
    raise NotImplementedError


def run(
    X: np.ndarray, mixture: GaussianMixture, post: np.ndarray
) -> Tuple[GaussianMixture, np.ndarray, float]:
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
