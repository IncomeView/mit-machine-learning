"""Mixture model for matrix completion"""

from typing import Tuple
import numpy as np
import common
import naive_em
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Atribui de forma suave cada ponto de dados a um componente gaussiano

    Args:
                X: array (n, d) contendo os dados, com entradas incompletas (definidas como 0)
                mixture: a mistura gaussiana atual

    Returns:
                np.ndarray: array (n, K) contendo as contagens suaves
                        para todos os componentes e todos os exemplos
        float: log-likelihood da atribuição
    """
    raise NotImplementedError


def mstep(
    X: np.ndarray,
    post: np.ndarray,
    mixture: GaussianMixture,
    min_variance: float = 0.25,
) -> GaussianMixture:
    """M-step: Atualiza a mistura de gaussianas maximizando a log-verossimilhança
                                do conjunto de dados ponderado

    Args:
                X: array (n, d) contendo os dados, com entradas incompletas (definidas como 0)
                post: array (n, K) contendo as contagens suaves
                        para todos os componentes e todos os exemplos
                mixture: a mistura de gaussianas atual
                min_variance: a variância mínima para cada gaussiana

    Returns:
        GaussianMixture: a nova gaussian mixture
    """
    raise NotImplementedError


def run(
    X: np.ndarray, mixture: GaussianMixture, post: np.ndarray
) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
                X: array (n, d) contendo os dados
                post: array (n, K) contendo as contagens suaves
                        para todos os componentes e todos os exemplos

    Returns:
                GaussianMixture: a nova mistura gaussiana
                np.ndarray: array (n, K) contendo as contagens suaves
                        para todos os componentes e todos os exemplos
                float: log-verossimilhança da atribuição atual


    """
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Preenche uma matriz incompleta de acordo com um modelo de mistura


    Args:
                X: array (n, d) de dados incompletos (entradas incompletas = 0)
                        mixture: uma mistura de gaussianas

    Returns
                np.ndarray: um array (n, d) com os dados preenchidos
    """
    raise NotImplementedError
