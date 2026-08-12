"""Mixture model for matrix completion"""

from typing import Tuple
import numpy as np
import common
import naive_em
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    n, d = X.shape  # n - numero de usuarios, d - numero de filmes
    K = mixture.p.shape[0]  # K - numero de clusters da mistura Gaussiana
    post = np.zeros(
        (n, K)
    )  # p(j | u), a responsabilidade do cluster j sobre o usuario u
    ll = 0.0  # log-likelihood total do modelo

    for u in range(n):  # Loop sobre cada usuario
        mask = (
            X[u] != 0
        )  # um vetor booleano indicando quais filmes o usuário avaliou | Mascara das coordenadas observadas (não zero)
        x_obs = X[u, mask]  # contém apenas as avaliações observadas
        Cu = x_obs.shape[0]
        log_probs_u = np.zeros(
            K
        )  # aqui vamos guardar: log p(j) + log p(x_u | j) para cada cluster j

        for j in range(
            K
        ):  # Para cada cluster, calculamos a probabilidade de o usuário pertencer a ele.
            mu_obs = mixture.mu[
                j, mask
            ]  # média do cluster j somente nos filmes avaliados pelo usuário
            var_obs = mixture.var[
                j
            ]  # var[j] é escalar → expandimos para o tamanho de mu_obs
            log_gauss = -0.5 * (  # log da gaussiana somente nas coordenadas observadas
                Cu
                * np.log(
                    2 * np.pi * var_obs
                )  # das gaussianas unidimensionais: somados apenas sobre coordenadas observadas
                + np.sum((x_obs - mu_obs) ** 2) / var_obs
            )  # log N(x_obs | mu_obs, var_obs) = -0.5 * sum(log(2*pi*var_obs) + (x_obs - mu_obs)^2 / var_obs)

            log_probs_u[j] = (
                np.log(mixture.p[j]) + log_gauss
            )  # log da probabilidade conjunta: log(j) + log p(x_u | j)

        log_norm = logsumexp(
            log_probs_u
        )  # Normalização com LogSumExp - normalização: p(j | u) = exp(log p(j) + log p(j | x_u)) / sum_j' exp(log p(j' | x_u))
        post[u] = np.exp(
            log_probs_u - log_norm
        )  # Responsabilidades normalizadas | Evita underflow ao normalizar
        ll += (
            log_norm  # likelihood total do modelo: \sum_u log(sum_j p(j) * p(x_u | j))
        )

    return post, ll
    """E-step: Atribui de forma suave cada ponto de dados a um componente gaussiano

    Args:
                X: array (n, d) contendo os dados, com entradas incompletas (definidas como 0)
                mixture: a mistura gaussiana atual

    Returns:
                np.ndarray: array (n, K) contendo as contagens suaves
                        para todos os componentes e todos os exemplos
        float: log-likelihood da atribuição
    """


def mstep(
    X: np.ndarray,
    post: np.ndarray,
    mixture: GaussianMixture,
    min_variance: float = 0.25,
) -> GaussianMixture:
    n, d = X.shape
    K = post.shape[1]

    p = (
        post.sum(axis=0) / n
    )  # Cada cluster recebe peso proporcional ao total de responsabilidades | p_j = (1/n) * sum_{u=1}^n post(u | j) | isso garante \sum_j p_j = 1
    mu = (
        mixture.mu.copy()
    )  # Manter as médias anteriores para coordenadas não observadas
    var = (
        mixture.var.copy()
    )  # Manter as variâncias anteriores para coordenadas não observadas
    ''' Por quê copiar?
            - porque nem todas as coordenadas serão atualizadas
            - coordenadas sem suporte (sem usuários que avaliaram aquele filme) devem manter o valor anterior
            - isso evita médias erráticas e variâncias colapsadas '''
    for j in range(K):  # Loop sobre clusters e filmes
        for i in range(
            d
        ):  # atualizar μ[j, i] e σ²[j, i] para cada cluster e cada filme.
            mask = X[:, i] != 0  # zeros não ignorados
            weight_sum = post[
                mask, j
            ].sum()  # Soma dos pesos dos usuários que têm esse filme observado, weight_sum < 0, não atualizamos a média | \sum_{u: X[u, i] != 0} post(u | j)
            if (
                weight_sum >= 1
            ):  # \mu_{j, i} = \sum_{u: X[u, i] != 0} post(u | j) * X[u, i] / \sum_{u: X[u, i] != 0} post(u | j)
                mu[j, i] = (
                    np.sum(post[mask, j] * X[mask, i]) / weight_sum
                )  # Atualização da média | É uma média ponderada

        num = 0.0
        den = 0.0
        for i in range(d):
            mask = X[:, i] != 0
            diff = X[mask, i] - mu[j, i]
            num += np.sum(post[mask, j] * diff * diff)
            den += post[mask, j].sum()

        var_j = num / den if den > 0 else 0
        var[j] = max(var_j, min_variance)
        ''' Por quê aplicar variância mínima?
            - clusters com poucos pontos podem colapsar
            - variância muito pequena causa explosão numérica no E-step
            - o MIT recomenda min_variance = 0.25
        Isso garante estabilidade.'''
        # Caso contrário: mantém μ e σ² anteriores
    return GaussianMixture(mu, var, p)
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


def run(
    X: np.ndarray, mixture: GaussianMixture, post: np.ndarray
) -> Tuple[GaussianMixture, np.ndarray, float]:
    '''O run() recebe:
    - X: matriz de avaliações (com zeros indicando ausentes)
    - mixture: parâmetros iniciais (μ, σ², p)
    - post: responsabilidades iniciais (p(j|u))'''
    prev_ll = None  # log-likelihood da interação anterior
    ll = None  # log-likelihood atual

    while True:  # Loop até convergência
        post, ll = estep(X, mixture)
        ''' O E-step:
                - calcula responsabilidades p(j|u)
                - calcula log-verossimilhança atual
            Esse é o passo que usa:
                - log-domínio
                - logsumexp
                - tratamento de zeros como ausentes'''

        if prev_ll is not None and ll - prev_ll <= 1e-6 * abs(ll):
            break
        ''' Isso implementa: parar quando Δ LL ≤ 10^(-6) ⋅ |LL|
            Ou seja:
                - se o ganho de log-verossimilhança for muito pequeno
                - o EM convergiu
                - paramos o loop
            Esse critério é exatamente o usado pelo MIT.'''
        mixture = mstep(X, post, mixture)
        ''' O M-step:
                - atualiza p
                - atualiza μ (somente com suporte suficiente)
                - atualiza σ² (com variância mínima)
            Esse passo usa:
                - médias ponderadas
                - variâncias ponderadas
                - tratamento de zeros como ausentes'''
        prev_ll = ll  # Atualizar likelihhood anterior para a próxima iteração
    return mixture, post, ll
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
    '''Essa função recebe:
        - X → matriz incompleta (n usuários x d filmes), com zeros representando avaliações faltantes
        - mixture → o modelo EM já treinado, contendo:
            - mu (K x d)
            - var (K)
            - p (K)
    E retorna:
        - uma matriz preenchida, onde cada zero foi substituído por uma estimativa baseada no modelo.
    '''
    n, d = X.shape  # n - numero de usuarios | d - numero de filmes
    K = mixture.p.shape[0]  # numero de clusters da mistura gaussiana

    filled = X.copy()  # matriz original
    post, _ = estep(
        X, mixture
    )  # posteriores | calcular responsabilidades | → probabilidade do usuário u pertencer ao cluster j

    for u in range(n):
        for i in range(d):
            if filled[u, i] == 0:  # 0 significa "não observado"
                filled[u, i] = np.dot(
                    post[u], mixture.mu[:, i]
                )  # Preenchimento usando a média ponderada dos clusters
                ''' Essa linha é a alma do fill_matrix.
                    Vamos destrinchar:
                        ✔️ post[u] é um vetor de tamanho K
                            → [p(j=1|u), p(j=2|u), ..., p(j=K|u)]
                        ✔️ mixture.mu[:, i] é um vetor de tamanho K
                            → [μ₁ᵢ, μ₂ᵢ, ..., μ_Kᵢ]
                        ✔️ np.dot(post[u], mixture.mu[:, i]) calcula:
                            → \sum_{j = 1}^k p(j | u) \mu_{j, i}
                    Que é exatamente a equação do MIT: \hat{X}_{u, i} = \sum{j = 1}^k p(j | u) \mu_{j, i}
                        Ou seja:
                            - você preenche o filme i do usuário u
                            - com a média ponderada das médias dos clusters
                            - ponderada pelas responsabilidades do usuário
                        Isso é 100% MIT.'''
    return filled
    """Preenche uma matriz incompleta de acordo com um modelo de mistura


    Args:
                X: array (n, d) de dados incompletos (entradas incompletas = 0)
                        mixture: uma mistura de gaussianas

    Returns
                np.ndarray: um array (n, d) com os dados preenchidos
    """
