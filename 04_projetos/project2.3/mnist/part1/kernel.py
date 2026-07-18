import numpy as np

### Functions for you to fill in ###


def polynomial_kernel(X, Y, c, p):
    """
    Calcula o kernel polinomial entre duas matrizes X e Y:
    K(x, y) = (<x, y> + c)^p
    para cada par de linhas x em X e y em Y.

    Argumentos:
        X - array NumPy de dimensões (n, d) (n pontos de dados, cada um com d características)
        Y - array NumPy de dimensões (m, d) (m pontos de dados, cada um com d características)
        c - coeficiente para equilibrar termos de ordem superior e inferior (escalar)
        p - grau do kernel polinomial
    Retorno:
        kernel_matrix - array NumPy de dimensões (n, m) contendo a matriz do kernel
    """
    # Produto interno entre todas as linhas de X e Y
    dot_product = np.matmul(X, Y.T)  # shape (n, m)

    # Aplicar o kernel polinomial
    kernel_matrix = (dot_product + c) ** p

    return kernel_matrix


def rbf_kernel(X, Y, gamma):
    """
    Calcula o kernel RBF gaussiano entre duas matrizes X e Y::
    K(x, y) = exp(-gamma ||x-y||^2)
    para cada par de linhas x em X e y em Y.

    Argumentos:
        X - array NumPy (n, d) (n pontos de dados, cada um com d características)
        Y - array NumPy (m, d) (m pontos de dados, cada um com d características)
        gamma - o parâmetro gamma da função gaussiana (escalar)
    Retorno:
        kernel_matrix - array NumPy (n, m) contendo a matriz do kernel

    Compute the Gaussian RBF kernel between two matrices X and Y::
        K(x, y) = exp(-gamma ||x-y||^2)
    for each pair of rows x in X and y in Y.
    """

    # Expande X e Y para calcular ||x - y||^2 de forma vetorizada
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)  # shape (n, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)  # shape (1, m)

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    sq_dist = X_norm + Y_norm - 2 * np.matmul(X, Y.T)  # shape (n, m)

    kernel_matrix = np.exp(-gamma * sq_dist)

    return kernel_matrix

    raise NotImplementedError
