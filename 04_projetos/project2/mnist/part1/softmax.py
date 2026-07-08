import sys

sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adiciona x[i][0] = 1 feature para cada ponto de dados x[i].

    Argumentos:
        X - uma matriz NumPy de n pontos de dados, cada um com d - 1 características
    Retorna:
        X_augment, um array NumPy (n, d) com a característica adicionada para cada ponto de dados
    """
    column_of_ones = np.zeros([len(X), 1]) + 1

    return np.hstack((column_of_ones, X))


def compute_probabilities(X, theta, temp_parameter):
    """
    Calcula, para cada ponto de dados X[i], a probabilidade de X[i] ser rotulado como j
        para j = 0, 1, ..., k-1

    Argumentos:
        X - (n, d) array NumPy de dimensões (n pontos de dados, cada um com d características)
        theta - (k, d) array NumPy de dimensões, onde a linha j representa os parâmetros do nosso modelo para o rótulo j
        temp_parameter - o parâmetro de temperatura da função softmax (escalar)
    Retorna:
        H - (k, n) array NumPy de dimensões, onde cada entrada H[j][i] é a probabilidade de X[i] ser rotulado como j
    """
    Z = (theta @ X.T) / temp_parameter
    c = np.max(Z, axis=0)
    Zc = Z - c
    exp_Zc = np.exp(Zc)

    sum_exp_Zc = np.sum(exp_Zc, axis=0)
    H = exp_Zc / sum_exp_Zc

    return H


def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Calcula o custo total considerando todos os pontos de dados.

        Argumentos:
        X - (n, d) Array NumPy de dimensões (n pontos de dados, cada um com d características)
        Y - (n, ) Array NumPy contendo os rótulos (um número de 0 a 9) para cada
        ponto de dados
        theta - (k, d) Array NumPy de dimensões, onde a linha j representa os parâmetros do nosso
        modelo para o rótulo j
        lambda_factor - a constante de regularização (escalar)
        temp_parameter - o parâmetro de temperatura da função softmax (escalar)
    Retorno:
        c - o valor do custo (escalar)
    """
    H = compute_probabilities(X, theta, temp_parameter)
    soft = H[Y, np.arange(X.shape[0])]
    log_likelihood = -np.log(soft)
    mean_likelihood = np.mean(log_likelihood)

    regularization = (lambda_factor / 2) * np.sum(theta**2)
    cost = mean_likelihood + regularization

    return cost

    raise NotImplementedError


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Executa uma etapa do gradiente descendente em lote (batch gradient descent)

    Argumentos:
        X - (n, d) Array NumPy de dimensões (n pontos de dados, cada um com d atributos)
        Y - (n, ) Array NumPy contendo os rótulos (um número de 0 a 9) para cada
        ponto de dados
        theta - (k, d) Array NumPy de dimensões, onde a linha j representa os parâmetros do nosso
        modelo para o rótulo j
        alpha - a taxa de aprendizado (escalar)
        lambda_factor - a constante de regularização (escalar)
        temp_parameter - o parâmetro de temperatura da função softmax (escalar)
    Retorna:
        theta - Array NumPy de dimensões (k, d) contendo o valor final dos parâmetros theta
    """
    n = X.shape[0]  # Número de pontos de dados
    k = theta.shape[0]  # Número de classes

    # Softmax (k x n) | Probabilidades de cada classe para cada ponto de dados
    H = compute_probabilities(X, theta, temp_parameter)

    # Matriz indicador (k x n) | sparse
    # onde -> M[j, i] = 1 se Y[i] == j
    M = sparse.coo_matrix(([1] * n, (Y, range(n))), shape=(k, n)).toarray()

    # Erro (k x n) | Diferença entre a matriz de probabilidades e a matriz indicadora
    diff = M - H

    # Gradiente da parte de dados
    # diff (k x n) @ X (n x d) = grad_matrix (k x d)
    grad_P = -(1 / (n * temp_parameter)) * (diff @ X)

    # Regularização
    grad_reg = lambda_factor * theta

    # Gradiente total
    gradient = grad_P + grad_reg

    # Atualização de theta
    theta = theta - alpha * gradient

    return theta


def update_y(train_y, test_y):
    """
    Altera os rótulos originais dos dígitos dos conjuntos de treino e teste para os novos rótulos (mod 3).

    Argumentos:
        train_y - (n, ) Array NumPy de formato contendo os rótulos (um número entre 0 e 9)
        para cada ponto de dados no conjunto de treino
        test_y - (n, ) Array NumPy de formato contendo os rótulos (um número entre 0 e 9)
        para cada ponto de dados no conjunto de teste

    Retornos:
        train_y_mod3 - (n, ) Array NumPy de formato, contendo os novos rótulos (um número entre 0 e 2)
        para cada ponto de dados no conjunto de treino
        test_y_mod3 - (n, ) Array NumPy de formato, contendo os novos rótulos (um número entre 0 e 2)
        para cada ponto de dados no conjunto de teste
    """

    return train_y % 3, test_y % 3


def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Retorna o erro associado a esses novos rótulos quando o classificador prevê o dígito (módulo 3).

    Argumentos:
        X - (n, d - 1) Array NumPy de dimensão (n pontos de dados, cada um com d - 1 atributos)
        Y - (n, ) Array NumPy de dimensão (n, ) contendo os rótulos (um número de 0 a 2) para cada
        ponto de dados
        theta - (k, d) Array NumPy de dimensão (k, d), onde a linha j representa os parâmetros do nosso
        modelo para o rótulo j
        temp_parameter - o parâmetro de temperatura da função softmax (escalar)
    Retorno:
        test_error - a taxa de erro do classificador (escalar)
    """
    # Classificação normal (0 - 9)
    lables = get_classification(X, theta, temp_parameter)
    # Converter previsões para mod 3
    lables_mod3 = lables % 3
    # Comparar com os rótulos corretos (já em mod 3)
    test_erro = 1 - np.mean(lables_mod3 == Y)

    return test_erro


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Executa o gradiente descendente em lote (*batch gradient descent*) por um número especificado de iterações em um conjunto de dados,
    com theta inicializado como um array de zeros. Aqui, theta é um array NumPy de dimensões k por d,
    no qual a linha j representa os parâmetros do nosso modelo para o rótulo j,
    para j = 0, 1, ..., k-1.

    Argumentos:
        X - (n, d - 1) array NumPy de dimensões (n pontos de dados, cada um com d-1 atributos)
        Y - (n, ) array NumPy de dimensão, contendo os rótulos (um número de 0 a 9) para cada
        ponto de dados
        temp_parameter - o parâmetro de temperatura da função softmax (escalar)
        alpha - a taxa de aprendizado (escalar)
        lambda_factor - a constante de regularização (escalar)
        k - o número de rótulos (escalar)
        num_iterations - o número de iterações para executar o gradiente descendente (escalar)
    Retornos:
        theta - (k, d) array NumPy de dimensões, contendo o valor final dos parâmetros theta
        cost_function_progression - uma lista Python contendo o custo calculado em cada etapa do gradiente descendente
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(
            compute_cost_function(X, Y, theta, lambda_factor, temp_parameter)
        )
        theta = run_gradient_descent_iteration(
            X, Y, theta, alpha, lambda_factor, temp_parameter
        )
    return theta, cost_function_progression


def get_classification(X, theta, temp_parameter):
    """
    Realiza previsões classificando um determinado conjunto de dados

    Argumentos:
        X - Array NumPy de dimensões (n, d - 1) (n pontos de dados, cada um com d - 1 atributos)
        theta - Array NumPy de dimensões (k, d), onde a linha j representa os parâmetros do nosso modelo para
        o rótulo j
        temp_parameter - o parâmetro de temperatura da função softmax (escalar)
    Retorno:
        Y - Array NumPy de dimensão (n, ), contendo o rótulo previsto (um número entre 0 e 9) para
        cada ponto de dados
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis=0)


def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()


def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.0
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
