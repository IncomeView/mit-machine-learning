import numpy as np

### Functions for you to fill in ###


def closed_form(X, Y, lambda_factor):
    """
    Calcula a solução de forma fechada para regressão linear com regularização L2

    Argumentos:
    X - (n, d + 1) Array NumPy de dimensões (n pontos de dados, cada um com d atributos mais o atributo de viés na primeira dimensão)
    Y - (n, ) Array NumPy contendo os rótulos (um número de 0 a 9) para cada
    ponto de dados
    lambda_factor - a constante de regularização (escalar)
    Retorna:
    theta - (d + 1, ) Array NumPy contendo os pesos da regressão linear. Note que theta[0]
    representa a interceptação no eixo y do modelo e, portanto, X[0] = 1

        Regressão Linear Ridge: Regularização L2 - Solução Fechada
    """
    d = X.shape[1]
    I = np.eye(d)
    #    I[0,0] = 0

    A = X.T @ X + lambda_factor * I
    b = X.T @ Y

    theta = np.linalg.solve(A, b)
    return theta


### Funções já prontas para você usar ###


def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
