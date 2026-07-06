import numpy as np
from sklearn.svm import LinearSVC


### Functions for you to fill in ###


def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Treina uma SVM linear para classificação binária

    Argumentos:
        train_x - features (imagens achatadas) | (n, d) Array NumPy de formato (n pontos de dados, cada um com d características)
        train_y - (n,) Array NumPy de formato, contendo os rótulos (0 ou 1) para cada ponto de dados de treinamento, já preparados em run_svm_one_vs_rest_on_MNIST
        test_x - (m, d) Array NumPy de formato (m pontos de dados, cada um com d características)
    Retorno:
        pred_test_y - (m,) Array NumPy de formato, contendo os rótulos (0 ou 1) para cada ponto de dados de teste
    """
    svc = LinearSVC(C=0.1, random_state=0)
    svc.fit(train_x, train_y)
    pred_test_y = svc.predict(test_x)

    return pred_test_y


def multi_class_svm(train_x, train_y, test_x):
    """
    Treina uma SVM linear para classificação multiclasse utilizando a estratégia *one-vs-rest*

    Argumentos:
        train_x - Array NumPy de dimensões (n, d) (n pontos de dados, cada um com d características)
        train_y - Array NumPy de dimensão (n,) contendo os rótulos (int) para cada ponto de dados de treinamento
        test_x - Array NumPy de dimensões (m, d) (m pontos de dados, cada um com d características)
    Retorno:
        pred_test_y - Array NumPy de dimensão (m,) contendo os rótulos (int) para cada ponto de dados de teste
    """
    svc = LinearSVC(C=0.1, random_state=0)
    svc.fit(train_x, train_y)
    pred_test_y = svc.predict(test_x)

    return pred_test_y


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)
