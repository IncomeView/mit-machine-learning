import sys
import numpy as np

import matplotlib.pyplot as plt
from sklearn.svm import SVC

sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

RUN_LINEAR = False
RUN_LINEAR_TEST = False
RUN_SVM_ONE = False
RUN_SVM_MULT = False
RUN_SOFTMAX = False
RUN_SOFTMAX_TESTS = False
RUN_SOFTMAX_MOD3 = False
RUN_PCA = False
RUN_PCA_CUBIC = True
RUN_SVM_CUBIC = False
RUN_SVM_RBF_PCA_CUBIC = False


#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
# train_x, train_y, test_x, test_y = get_MNIST_data()
# Plote as primeiras 20 imagens do conjunto de treinamento.
# plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################

# TODO: Primeiro, preencha as funções em linear_regression.py; caso contrário, as funções abaixo não funcionarão.


def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Treina a regressão linear, classifica os dados de teste e calcula o erro no conjunto de teste.

    Returns:
        Erro no teste final
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error


# Não execute isto até que as funções relevantes em linear_regression.py tenham sido totalmente implementadas.
if RUN_LINEAR:
    print(
        'Linear Regression test_error =',
        run_linear_regression_on_MNIST(lambda_factor=1),
    )


def test_lambdas():
    lambdas = [1, 0.1, 0.01]
    results = {}
    for l in lambdas:
        results[l] = run_linear_regression_on_MNIST(lambda_factor=l)
        print('Linear Regression test_error for lambda =', l, 'is', results[l])
    return results


if RUN_LINEAR_TEST:
    test_lambdas()


#######################################################################
# 3. Support Vector Machine
#######################################################################

# TODO: first fill out functions in svm.py, or the functions below will not work


def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


if RUN_SVM_ONE:
    print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


if RUN_SVM_MULT:
    print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())


#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################

# TODO: Primeiro, preencha as funções em softmax.py; caso contrário, o run_softmax_on_MNIST não funcionará.


def run_softmax_on_MNIST(temp_parameter=1):
    """
    Treina o modelo softmax, classifica os dados de teste, calcula o erro de teste e plota a função de custo

    Executa a regressão softmax no conjunto de treinamento MNIST e calcula o erro de teste
    utilizando o conjunto de teste. Utiliza os seguintes valores para os parâmetros:
        alpha = 0,3
        lambda = 1e-4
        num_iterations = 150
    Salva o parâmetro theta final em ./theta.pkl.gz

    Retorna:
        Erro de teste final
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(
        train_x,
        train_y,
        temp_parameter,
        alpha=0.3,
        lambda_factor=1.0e-4,
        k=10,
        num_iterations=150,
    )
    #    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Salve em disco os parâmetros do modelo theta obtidos ao chamar softmax_regression.
    write_pickle_data(theta, "./theta.pkl.gz")

    # TODO: adicione seu código aqui para a questão "Usando o Modelo Atual" na aba 6.
    #      e imprima o test_error_mod3
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)

    # Calcular o erro de teste para mod 3
    test_erro_mod3 = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)

    return test_error, test_erro_mod3


if RUN_SOFTMAX:
    print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))


# TODO: Encontre a taxa de erro para temp_parameter = [.5, 1.0, 2.0]
#      Lembre-se de retornar tempParameter para 1 e executar novamente run_softmax_on_MNIST


if RUN_SOFTMAX_TESTS:
    print("\nTestando diferentes valores de temp_parameter")

    # 0.5, 1.0, 2.0

    for T in [0.5, 0.3, 0.1]:
        print(f"\nExecutando softmax com T={T}")
        erro_T = run_softmax_on_MNIST(temp_parameter=T)
        print(f"Erro para T={T}: {erro_T}")

    print("\nRetornando temp_parameter para 1")
    final_erro = run_softmax_on_MNIST(temp_parameter=1)
    print(f"Erro final com T=1: {final_erro}")


#######################################################################
# 6. Changing Labels
#######################################################################


def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Treina a regressão Softmax em classificações de dígitos (módulo 3).
    Consulte run_softmax_on_MNIST para mais informações.
    """
    # Carregar os dados MNIST
    train_x, train_y, test_x, test_y = get_MNIST_data()

    # Alterar os rótulos para mod 3
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)

    # Treinar o modelo normal (0 - 9)
    theta, cost_function_history = softmax_regression(
        train_x,
        train_y_mod3,
        temp_parameter,
        alpha=0.3,
        lambda_factor=1.0e-4,
        k=3,
        num_iterations=150,
    )

    # Calcular o erro de teste para mod 3
    test_erro_mod3 = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)

    return test_erro_mod3


if RUN_SOFTMAX_MOD3:
    print('softmax mod3 test_error=', run_softmax_on_MNIST_mod3(temp_parameter=1))


# # TODO: Execute run_softmax_on_MNIST_mod3() e informe a taxa de erro.


#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Redução de dimensionalidade via PCA ##

# TODO: Primeiro preencha as funções PCA em features.py, pois o código abaixo depende delas.

if RUN_PCA:
    train_x, train_y, test_x, test_y = get_MNIST_data()

    print("Rodando PCA com 18 componentes ...")
    n_components = 18

    ###Nota de correção: as 4 linhas a seguir foram modificadas desde o lançamento.

    # Centralizar dados de treino
    train_x_centered, feature_means = center_data(train_x)

    # Calcular componentes principais
    pcs = principal_components(train_x_centered)

    # Projetar treino e teste nos PCs
    train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

    # train_pca (e test_pca) é uma representação dos nossos dados de treinamento (e de teste)
    # após projetar cada exemplo nos 18 primeiros componentes principais.

    # TODO: Treine seu modelo de regressão softmax usando (train_pca, train_y)
    #       e avalie sua acurácia em (test_pca, test_y).

    # Treinar softmax usando train_pca
    theta, _ = softmax_regression(
        train_pca,
        train_y,
        temp_parameter=1,
        alpha=0.3,
        lambda_factor=1.0e-4,
        k=10,
        num_iterations=150,
    )

    # Calcular erro no teste usando test_pca
    test_error = compute_test_error(
        test_pca,
        test_y,
        theta,
        temp_parameter=1,
    )

    print("Taxa de erro para características PCA de 18 dimensões =", test_error)

    # TODO: Use a função plot_PC em features.py para gerar um gráfico de dispersão
    #       das primeiras 100 imagens do MNIST, representadas no espaço definido pelos
    #       dois primeiros componentes principais encontrados acima.

    plot_PC(
        train_x[0:100],
        pcs,
        train_y[0:100],
        feature_means,
        #        train_x[range(000, 100),], pcs, train_y[range(000, 100)], feature_means
    )  # médias das características adicionadas desde o lançamento

    # TODO: Utilize a função reconstruct_PC em features.py para exibir
    #       a primeira e a segunda imagens do MNIST reconstruídas exclusivamente a partir
    #       de sua representação de componentes principais de 18 dimensões.
    #       Compare as imagens reconstruídas com as originais.

    firstimage_reconstructed = reconstruct_PC(
        train_pca[0,], pcs, n_components, train_x, feature_means
    )  # médias das características adicionadas desde o lançamento
    plot_images(firstimage_reconstructed)
    plot_images(train_x[0,])

    secondimage_reconstructed = reconstruct_PC(
        train_pca[1,], pcs, n_components, train_x, feature_means
    )  # médias das características adicionadas desde o lançamento
    plot_images(secondimage_reconstructed)
    plot_images(train_x[1,])


## Cubic Kernel ##
# TODO: Obtenha a representação PCA de 10 dimensões dos conjuntos de treinamento e teste.

if RUN_PCA_CUBIC:
    train_x, train_y, test_x, test_y = get_MNIST_data()

    print("Rodando PCA com 10 componentes ...")
    n_components = 10

    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_pca10 = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_pca10 = project_onto_PC(test_x, pcs, n_components, feature_means)

    # TODO: Primeiro, preencha a função `cubicFeatures()` no arquivo `features.py`, conforme exigido pelo código abaixo.

    train_cube = cubic_features(train_pca10)
    test_cube = cubic_features(test_pca10)

    # train_cube (e test_cube) é uma representação dos nossos dados de treinamento (e de teste)
    # após a aplicação do mapeamento de características com kernel cúbico às representações PCA de 10 dimensões.

    # TODO: Treine seu modelo de regressão softmax usando (train_cube, train_y)
    #       e avalie sua acurácia em (test_cube, test_y).

    # Treinar softmax usando train_pca_cube
    theta, _ = softmax_regression(
        train_cube,
        train_y,
        temp_parameter=1,
        alpha=0.3,
        lambda_factor=1.0e-4,
        k=10,
        num_iterations=150,
    )

    # Calcular erro no teste usando test_pca_cube
    test_error = compute_test_error(
        test_cube,
        test_y,
        theta,
        temp_parameter=1,
    )

    print("Taxa de erro cubic features de 10 dimensões =", test_error)


if RUN_SVM_CUBIC:
    train_x, train_y, test_x, test_y = get_MNIST_data()

    print("Rodando PCA | SVM com 10 componentes ...")
    n_components = 10

    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_pca10 = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_pca10 = project_onto_PC(test_x, pcs, n_components, feature_means)

    # Treinar SVM usando train_pca

    svm_model = SVC(
        kernel='poly',
        degree=3,
        random_state=0,
        #        gamma=0.05  -  Sugestão copilot para chegar a 0.0686
    )
    svm_model.fit(train_pca10, train_y)

    test_accuracy = svm_model.score(test_pca10, test_y)
    test_error = 1 - test_accuracy

    print("Taxa de erro PCA de 10 dimensões usando SVM polinomial cúbico =", test_error)


if RUN_SVM_RBF_PCA_CUBIC:
    train_x, train_y, test_x, test_y = get_MNIST_data()

    print("Rodando PCA | SVM com 10 componentes ...")
    n_components = 10

    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_pca10 = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_pca10 = project_onto_PC(test_x, pcs, n_components, feature_means)

    # Treinar SVM RBF usando train_pca

    svm_model = SVC(kernel='rbf', random_state=0)
    svm_model.fit(train_pca10, train_y)

    test_accuracy = svm_model.score(test_pca10, test_y)
    test_error = 1 - test_accuracy

    print("Taxa de erro PCA de 10 dimensões usando SVM polinomial cúbico =", test_error)
