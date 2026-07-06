import sys
import numpy as np

import matplotlib.pyplot as plt

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
RUN_SOFTMAX = True
RUN_TESTS = False
RUN_PLOTS = False


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
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Salve em disco os parâmetros do modelo theta obtidos ao chamar softmax_regression.
    write_pickle_data(theta, "./theta.pkl.gz")

    # TODO: adicione seu código aqui para a questão "Usando o Modelo Atual" na aba 6.
    #      e imprima o test_error_mod3
    return test_error


if RUN_SOFTMAX:
    print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))


# TODO: Encontre a taxa de erro para temp_parameter = [.5, 1.0, 2.0]
#      Lembre-se de retornar tempParameter para 1 e executar novamente run_softmax_on_MNIST


def test_softmax(alpha_list, temp_list, lambda_factor=1e-4, num_iterations=150):
    """
    Testa várias combinações de hiperparâmetros para o modelo softmax no MNIST.

    alpha_list: lista de taxas de aprendizado
    temp_list: lista de temperaturas
    lambda_factor: regularização
    num_iterations: número de iterações
    """

    train_x, train_y, test_x, test_y = get_MNIST_data()

    results = []

    for alpha in alpha_list:
        for temp_parameter in temp_list:
            print("\n==============================================")
            print(f"Testando alpha={alpha}, temp={temp_parameter}")
            print("==============================================")

            theta, cost_history = softmax_regression(
                train_x,
                train_y,
                temp_parameter,
                alpha=alpha,
                lambda_factor=lambda_factor,
                k=10,
                num_iterations=num_iterations,
            )

            test_error = compute_test_error(test_x, test_y, theta, temp_parameter)

            print(f"Erro de teste: {test_error:.4f}")

            results.append(
                {
                    "alpha": alpha,
                    "temp": temp_parameter,
                    "test_error": test_error,
                    "cost_history": cost_history,
                }
            )

    return results


if RUN_TESTS:
    alpha_list = [0.1, 0.3, 0.5]
    temp_list = [0.5, 1.0, 2.0]
    test_results = test_softmax(alpha_list, temp_list)


def plot_all_costs(results):
    plt.figure(figsize=(12, 8))

    for r in results:
        plt.figure()
        plt.plot(r["cost_history"])
        plt.title(f"a={r['alpha']}, T={r['temp']}, err={r['test_error']:.3f}")
        plt.show()

    # for i, (alpha, temp, err, cost) in enumerate(results):
    #     plt.subplot(3, 4, i+1)
    #     plt.plot(cost)
    #     plt.title(f"a={alpha}, T={temp}\nerr={err:.3f}")
    #     plt.xlabel("Iteração")
    #     plt.ylabel("Custo")

    plt.tight_layout()
    plt.show()


if RUN_PLOTS:
    plot_all_costs(test_results)


# #######################################################################
# # 6. Changing Labels
# #######################################################################


# def run_softmax_on_MNIST_mod3(temp_parameter=1):
#     """
#     Trains Softmax regression on digit (mod 3) classifications.

#     See run_softmax_on_MNIST for more info.
#     """
#     # YOUR CODE HERE
#     raise NotImplementedError


# # TODO: Run run_softmax_on_MNIST_mod3(), report the error rate


# #######################################################################
# # 7. Classification Using Manually Crafted Features
# #######################################################################

# ## Dimensionality reduction via PCA ##

# # TODO: First fill out the PCA functions in features.py as the below code depends on them.


# n_components = 18

# ###Correction note:  the following 4 lines have been modified since release.
# train_x_centered, feature_means = center_data(train_x)
# pcs = principal_components(train_x_centered)
# train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
# test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

# # train_pca (and test_pca) is a representation of our training (and test) data
# # after projecting each example onto the first 18 principal components.


# # TODO: Train your softmax regression model using (train_pca, train_y)
# #       and evaluate its accuracy on (test_pca, test_y).


# # TODO: Use the plot_PC function in features.py to produce scatterplot
# #       of the first 100 MNIST images, as represented in the space spanned by the
# #       first 2 principal components found above.
# plot_PC(
#     train_x[range(000, 100),], pcs, train_y[range(000, 100)], feature_means
# )  # feature_means added since release


# # TODO: Use the reconstruct_PC function in features.py to show
# #       the first and second MNIST images as reconstructed solely from
# #       their 18-dimensional principal component representation.
# #       Compare the reconstructed images with the originals.
# firstimage_reconstructed = reconstruct_PC(
#     train_pca[0,], pcs, n_components, train_x, feature_means
# )  # feature_means added since release
# plot_images(firstimage_reconstructed)
# plot_images(train_x[0,])

# secondimage_reconstructed = reconstruct_PC(
#     train_pca[1,], pcs, n_components, train_x, feature_means
# )  # feature_means added since release
# plot_images(secondimage_reconstructed)
# plot_images(train_x[1,])


# ## Cubic Kernel ##
# # TODO: Find the 10-dimensional PCA representation of the training and test set


# # TODO: First fill out cubicFeatures() function in features.py as the below code requires it.

# train_cube = cubic_features(train_pca10)
# test_cube = cubic_features(test_pca10)
# # train_cube (and test_cube) is a representation of our training (and test) data
# # after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.


# # TODO: Train your softmax regression model using (train_cube, train_y)
# #       and evaluate its accuracy on (test_cube, test_y).
