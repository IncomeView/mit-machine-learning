import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Gera uma rede neural com a seguinte arquitetura:
    Rede neural totalmente conectada. 
    O vetor de entrada recebe duas características. 
    Uma camada oculta com três neurônios cuja função de ativação é a ReLU. 
    Um neurônio de saída cuja função de ativação é a função identidade.
"""


def rectified_linear_unit(x):
    """Retorna a ReLU de x, ou o máximo entre 0 e x."""
    return max(0, x)


def rectified_linear_unit_derivative(x):
    """Retorna a derivada da ReLU."""
    return 1 if x > 0 else 0


def output_layer_activation(x):
    """Função linear; retorna a entrada como está.
    a = f(u) = u
    """
    return x


def output_layer_activation_derivative(x):
    """Retorna a derivada de uma função linear: 1."""
    return 1


class NeuralNetwork:
    """
    Contém as seguintes funções:
            -train: ajusta os parâmetros da rede neural com base no erro obtido na propagação direta (forward propagation).
            -predict: prevê o rótulo de um vetor de características com base nos parâmetros da classe.
            -train_neural_network: treina uma rede neural utilizando todos os pontos de dados pelo número de épocas especificado durante a inicialização da classe.
            -test_neural_network: utiliza os parâmetros especificados no momento para verificar se a rede neural classifica os pontos fornecidos em testing_points dentro de uma margem de erro.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS (Initialized to floats instead of ints)
        # matriz W de tamanho 3x2, que representa os pesos da camada de entrada para a camada oculta
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.')
        # matriz V de tamanho 1x3, que representa os pesos da camada oculta para a camada de saída
        self.hidden_to_output_weights = np.matrix('1. 1. 1.')
        self.biases = np.matrix('0.; 0.; 0.')
        self.learning_rate = 0.001
        self.epochs_to_train = 10
        self.training_points = [((2, 1), 10), ((3, 3), 21), ((4, 5), 32), ((6, 6), 42)]
        self.testing_points = [(1, 1), (2, 2), (3, 3), (5, 5), (10, 10)]

    def train(self, x1, x2, y):
        ### Forward propagation ###
        # O que entra na camada oculta | x
        input_values = np.matrix([[x1], [x2]])  # 2x1

        # Calcule a entrada e a ativação da camada oculta.
        # z = Wx + b | pré-ativação, é o produto ponderado da entrada e dos pesos + o viés.
        hidden_layer_weighted_input = (
            self.input_to_hidden_weights * input_values + self.biases
        )  # 3x1
        # é o h = ReLU(z) = max(0, z)
        relu = np.vectorize(rectified_linear_unit)
        hidden_layer_activation = relu(hidden_layer_weighted_input)  # 3x1

        # output é u | u = V * h
        output = self.hidden_to_output_weights * hidden_layer_activation  # 1x1
        # a = f(u) = u | É a ativação da camada de saída, que neste caso é a função identidade
        activated_output = output_layer_activation(output)

        ### Backpropagation ###
        # Calcular gradientes
        # 𝛿_out = a - y | É o erro da camada de saída
        output_layer_error = (
            (y - activated_output) * -1 * output_layer_activation_derivative(output)
        )

        # Erro da camada oculta
        # ReLU'(z)
        relu_partial = np.vectorize(rectified_linear_unit_derivative)
        # 𝛿_hidden = (V^T 𝛿_out) ⊙ ReLU′(𝑧)
        hidden_layer_error = np.multiply(
            np.transpose(self.hidden_to_output_weights) * output_layer_error,
            relu_partial(hidden_layer_weighted_input),
        )  # 3x1

        # Gradiente do custo em relação aos vises b
        bias_gradients = hidden_layer_error
        # Gradiente de Custo em relação ao Vetor de ativação V | 1x3
        hidden_to_output_weight_gradients = output_layer_error * np.transpose(
            hidden_layer_activation
        )
        # Gradiente de custo em relação ao peso de entrada W | 1x3
        input_to_hidden_weight_gradients = hidden_layer_error * np.transpose(
            input_values
        )

        ### Gradient Descent Update ###
        # Use gradientes para ajustar pesos e vieses utilizando o gradiente descendente.
        # Learning_rate = 𝜂 (eta)
        self.biases = self.biases - self.learning_rate * bias_gradients
        self.input_to_hidden_weights = (
            self.input_to_hidden_weights
            - self.learning_rate * input_to_hidden_weight_gradients
        )
        self.hidden_to_output_weights = (
            self.hidden_to_output_weights
            - self.learning_rate * hidden_to_output_weight_gradients
        )

    def predict(self, x1, x2):
        input_values = np.matrix([[x1], [x2]])  # 2x1

        # Calcule a saída para uma única entrada (deve ser igual à forward propagation durante o treinamento).
        # z = Wx + b | Valores Intermediarios 3x1
        hidden_layer_weighted_input = (
            self.input_to_hidden_weights * input_values + self.biases
        )  # 3x1
        # h = ReLU(z) = max(0, z)
        relu = np.vectorize(rectified_linear_unit)
        hidden_layer_activation = relu(hidden_layer_weighted_input)  # 3x1

        # output é u | u = V * h
        output = self.hidden_to_output_weights * hidden_layer_activation  # 1x1
        # a = f(u) = u | É a ativação da camada de saída, que neste caso é a função identidade
        activated_output = output_layer_activation(output)  # identity

        return activated_output.item()

        # Run this to train your neural network once you complete the train method

    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x, y in self.training_points:
                self.train(x[0], x[1], y)

                # Run this to test your neural network implementation for correctness after it is trained

    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7 * point[0]) < 0.1:
                print("Test Passed")
            else:
                print(
                    "Point ", point[0], point[1], " failed to be predicted correctly."
                )
        return


x = NeuralNetwork()
x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
# x.test_neural_network()
