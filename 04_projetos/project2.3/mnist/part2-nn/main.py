import numpy as np
import matplotlib.pyplot as plt
from neural_nets import NeuralNetwork  # importa sua classe


def calculate_total_error(nn):
    """Calcula o erro total da rede para todos os pontos de treinamento."""
    total_error = 0
    for (x1, x2), y in nn.training_points:
        prediction = nn.predict(x1, x2)
        total_error += 0.5 * (y - prediction) ** 2
    return total_error


def calculate_accuracy(nn, dataset):
    accuracies = []
    for (x1, x2), y in dataset:
        pred = nn.predict(x1, x2)
        acc = 1 - abs(y - pred) / y
        accuracies.append(acc)
    return np.mean(accuracies)


def main():
    nn = NeuralNetwork()

    epoch_errors = []
    train_accuracies = []
    test_accuracies = []

    print("\n=== TREINANDO REDE NEURAL ===\n")

    for epoch in range(nn.epochs_to_train):
        for (x1, x2), y in nn.training_points:
            nn.train(x1, x2, y)

        total_error = calculate_total_error(nn)
        epoch_errors.append(total_error)

        print(f"Epoch {epoch+1}/{nn.epochs_to_train} | Erro total: {total_error:.4f}")

    # Teste acuracia
    # train_acc = calculate_accuracy(nn, nn.training_points)
    # test_acc = calculate_accuracy(nn, nn.testing_points)
    # train_accuracies.append(train_acc)
    # test_accuracies.append(test_acc)
    # print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f} | Test Acc={test_acc:.4f}")

    # plt.plot(train_accuracies, label="Train")
    # plt.plot(test_accuracies, label="Test")
    # plt.title("Acurácia por época")
    # plt.xlabel("Época")
    # plt.ylabel("Acurácia")
    # plt.legend()
    # plt.grid(True)
    plt.show()

    print("\n=== TESTANDO REDE ===\n")
    nn.test_neural_network()

    print("\n=== GERANDO GRÁFICO DE ERRO ===\n")

    plt.plot(epoch_errors, marker='o')
    plt.title("Erro total por época")
    plt.xlabel("Época")
    plt.ylabel("Erro")
    plt.grid(True)
    plt.show()

    print("\n=== COMPARAÇÃO REAL VS PREVISTO ===\n")
    for (x1, x2), y in nn.training_points:
        pred = nn.predict(x1, x2)
        print(f"Entrada ({x1}, {x2}) | Real: {y} | Previsto: {pred:.4f}")


if __name__ == "__main__":
    main()

# self.testing_points = [((1, 1), 7), ((2, 2), 14), ((3, 3), 21), ((5, 5), 35), ((10, 10), 70)]	# teste acuracia
