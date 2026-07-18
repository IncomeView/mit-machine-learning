from neural_nets.neural_network import NeuralNetwork
import matplotlib.pyplot as plt

nn = NeuralNetwork()

errors = []

for epoch in range(nn.epochs_to_train):
    for (x1, x2), y in nn.training_points:
        nn.train(x1, x2, y)

    # calcular erro total da época
    epoch_error = nn.calculate_total_error()
    errors.append(epoch_error)
    print(f"Epoch {epoch}: erro = {epoch_error}")

# plotar erros
plt.plot(errors)
plt.show()

# testar rede
for point in nn.testing_points:
    print(point, nn.predict(*point))
