import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import project1 as p1

if sys.version_info[0] < 3:
    PYTHON3 = False
else:
    PYTHON3 = True


def load_toy_data(path_toy_data):
    """
    Carrega o conjunto de dados de brinquedo 2D como arrays numpy.
    Retorna a tupla (características, rótulos) em que características é uma matriz numpy Nx2 e
    rótulos é um vetor de comprimento N com rótulos +1/-1.
    """
    labels, xs, ys = np.loadtxt(path_toy_data, delimiter='\t', unpack=True)
    return np.vstack((xs, ys)).T, labels


def load_data(path_data, extras=False):
    """
    Retorna uma lista de dicionários com as seguintes chaves:
        * sentiment: +1 ou -1 se a avaliação foi positiva ou negativa, respectivamente
        * text: o texto da avaliação

    Além disso, se o argumento `extras` for True, cada dicionário também incluirá as
    seguintes informações:
        * productId: uma string que identifica exclusivamente cada produto
        * userId: uma string que identifica exclusivamente cada usuário
        * summary: o título da avaliação
        * helpfulY: o número de usuários que consideraram esta avaliação útil
        * helpfulN: o número de usuários que consideraram esta avaliação NÃO útil
    """

    global PYTHON3

    basic_fields = {'sentiment', 'text'}
    numeric_fields = {'sentiment', 'helpfulY', 'helpfulN'}

    data = []
    if PYTHON3:
        f_data = open(path_data, encoding="latin1")
    else:
        f_data = open(path_data)

    for datum in csv.DictReader(f_data, delimiter='\t'):
        for field in list(datum.keys()):
            if not extras and field not in basic_fields:
                del datum[field]
            elif field in numeric_fields and datum[field]:
                datum[field] = int(datum[field])

        data.append(datum)

    f_data.close()

    return data


def write_predictions(path_submit_data, preds):
    if PYTHON3:
        f_data = open(path_submit_data, encoding="latin1")
    else:
        f_data = open(path_submit_data)

    reader = csv.DictReader(f_data, delimiter='\t')
    data = list(reader)

    assert len(preds) == len(data), 'Expected {} predictions but {} were given.'.format(
        len(data), len(preds)
    )

    for pred, datum in zip(preds.astype(int), data):
        assert pred == 1 or pred == -1, 'Invalid prediction: {}.'.format(pred)
        datum['sentiment'] = pred
    f_data.close()

    if PYTHON3:
        f_out = open(path_submit_data, 'w')
    else:
        f_out = open(path_submit_data, 'wb')

    writer = csv.DictWriter(f_out, delimiter='\t', fieldnames=reader.fieldnames)
    writer.writeheader()
    for datum in data:
        writer.writerow(datum)
    f_out.close()


def plot_toy_data(algo_name, features, labels, thetas):
    """
    Plota os dados de teste em 2D.
    Argumentos:
        * features - um ndarray Nx2 de características (pontos)
        * labels - um vetor de comprimento N com rótulos +1/-1
        * thetas - a tupla (theta, theta_0) que representa a saída do algoritmo de aprendizado
        * algorithm - o nome do algoritmo de aprendizado utilizado (string)
    """
    # plot the points with labels represented as colors
    plt.subplots()
    colors = ['b' if label == 1 else 'r' for label in labels]
    plt.scatter(features[:, 0], features[:, 1], s=40, c=colors)
    xmin, xmax = plt.axis()[:2]

    # plot the decision boundary
    theta, theta_0 = thetas
    xs = np.linspace(xmin, xmax)
    ys = -(theta[0] * xs + theta_0) / (theta[1] + 1e-16)
    plt.plot(xs, ys, 'k-')

    # show the plot
    algo_name = ' '.join((word.capitalize() for word in algo_name.split(' ')))
    plt.suptitle('Classified Toy Data ({})'.format(algo_name))
    plt.show()


def plot_tune_results(algo_name, param_name, param_vals, acc_train, acc_val):
    """
    Gráfico da precisão da classificação nos dados de treinamento e validação versus
    vários valores de um hiperparâmetro usado durante o treinamento.
    """
    # put the data on the plot
    plt.subplots()
    plt.plot(param_vals, acc_train, '-o')
    plt.plot(param_vals, acc_val, '-o')

    # make the plot presentable
    algo_name = ' '.join((word.capitalize() for word in algo_name.split(' ')))
    param_name = param_name.capitalize()
    plt.suptitle('Classification Accuracy vs {} ({})'.format(param_name, algo_name))
    plt.legend(['train', 'val'], loc='upper right', title='Partition')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy (%)')
    plt.savefig(f"../images/{algo_name}_{param_name}.png")
    plt.show()


def tune(train_fn, param_vals, train_feats, train_labels, val_feats, val_labels):
    train_accs = np.ndarray(len(param_vals))
    val_accs = np.ndarray(len(param_vals))

    for i, val in enumerate(param_vals):
        theta, theta_0 = train_fn(train_feats, train_labels, val)

        train_preds = p1.classify(train_feats, theta, theta_0)
        train_accs[i] = p1.accuracy(train_preds, train_labels)

        val_preds = p1.classify(val_feats, theta, theta_0)
        val_accs[i] = p1.accuracy(val_preds, val_labels)

    return train_accs, val_accs


def tune_perceptron(*args):
    return tune(p1.perceptron, *args)


def tune_avg_perceptron(*args):
    return tune(p1.average_perceptron, *args)


def tune_pegasos_T(best_L, *args):
    def train_fn(features, labels, T):
        return p1.pegasos(features, labels, T, best_L)

    return tune(train_fn, *args)


def tune_pegasos_L(best_T, *args):
    def train_fn(features, labels, L):
        return p1.pegasos(features, labels, best_T, L)

    return tune(train_fn, *args)


def most_explanatory_word(theta, wordlist):
    """Retorna a palavra associada à característica do modelo de saco de palavras com maior peso."""
    return [word for (theta_i, word) in sorted(zip(theta, wordlist))[::-1]]
