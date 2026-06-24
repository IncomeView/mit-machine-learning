import random
from string import digits, punctuation

import numpy as np

# ==============================================================================
# ===  PART I  =================================================================
# ==============================================================================


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Calcula a perda de dobradiça em um único ponto de dados, dados parâmetros de classificação específicos.
    Argumentos:
        x   = `feature_vector` - array numpy descrevendo o ponto de dados fornecido.
        y   = `label` - float, a classificação correta do ponto de dados.
        𝜃   = `theta` - array numpy descrevendo o classificador linear.
        𝜃_0 = `theta_0` - float representando o parâmetro de deslocamento.
    Retorna:
        a perda de dobradiça, como um float, associada ao ponto de dados e aos parâmetros fornecidos.
    """
    score = np.matmul(theta, feature_vector) + theta_0
    loss = max(0, 1 - label * score)
    return loss


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Calcula a perda de dobradiça para parâmetros de classificação fornecidos, calculada como a média em um
    conjunto de dados fornecido.
    Argumentos:
        `feature_matrix` - matriz numpy que descreve os dados fornecidos. Cada linha
           representa um único ponto de dados.
        `labels` - array numpy onde o k-ésimo elemento do array é a
            classificação correta da k-ésima linha da matriz de características.
        `theta` - array numpy que descreve o classificador linear.
        `theta_0` - número real que representa o parâmetro de deslocamento.
    Retorna:
        a perda de dobradiça, como um número de ponto flutuante, associada ao conjunto de dados e aos
        parâmetros fornecidos. Este número deve ser a perda de dobradiça média em todos os dados.
    """
    loss_full = 0
    n_samples = feature_matrix.shape[0]
    for i in range(n_samples):
        loss_full += hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
    return loss_full / n_samples


def perceptron_single_step_update(
    feature_vector, label, current_theta, current_theta_0
):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
    score = np.matmul(current_theta, feature_vector) + current_theta_0
    if label * score <= 0:
        theta = current_theta + label * feature_vector
        theta_0 = current_theta_0 + label
    else:
        theta = current_theta
        theta_0 = current_theta_0
    return theta, theta_0


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set: we do not stop early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the feature-coefficient parameter `theta` as a numpy array
            (found after T iterations through the feature matrix)
        the offset parameter `theta_0` as a floating point number
            (found also after T iterations through the feature matrix).
    """
    n_samples, n_features = feature_matrix.shape
    theta = np.zeros(n_features)
    theta_0 = 0.0

    for t in range(T):
        for i in get_order(n_samples):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0
            )

    return theta, theta_0


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given dataset.  Runs `T`
    iterations through the dataset (we do not stop early) and therefore
    averages over `T` many parameter values.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: It is more difficult to keep a running average than to sum and
    divide.

    Args:
        `feature_matrix` -  A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the average feature-coefficient parameter `theta` as a numpy array
            (averaged over T iterations through the feature matrix)
        the average offset parameter `theta_0` as a floating point number
            (averaged also over T iterations through the feature matrix).
    """
    n_samples, n_features = feature_matrix.shape
    theta = np.zeros(n_features)
    theta_0 = 0.0

    theta_soma = np.zeros(n_features)
    theta_0_soma = 0.0
    nT = 0

    for t in range(T):
        for i in get_order(n_samples):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0
            )
            theta_soma += theta
            theta_0_soma += theta_0
            nT += 1
    theta_med = theta_soma / nT
    theta_0_med = theta_0_soma / nT

    return theta_med, theta_0_med


def pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0):
    """
    Atualiza os parâmetros de classificação `theta` e `theta_0` por meio de uma única
    passagem do algoritmo Pegasos. Retorna novos parâmetros em vez de
    modificá-los no local.
    Argumentos:
        x   = `feature_vector` - Um array numpy descrevendo um único ponto de dados.
        y   = `label` - A classificação correta do vetor de características.
        𝜆   = `L` - O valor lambda usado para atualizar os parâmetros.
        𝜂   = `eta` - Taxa de aprendizado para atualizar os parâmetros.
        𝜃   = `theta` - O valor antigo de theta usado pelo algoritmo Pegasos
                antes desta atualização.
        𝜃_0 = `theta_0` - O valor antigo de theta_0 usado pelo
                algoritmo Pegasos antes desta atualização.
    Retorna:
    uma tupla onde o primeiro elemento é um array numpy com o valor de
    theta após a conclusão da atualização anterior e o segundo elemento é um
    número real com o valor de theta_0 após a conclusão da atualização anterior.
    """
    theta_novo = (1 - eta * L) * theta
    theta_0_novo = theta_0
    score = label * (np.matmul(theta, feature_vector) + theta_0)

    if score <= 1:
        theta_novo = theta_novo + eta * label * feature_vector
        theta_0_novo = theta_0_novo + eta * label

    return theta_novo, theta_0_novo


def pegasos(feature_matrix, labels, T, L):
    """
    Executa o algoritmo Pegasos em um conjunto de dados fornecido. Executa T iterações
    pelo conjunto de dados, não há necessidade de se preocupar em parar prematuramente. Para
    cada atualização, defina a taxa de aprendizado = 1/sqrt(t), onde t é um contador para o
    número de atualizações realizadas até o momento (entre 1 e nT, inclusive).

    NOTE: Utilize as funções implementadas anteriormente quando aplicável.
            Não copie e cole código de partes anteriores.

    Argumentos:
        x  = `feature_matrix` - Uma matriz numpy que descreve os dados fornecidos. Cada linha
                representa um único ponto de dados.
        y  = `labels` - Um array numpy onde o k-ésimo elemento do array é a
                classificação correta da k-ésima linha da matriz de características.
        T  = `T` - Um inteiro que indica quantas vezes o algoritmo
                deve iterar pela matriz de características.
        𝜆  = `L` - O valor lambda usado para atualizar os parâmetros do algoritmo Pegasos.
    Retorna:
        uma tupla
        onde o primeiro elemento é um array numpy com o valor de theta,
        o parâmetro de classificação linear, encontrado após T iterações
        na matriz de características
        e o segundo elemento é um número real com o valor de theta_0,
        o parâmetro de classificação de deslocamento, encontrado
        após T iterações na matriz de características.

    """
    n_samples, n_features = feature_matrix.shape
    theta = np.zeros(n_features)
    theta_0 = 0.0
    t = 1

    for _ in range(T):
        for i in get_order(n_samples):
            eta = 1 / np.sqrt(t)
            theta, theta_0 = pegasos_single_step_update(
                feature_matrix[i], labels[i], L, eta, theta, theta_0
            )
            t += 1

    return theta, theta_0


# ==============================================================================
# ===  PART II  ================================================================
# ==============================================================================


##  #pragma: coderesponse template
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end


def classify(feature_matrix, theta, theta_0):
    """
    Uma função de classificação que usa parâmetros fornecidos para classificar um conjunto de
        pontos de dados.
    Argumentos:
        x   = `feature_matrix` - matriz numpy que descreve os dados fornecidos. Cada linha
                            representa um único ponto de dados.
        𝜃   = `theta` - array numpy que descreve o classificador linear.
        𝜃_0 = `theta_0` - número real que representa o parâmetro de deslocamento.
    Retorna:
        um array numpy de 1s e -1s, onde o k-ésimo elemento do array é a
        classificação prevista da k-ésima linha da matriz de características usando os
        valores de theta e theta_0 fornecidos. Se uma previsão for MAIOR QUE zero, ela
        deve ser considerada uma classificação positiva.

    """
    scores = np.matmul(feature_matrix, theta) + theta_0
    return np.where(scores > 0, 1, -1)


def classifier_accuracy(
    classifier,
    train_feature_matrix,
    val_feature_matrix,
    train_labels,
    val_labels,
    **kwargs,
):
    """
    Treina um classificador linear e calcula a acurácia. O classificador é
    treinado nos dados de treinamento. A acurácia do classificador nos dados de treinamento e
    validação é então retornada.
    Argumentos:
        `classifier` - Uma função de aprendizado que recebe argumentos
            (matriz de características, rótulos, **kwargs) e retorna (theta, theta_0)
        `train_feature_matrix` - Uma matriz numpy que descreve os dados de treinamento.
            Cada linha representa um único ponto de dados.
        `val_feature_matrix` - Uma matriz numpy que descreve os dados de validação.
            Cada linha representa um único ponto de dados.
        `train_labels` - Um array numpy onde o k-ésimo elemento do array
            é a classificação correta da k-ésima linha da matriz de características de treinamento.
        `val_labels` - Um array numpy onde o k-ésimo elemento do array
            é a classificação correta da k-ésima linha da matriz de características de validação.
        `kwargs` - Argumentos nomeados adicionais para passar ao classificador
            (ex.: T ou L)
    Retorna:
        uma tupla na qual o primeiro elemento é a acurácia (escalar) do
        classificador treinado nos dados de treinamento e o segundo elemento é a
        acurácia do classificador treinado nos dados de validação.
    """
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_preds = classify(train_feature_matrix, theta, theta_0)
    val_preds = classify(val_feature_matrix, theta, theta_0)
    train_accuracy = np.mean(train_preds == train_labels)
    val_accuracy = np.mean(val_preds == val_labels)

    return train_accuracy, val_accuracy


def extract_words(text):
    """
    Função auxiliar para `bag_of_words(...)`.
    Argumentos:
        uma string `text`.
    Retorno:
        uma lista de palavras em minúsculas na string, onde pontuação e dígitos
        contam como palavras separadas.
    """
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()


def bag_of_words(texts, remove_stopword=None):
    """
    NOTE: sinta-se à vontade para alterar este código conforme as orientações da Seção 3 (por exemplo, remover
            stopwords, adicionar bigramas etc.)
    Argumentos:
        `texts` - uma lista de strings em linguagem natural.
    Retorna:
        um dicionário que mapeia cada palavra presente em `texts` para um índice inteiro único.
    """
    indices_by_word = {}  # mapear cada palavra para um índice inteiro único
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if remove_stopword and word in remove_stopword:
                continue
            if word not in indices_by_word:
                indices_by_word[word] = len(indices_by_word)
    return indices_by_word


def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Argumentos:
        `reviews` - ​​uma lista de strings em linguagem natural
        `indices_by_word` - um dicionário de palavras com índices únicos.
    Retorna:
        uma matriz representando cada avaliação por meio de características de saco de palavras.
        Esta matriz tem, portanto, formato (n, m), onde n conta as avaliações e m conta as palavras
        no dicionário.
    """
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word:
                continue
            feature_matrix[i, indices_by_word[word]] += 1
    if binarize:
        feature_matrix[feature_matrix > 0] = 1
    return feature_matrix


def accuracy(preds, targets):
    """
    Dado um vetor de comprimento N contendo rótulos previstos e alvo,
    retorna a fração de previsões que estão corretas.
    """
    return (preds == targets).mean()
