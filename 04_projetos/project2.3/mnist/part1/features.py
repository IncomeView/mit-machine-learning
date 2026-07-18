import numpy as np
import matplotlib.pyplot as plt


def project_onto_PC(X, pcs, n_components, feature_means):
    """
    Dados os vetores de componentes principais `pcs = principal_components(X)`,
    esta função retorna um novo array de dados no qual cada amostra de X
    foi projetada nos primeiros `n_components` componentes principais.
    """
    # TODO: primeiro, centralize os dados usando as médias das características (feature_means)
    # TODO: Retorne a projeção do conjunto de dados centralizado
    #       nos primeiros `n_components` componentes principais.
    #       O resultado deve ser um array com dimensões: n x n_components.
    # Dica: esses componentes principais correspondem às primeiras `n_components` colunas
    #       dos autovetores retornados por `principal_components()`.
    #       Observe que cada autovetor já é um vetor unitário,
    #       portanto, a projeção pode ser realizada utilizando multiplicação de matrizes.

    # Centralizar X nos primeiros n_components componentes principais
    X_centered = X - feature_means

    # Selecionar os primeiros n_components autovetores (componentes principais)
    # pcs tem shape (d, d), Regamos as primeiras colunas
    V = pcs[:, :n_components]

    # Projeção: X_centered * V
    X_pca = np.dot(X_centered, V)

    return X_pca


### Functions which are already complete, for you to use ###


def cubic_features(X):
    """
    Retorna um novo conjunto de dados com atributos definidos pelo mapeamento
    que corresponde ao kernel cúbico.
    """
    n, d = X.shape  # dataset size, input dimension
    X_withones = np.ones((n, d + 1))
    X_withones[:, :-1] = X
    new_d = 0  # dimension of output
    new_d = int((d + 1) * (d + 2) * (d + 3) / 6)

    new_data = np.zeros((n, new_d))
    col_index = 0
    for x_i in range(n):
        X_i = X[x_i]
        X_i = X_i.reshape(1, X_i.size)

        if d > 2:
            comb_2 = np.matmul(np.transpose(X_i), X_i)

            unique_2 = comb_2[np.triu_indices(d, 1)]
            unique_2 = unique_2.reshape(unique_2.size, 1)
            comb_3 = np.matmul(unique_2, X_i)
            keep_m = np.zeros(comb_3.shape)
            index = 0
            for i in range(d - 1):
                keep_m[index + np.arange(d - 1 - i), i] = 0

                tri_keep = np.triu_indices(d - 1 - i, 1)

                correct_0 = tri_keep[0] + index
                correct_1 = tri_keep[1] + i + 1

                keep_m[correct_0, correct_1] = 1
                index += d - 1 - i

            unique_3 = np.sqrt(6) * comb_3[np.nonzero(keep_m)]

            new_data[x_i, np.arange(unique_3.size)] = unique_3
            col_index = unique_3.size

    for i in range(n):
        newdata_colindex = col_index
        for j in range(d + 1):
            new_data[i, newdata_colindex] = X_withones[i, j] ** 3
            newdata_colindex += 1
            for k in range(j + 1, d + 1):
                new_data[i, newdata_colindex] = (
                    X_withones[i, j] ** 2 * X_withones[i, k] * (3 ** (0.5))
                )
                newdata_colindex += 1

                new_data[i, newdata_colindex] = (
                    X_withones[i, j] * X_withones[i, k] ** 2 * (3 ** (0.5))
                )
                newdata_colindex += 1

                if k < d:
                    new_data[i, newdata_colindex] = (
                        X_withones[i, j] * X_withones[i, k] * (6 ** (0.5))
                    )
                    newdata_colindex += 1

    return new_data


def center_data(X):
    """
    Retorna uma versão centralizada dos dados, na qual cada atributo passa a ter média = 0

    Argumentos:
        X - array NumPy de dimensão n x d contendo n pontos de dados, cada um com d atributos
    Retorna:
        - array NumPy X' de dimensão (n, d), onde para cada i = 1, ..., n e j = 1, ..., d:
        X'[i][j] = X[i][j] - means[j]
        - array NumPy de dimensão (d, ) contendo as médias das colunas
    """
    feature_means = X.mean(axis=0)
    return (X - feature_means), feature_means


def principal_components(centered_data):
    """
    Retorna os vetores dos componentes principais dos dados, ordenados em ordem decrescente
    de magnitude do autovalor. Esta função primeiro calcula a matriz de covariância
    e depois encontra seus autovetores.

    Argumentos:
        centered_data - matriz NumPy n x d de n pontos de dados, cada um com d características

    Retorna:
        matriz NumPy d x d cujas colunas são as direções dos componentes principais, ordenadas
        em ordem decrescente pela quantidade de variação em cada direção (estes são
        equivalentes aos d autovetores da matriz de covariância, ordenados em ordem decrescente
        de autovalores, portanto, a primeira coluna corresponde ao autovetor com
        o maior autovalor
    """
    scatter_matrix = np.dot(centered_data.transpose(), centered_data)
    eigen_values, eigen_vectors = np.linalg.eig(scatter_matrix)
    # Re-order eigenvectors by eigenvalue magnitude:
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    return eigen_vectors


###Correction note:  Differing from the release, this function takes an extra input feature_means.


def plot_PC(X, pcs, labels, feature_means):
    """
    Dado os vetores dos componentes principais como as colunas da matriz pcs,
    esta função projeta cada amostra em X nos dois primeiros componentes principais
    e produz um gráfico de dispersão onde os pontos são marcados com o dígito representado na
    imagem correspondente.

    labels = um array numpy contendo os dígitos correspondentes a cada imagem em X.
    """
    pc_data = project_onto_PC(X, pcs, n_components=2, feature_means=feature_means)
    text_labels = [str(z) for z in labels.tolist()]
    fig, ax = plt.subplots()
    ax.scatter(pc_data[:, 0], pc_data[:, 1], alpha=0, marker=".")
    for i, txt in enumerate(text_labels):
        ax.annotate(txt, (pc_data[i, 0], pc_data[i, 1]))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.show()


###Correction note:  Differing from the release, this function takes an extra input feature_means.


def reconstruct_PC(x_pca, pcs, n_components, X, feature_means):
    """
    Dados os vetores de componentes principais como as colunas da matriz `pcs`,
    esta função reconstrói uma única imagem a partir de sua representação
    por componentes principais, `x_pca`.
    `X` = os dados originais aos quais o PCA foi aplicado para obter `pcs`.
    """
    x_reconstructed = np.dot(x_pca, pcs[:, range(n_components)].T) + feature_means
    return x_reconstructed
