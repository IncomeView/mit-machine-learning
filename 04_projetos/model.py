# %% [markdown]
# # MIT 6.86x — Project 1
# ## Sentiment Analysis with Linear Classifiers
# ### Notebook em formato .py — Moacir
#
# Este arquivo foi pensado para ser convertido em Jupyter Notebook
# usando **Jupytext**:
#
# ```bash
# jupytext --to ipynb model.py
# ```
#
# Ele contempla:
# - Problem 2: perceptron_single_step_update
# - Problem 3: perceptron
# - Problem 4: average perceptron
# - Problem 5: Pegasos
# - Problem 7: comparação dos algoritmos
# - Problem 8: tuning de hiperparâmetros
# - Modelo final + palavras mais explicativas

# %% [markdown]
# ## Imports e configuração

# %%
import sys
import numpy as np

# Ajuste do path para acessar os módulos do projeto
sys.path.append("sentiment_analysis")

import utils
import project1 as p1

# %% [markdown]
# ## Carregamento dos dados
#
# Usamos os arquivos:
# - `reviews_train.tsv`
# - `reviews_val.tsv`
# - `reviews_test.tsv`
#
# dentro de `sentiment_analysis/`.

# %%
train_data = utils.load_data('sentiment_analysis/reviews_train.tsv')
val_data = utils.load_data('sentiment_analysis/reviews_val.tsv')
test_data = utils.load_data('sentiment_analysis/reviews_test.tsv')

train_texts, train_labels = zip(*[(d['text'], d['sentiment']) for d in train_data])
val_texts, val_labels = zip(*[(d['text'], d['sentiment']) for d in val_data])
test_texts, test_labels = zip(*[(d['text'], d['sentiment']) for d in test_data])

# %% [markdown]
# ## Problem 2 — `perceptron_single_step_update`
#
# Aqui assumimos que a função já está implementada em `project1.py`.
# Vamos apenas demonstrar uma chamada simples para visualizar o comportamento.

# %%
x_example = np.array([1.0, 2.0, -1.0])
y_example = 1
theta_example = np.zeros_like(x_example)
theta_0_example = 0.0

theta_updated, theta_0_updated = p1.perceptron_single_step_update(
    x_example, y_example, theta_example, theta_0_example
)

print("Problem 2 — perceptron_single_step_update")
print("theta antes:", theta_example)
print("theta depois:", theta_updated)
print("theta_0 depois:", theta_0_updated)

# %% [markdown]
# ## Bag-of-Words
#
# Construímos o dicionário a partir dos textos de treino
# e extraímos as matrizes de features em formato BoW.

# %%
dictionary = p1.bag_of_words(train_texts)

train_bow = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow = p1.extract_bow_feature_vectors(test_texts, dictionary)

# %% [markdown]
# ## Problem 3 — Perceptron completo
#
# Rodamos o perceptron no conjunto de treino BoW.

# %%
T = 10

theta_p, theta_0_p = p1.perceptron(train_bow, train_labels, T=T)

print("\nProblem 3 — Perceptron")
print("theta shape:", theta_p.shape)
print("theta_0:", theta_0_p)

# %% [markdown]
# ## Problem 4 — Average Perceptron

# %%
theta_avg, theta_0_avg = p1.average_perceptron(train_bow, train_labels, T=T)

print("\nProblem 4 — Average Perceptron")
print("theta shape:", theta_avg.shape)
print("theta_0:", theta_0_avg)

# %% [markdown]
# ## Problem 5 — Pegasos

# %%
L = 0.01

theta_peg, theta_0_peg = p1.pegasos(train_bow, train_labels, T=T, L=L)

print("\nProblem 5 — Pegasos")
print("theta shape:", theta_peg.shape)
print("theta_0:", theta_0_peg)

# %% [markdown]
# ## Problem 7 — Comparação dos algoritmos
#
# Aqui usamos `classifier_accuracy` para comparar:
# - Perceptron
# - Average Perceptron
# - Pegasos

# %%
pct_train, pct_val = p1.classifier_accuracy(
    p1.perceptron, train_bow, val_bow, train_labels, val_labels, T=T
)
avg_train, avg_val = p1.classifier_accuracy(
    p1.average_perceptron, train_bow, val_bow, train_labels, val_labels, T=T
)
peg_train, peg_val = p1.classifier_accuracy(
    p1.pegasos, train_bow, val_bow, train_labels, val_labels, T=T, L=L
)

print("\n=== Problem 7 — Resultados ===")
print("Perceptron:       treino =", pct_train, "val =", pct_val)
print("Avg Perceptron:   treino =", avg_train, "val =", avg_val)
print("Pegasos:          treino =", peg_train, "val =", peg_val)

# %% [markdown]
# ## Problem 8 — Tuning de hiperparâmetros
#
# Testamos:
# - vários valores de T para cada algoritmo
# - vários valores de λ (L) para Pegasos

# %%
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]

# Perceptron
percep_results = [
    (
        T_i,
        p1.classifier_accuracy(
            p1.perceptron, train_bow, val_bow, train_labels, val_labels, T=T_i
        )[1],
    )
    for T_i in Ts
]

# Avg Perceptron
avg_results = [
    (
        T_i,
        p1.classifier_accuracy(
            p1.average_perceptron, train_bow, val_bow, train_labels, val_labels, T=T_i
        )[1],
    )
    for T_i in Ts
]

# Pegasos — tuning T
peg_T_results = [
    (
        T_i,
        p1.classifier_accuracy(
            p1.pegasos, train_bow, val_bow, train_labels, val_labels, T=T_i, L=0.01
        )[1],
    )
    for T_i in Ts
]

# Pegasos — tuning L
peg_L_results = [
    (
        L_i,
        p1.classifier_accuracy(
            p1.pegasos, train_bow, val_bow, train_labels, val_labels, T=25, L=L_i
        )[1],
    )
    for L_i in Ls
]

print("\n=== Problem 8 — Tuning ===")
print("Perceptron (T):", percep_results)
print("Avg Perceptron (T):", avg_results)
print("Pegasos (T):", peg_T_results)
print("Pegasos (L):", peg_L_results)

# %% [markdown]
# ## Modelo final — Pegasos com melhor T e λ
#
# Usamos:
# - T = 25
# - λ = 0.01

# %%
best_T = 25
best_L = 0.01

theta_final, theta_0_final = p1.pegasos(train_bow, train_labels, T=best_T, L=best_L)

test_preds = p1.classify(test_bow, theta_final, theta_0_final)
test_acc = np.mean(test_preds == test_labels)

print("\n=== Modelo Final ===")
print("Pegasos com T =", best_T, "e L =", best_L)
print("Test accuracy:", test_acc)

# %% [markdown]
# ## Palavras mais explicativas
#
# Usamos `most_explanatory_word` para ordenar as palavras
# pelo peso em θ (theta).

# %%
wordlist = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_words = utils.most_explanatory_word(theta_final, wordlist)

print("\nMost Explanatory Word Features:")
print(sorted_words[:20])

# %% [markdown]
# ## Fim
#
# Este arquivo `.py` pode ser convertido para `.ipynb` com:
#
# ```bash
# jupytext --to ipynb model.py
# ```

# %%
print("\nmodel.py executado com sucesso.")
~~~text
