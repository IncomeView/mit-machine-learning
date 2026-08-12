import numpy as np
import kmeans
import common
import naive_em
import em


def run_kmeans(X):
    Ks = [5, 6]
    seeds = [0, 1, 2, 3, 4]

    for K in Ks:
        best_cost = float("inf")
        best_mixture = None
        best_post = None
        best_seed = None

        for seed in seeds:
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)

            if cost < best_cost:
                best_cost = cost
                best_mixture = mixture
                best_post = post
                best_seed = seed

        print(f"K-means: Cost K={K} = {best_cost:.4f} (seed={best_seed})")
        common.plot(X, best_mixture, best_post, f"K-means K={K}, seed={best_seed}")


def run_em(X):
    Ks = [5, 6]
    seeds = [0, 1, 2, 3, 4]

    for K in Ks:
        best_LL = -np.inf
        best_mixture = None
        best_post = None
        best_seed = None

        for seed in seeds:
            mixture, post = common.init(X, K, seed)
            mixture, post, LL = naive_em.run(X, mixture, post)

            if LL > best_LL:
                best_LL = LL
                best_mixture = mixture
                best_post = post
                best_seed = seed

        print(f"EM: Log-verossimilhança K={K} = {best_LL:.4f} (seed={best_seed})")
        common.plot(X, best_mixture, best_post, f"EM K={K}, seed={best_seed}")


def run_em_single(X, K, seed):
    mixture, post = common.init(X, K, seed)
    mixture, post, L = naive_em.run(X, mixture, post)

    return mixture, post, L


def run_BIC(X):
    ks = [1, 2, 3, 4]
    seeds = [0, 1, 2, 3, 4]
    best_k = None
    best_bic = -np.inf

    for K in ks:
        best_L = -np.inf
        best_mix = None

        # rodar EM para várias seeds
        for seed in seeds:
            mixture, post, L = run_em_single(X, K, seed)

            if L > best_L:
                best_L = L
                best_mix = mixture

        # calcula BIC usando a função bic() do arquivo common.py
        bic_score = common.bic(X, best_mix, best_L)
        print(f"K={K}, BIC={bic_score: .4f}")

        # Escolher o melhor K
        if bic_score > best_bic:
            best_bic = bic_score
            best_k = K

    print("\nMelhor K =", best_k)
    print("Melhor BIC =", best_bic)


def run_ll_em(X):

    Ks = [1, 12]
    seeds = [0, 1, 2, 3, 4]

    for K in Ks:
        print(f"\n=== Avaliando K={K} ===")

        best_ll = -np.inf
        best_mixture = None
        best_post = None
        best_seed = None

        for seed in seeds:
            np.random.seed(seed)

            mixture, post = common.init(X, K, seed)
            mixture, post, ll = em.run(X, mixture, post)
            bic = common.bic(X, mixture, ll)

            print(f"seed={seed}, LL={ll:.4f}, BIC={bic:.4f}")

            if ll > best_ll:
                best_ll = ll
                best_mixture = mixture
                best_post = post
                best_seed = seed

        print(f"\n>>> Melhor seed para K={K}: {best_seed} com LL={best_ll:.4f}")
        print("p:", mixture.p[:5])
        print("var:", mixture.var[:5])
        print("mu[0,:5]:", mixture.mu[0, :5])


# def main():
#     X = np.loadtxt("toy_data.txt")
# #    run_kmeans(X)
# #    run_em(X)
# #    run_BIC(X)
#     run_ll_em(X)


def main():
    X = np.loadtxt("netflix_incomplete.txt")

    # roda o EM como você já faz
    run_ll_em(X)

    ''' TESTE DO FILL_MATRIX '''
    print("\n=== Testando fill_matrix ===")

    # Treina um modelo EM para preencher a matriz
    K = 12
    seed = 0
    mixture, post = common.init(X, K, seed)
    mixture, post, ll = em.run(X, mixture, post)

    # Preenche a matriz
    filled = em.fill_matrix(X, mixture)

    # Testes básicos
    print("Zeros antes:", np.sum(X == 0))
    print("Zeros depois:", np.sum(filled == 0))
    print("Min preenchido:", filled.min())
    print("Max preenchido:", filled.max())

    # Teste de plausibilidade para um usuário e filme
    u, i = 10, 20
    print(f"\nTeste manual para usuário {u}, filme {i}:")
    print("Valor preenchido:", filled[u, i])
    print("Média ponderada:", np.dot(post[u], mixture.mu[:, i]))

    # encontre um zero real
    u, i = np.argwhere(X == 0)[0]

    print("Par escolhido:", u, i)
    print("Original:", X[u, i])
    print("Preenchido:", filled[u, i])
    print("Média ponderada:", np.dot(post[u], mixture.mu[:, i]))

    print("\n=== Avaliando RMSE com matriz completa ===")

    # Carrega a matriz completa (gold standard)
    X_gold = np.loadtxt("netflix_complete.txt")

    # Usa o melhor modelo encontrado para K=12
    K = 12
    best_seed = 1  # você viu que seed=1 deu o melhor LL
    mixture, post = common.init(X, K, best_seed)
    mixture, post, ll = em.run(X, mixture, post)

    # Preenche a matriz incompleta
    X_pred = em.fill_matrix(X, mixture)

    # Calcula RMSE apenas nas posições que eram observadas no gold
    rmse_value = common.rmse(X_gold, X_pred)

    print("RMSE:", rmse_value)


if __name__ == "__main__":
    main()
