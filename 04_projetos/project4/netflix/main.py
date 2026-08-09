import numpy as np
import kmeans
import common
import naive_em
import em


def run_kmeans(X):
    Ks = [1, 2, 3, 4]
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
    Ks = [1, 2, 3, 4]
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


def main():
    X = np.loadtxt("toy_data.txt")

    run_kmeans(X)


#    run_em(X)


if __name__ == "__main__":
    main()
