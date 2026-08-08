import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

Ks = [5, 6]
seeds = [0, 1, 2, 3, 4]

for K in Ks:
    b_cost = float("inf")
    b_mixture = None
    b_post = None
    b_seed = None

    for seed in seeds:
        # Inicializa mistura e responsabilidades
        mixture, post = common.init(X, K, seed)
        # Roda K-means
        mixture, post, cost = kmeans.run(X, mixture, post)

        if cost < b_cost:
            b_cost = cost
            b_mixture = mixture
            b_post = post
            b_seed = seed

    common.plot(X, b_mixture, b_post, f"K={K}, seed={b_seed}")
    print(f"Cost K={K} = {b_cost}")
