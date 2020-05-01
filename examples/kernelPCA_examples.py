from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import emby
import numpy as np


def digits():
    digits = load_digits()

    x = digits.data
    y = digits.target
    x = x - x.mean()

    kcpa = emby.KernelPCA(Z=2, kernel="gaussian", variance=60)

    colors = np.array(list(mcolors.TABLEAU_COLORS.values()))

    embeddings = kcpa.fit_transform(x)

    plt.figure(figsize=(14, 14))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color=colors[y])
    plt.savefig("../images/digits-kpca.png", bbox_inches="tight")
    plt.show()


def simple_clusters():
    x = np.concatenate([
        np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
    ])

    y = np.concatenate([np.zeros(500), np.ones(500)]).astype(np.int)

    kcpa = emby.KernelPCA(Z=2, kernel="polynomial")

    colors = np.array(list(mcolors.TABLEAU_COLORS.values()))

    embeddings = kcpa.fit_transform(x) + np.random.normal(0, 0.01, size=(len(x), 2))

    plt.figure(figsize=(14, 14))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color=colors[y])
    plt.savefig("../images/digits-kpca.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    #simple_clusters()
    digits()
