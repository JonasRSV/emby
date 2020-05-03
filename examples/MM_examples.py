from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import emby
import numpy as np


def simple_clusters():
    x = np.concatenate([
        np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
    ])

    y = np.concatenate([np.zeros(500), np.ones(500)]).astype(np.int)

    mm = emby.MM(Z=2, z_variance=1.0, epochs=200, logging=emby.Logging.Everything)

    colors = np.array(list(mcolors.TABLEAU_COLORS.values()))

    embeddings = mm.fit_transform(x)

    plt.figure(figsize=(14, 14))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color=colors[y])
    plt.show()


def digits():
    digits = load_digits()

    x = digits.data
    y = digits.target
    x = x - x.mean()

    mm = emby.MM(Z=2, x_variance=40.0, z_variance=0.5, epochs=10000, logging=emby.Logging.Everything)

    colors = np.array(list(mcolors.TABLEAU_COLORS.values()))

    embeddings = mm.fit_transform(x)

    plt.figure(figsize=(14, 14))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color=colors[y])
    plt.savefig("../images/digits-mm.png", bbox_inches="tight")
    plt.show()


def mnist():
    import tensorflow_datasets as tfds

    mnist = tfds.load(name="mnist")["train"]
    mnist_data = []
    mnist_labels = []
    for sample in tfds.as_numpy(mnist):
        image = sample["image"]
        label = sample["label"]

        mnist_data.append(image.reshape(-1) / 256)
        mnist_labels.append(label)

    x, y = np.array(mnist_data), np.array(mnist_labels)

    em = emby.MM(Z=2, x_variance=50, z_variance=1.0, epochs=1000, logging=emby.Logging.Progress)

    colors = np.array(list(mcolors.TABLEAU_COLORS.values()))

    embeddings = em.fit_transform(x)

    plt.figure(figsize=(14, 14))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color=colors[y])
    plt.savefig("../images/mnist-mm.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # simple_clusters()
    digits()
    #mnist()
