from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import emby
import numpy as np


def digits():
    digits = load_digits()

    x = digits.data
    y = digits.target

    som = emby.SOM(Z=2, bases=100, epochs=100, logging=emby.Logging.Everything, y_variance=0.1)

    colors = np.array(list(mcolors.TABLEAU_COLORS.values()))

    embeddings = som.fit_transform(x) + np.random.normal(0, 0.01, size=(len(x), 2))

    plt.figure(figsize=(14, 14))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color=colors[y])
    plt.savefig("../images/digits-som.png", bbox_inches="tight")
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

    som = emby.SOM(Z=2, bases=100, epochs=40, logging=emby.Logging.Progress, y_variance=0.1)

    colors = np.array(list(mcolors.TABLEAU_COLORS.values()))

    embeddings = som.fit_transform(x) + np.random.normal(0, 0.01, size=(len(x), 2))

    plt.figure(figsize=(14, 14))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color=colors[y])
    plt.savefig("../images/mnist-som.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    digits()
    #mnist()
