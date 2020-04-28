from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import emby
import seaborn as sb
import numpy as np


def main():
    digits = load_digits()

    x = digits.data
    y = digits.target

    som = emby.SOM(Z=2, bases=9, verbose=True, y_variance=0.001)

    colors = np.array(list(mcolors.TABLEAU_COLORS.values()))

    embeddings = som.fit_transform(x) + np.random.normal(0, 0.05, size=(len(x), 2))

    plt.figure(figsize=(14, 14))
    plt.subplot(2, 1, 1)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color=colors[y])
    plt.subplot(2, 1, 2)
    sb.heatmap(som.base_similarities(), annot=True, cbar=False)
    plt.show()


if __name__ == "__main__":
    main()
