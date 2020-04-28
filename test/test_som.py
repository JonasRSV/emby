import unittest
import numpy as np
from emby import SOM


class SOMTest(unittest.TestCase):
    def test_similarities(self):
        x = np.concatenate([
            np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
            np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ])

        som = SOM(z=2, bases=3, epochs=30, y_variance=0.1, mode="uniform", learning_rate=0.01)
        som.fit(x)
        similarities = som.base_similarities()

        self.assertEqual(np.round(similarities).sum(), 5)


if __name__ == '__main__':
    unittest.main()
