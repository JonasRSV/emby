import unittest
import numpy as np
from emby import SOM


class SOMTest(unittest.TestCase):
    def test_similarities(self):
        x = np.concatenate([
            np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
            np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ])

        som = SOM(Z=2, bases=3)
        som.fit(x)
        similarities = som.base_similarities()

        self.assertEqual(np.round(similarities).sum(), 5)

    def test_closest_base(self):
        x = np.concatenate([
            np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
            np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ])

        som = SOM(Z=2, bases=2, epochs=50, learning_rate=0.01)
        som.fit(x)
        closest = som.closest_base(x)

        label = np.concatenate([np.zeros(500), np.ones(500)])

        self.assertTrue(np.all(closest == label) or np.all(closest == np.flip(label)))

    def test_different_z(self):
        x = np.concatenate([
            np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
            np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ])

        for z in range(1, 4):
            som = SOM(Z=z, bases=20, epochs=50, learning_rate=0.01)
            som.fit_transform(x)






if __name__ == '__main__':
    unittest.main()
