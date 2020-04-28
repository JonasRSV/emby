import unittest
import numpy as np
from emby import SOM


class CoreTest(unittest.TestCase):
    def test_something(self):

        x = np.concatenate([
           np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=5),
           np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=5)
        ])
        som = SOM(Z=2, bases=3)
        som.fit(x)
        similaritites = som.base_similarities()

        print(similaritites)


if __name__ == '__main__':
    unittest.main()
