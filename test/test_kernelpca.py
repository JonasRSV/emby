import unittest
import numpy as np
from emby import KernelPCA


class KernelPCATest(unittest.TestCase):
    def test_fittransform(self):
        x = np.concatenate([
            np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=10),
            np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=10)
        ])

        for z in range(1, 10):
            kpca = KernelPCA(Z=z)
            kpca.fit_transform(x)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
