import unittest
import numpy as np
from gene.distances import L2_gene, a, tag_gene  #, pL2_gene, tag_gene



class TestL2Dist(unittest.TestCase):
    def test_L2_gene_simple(self):
        n1 = np.array([3, 8, 7, 6, 9, -47, 8, 12, -1, 0.2])
        n2 = np.zeros(10)
        self.assertAlmostEqual(L2_gene(n1, n2), np.linalg.norm(n1 - n2, 2), places=4)
    
    def test_L2_gene_harder(self):
        d = 1000
        n1 = np.random.randint(-100, 100, size=d)
        n2 = np.random.randint(-100, 100, size=d)
        self.assertAlmostEqual(L2_gene(n1, n2), np.linalg.norm(n1 - n2, 2),places=4)

    def test_L2_gene_floats(self):
        d = 1000
        n1 = np.random.randn(d)
        n2 = np.random.randn(d)
        self.assertAlmostEqual(L2_gene(n1, n2), np.linalg.norm(n1 - n2, 2), places=4)


class TestA(unittest.TestCase):
    # https://numpy.org/doc/stable/reference/routines.testing.html
    def test_a_scalar(self):
        x = 67.12
        self.assertEqual(1., a(x))
        self.assertEqual(-1., a(-x))

    def test_a_vector(self):
        x = np.arange(start=-5, stop=5)
        self.assertIsNone(
            np.testing.assert_array_equal(np.array([-1, -1, -1, -1, -1, 0, 1, 1, 1, 1]), a(x)))


class TestTagDist(unittest.TestCase):
    def test_tag_gene_simple(self):
        n1 = np.array([3, 8, 7, 6, 9, -47, 8, 12, -1, 0.2])
        n2 = np.zeros(10)
        # NOTE: test for succesfull run
        tag_gene(n1, n2)
        # self.assertAlmostEqual(tag_gene(n1, n2), np.linalg.norm(n1 - n2, 2), places=4)
        