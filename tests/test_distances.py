import unittest

import numpy as np

from gene.distances import _L2_dist, L2_dist, pL2_dist, tag_dist, _a


class TestAFunc(unittest.TestCase):
    def test_a_(self):
        x = np.array([-5.0, -1.2, -0.5, 0.0, 0.5, 1.2, 5.0])

        self.assertIsNone(
            np.testing.assert_array_equal([-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0], _a(x))
        )


class TestL2Dist(unittest.TestCase):
    def test_L2_dist(self):
        x_1 = np.array([0.9, 1.2, -0.5, 4.0])
        x_2 = np.array([-1.0, 3.2, 0.4, -0.8])

        self.assertEqual(pL2_dist([x_1, x_2], 0, 1), np.linalg.norm(x_1 - x_2, 2))
        self.assertEqual(pL2_dist([-x_1, -x_2], 0, 1), np.linalg.norm(-x_1 - (-x_2), 2))


class TestpL2Dist(unittest.TestCase):
    def test_pL2_dist_pos(self):
        x_1 = np.array([0.9, 1.2, -0.5, 4.0])
        x_2 = np.array([-1.0, 3.2, 0.4, -0.8])

        self.assertEqual(pL2_dist([x_1, x_2], 0, 1), 1 * np.linalg.norm(x_1 - x_2, 2))

    def test_pL2_dist_neg(self):
        x_1 = np.array([0.9, 1.2, -0.5, -4.0])
        x_2 = np.array([-1.0, 3.2, 0.4, -0.8])

        self.assertEqual(pL2_dist([x_1, x_2], 0, 1), -1 * np.linalg.norm(x_1 - x_2, 2))