import unittest

from stafe.utilax import backend

class TestJax(unittest.TestCase):
    def test_GPU_active(self):
        self.assertEqual(backend(), "gpu")

