import unittest

from gene.utils import genome_size


class TestGenomeSize(unittest.TestCase):
    def test_genome_size_paper(self):
        layer_dimensions = [128, 64, 64, 18]
        d = 3
        n = sum(layer_dimensions)

        self.assertEqual(n * (d + 1), 1096)
        self.assertEqual(genome_size({"d": d, "net": {"layer_dimensions": layer_dimensions}}), 968)