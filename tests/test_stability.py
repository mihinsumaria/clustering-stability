import unittest
import numpy as np
from stability import Stability


class TestStability(unittest.TestCase):

    def test_create_connectivity_matrix(self):
        labels1 = ['a', 'b', 'b', 'a']
        matrix1 = Stability.create_connectivity_matrix(labels1)
        test_matrix1 = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0],
                                 [1, 0, 0, 1]])
        self.assertTrue(np.array_equal(matrix1, test_matrix1))

        labels2 = [1, 0, 1, 0]
        matrix2 = Stability.create_connectivity_matrix(labels2)
        test_matrix2 = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0],
                                 [0, 1, 0, 1]])
        self.assertTrue(np.array_equal(matrix2, test_matrix2))

        matrix3 = Stability.create_connectivity_matrix(labels1, n_jobs=2)
        self.assertTrue(np.array_equal(matrix3, test_matrix1))
        self.assertTrue(np.array_equal(matrix3, matrix1))

        matrix4 = Stability.create_connectivity_matrix(labels2, n_jobs=2)
        self.assertTrue(np.array_equal(matrix4, test_matrix2))
        self.assertTrue(np.array_equal(matrix3, matrix1))
