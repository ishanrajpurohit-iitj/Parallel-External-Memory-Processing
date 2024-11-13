import unittest
import numpy as np
from matrix_multiplication import matrix_multiply
from parallel_matrix_mult import parallel_matrix_mult

class TestMatrixMultiplication(unittest.TestCase):
  def test_sequential_multiplication(self):
    # Generate small test matrices
    A = np.random.rand(3, 4)
    B = np.random.rand(4, 5)

    # Calculate results using both methods
    C_sequential = matrix_multiply(A, B)
    C_expected = np.dot(A, B)  # Expected result from NumPy's dot product

    # Assert close similarity between results
    np.testing.assert_allclose(C_sequential, C_expected)

  def test_parallel_multiplication(self):
    # Generate larger test matrices
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)

    # Choose a reasonable block size
    block_size = 50

    # Calculate results using both methods (sequential for reference)
    C_sequential = matrix_multiply(A, B)
    C_parallel = parallel_matrix_mult(A, B, block_size=block_size)

    # Assert close similarity between results
    np.testing.assert_allclose(C_sequential, C_parallel)

if __name__ == '__main__':
  unittest.main()