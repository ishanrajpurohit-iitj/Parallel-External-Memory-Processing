# Main script to run the project
from prime_number_generation.parallel_prime_gen import parallel_prime_gen
from matrix_multiplication.parallel_matrix_mult import parallel_matrix_mult
import numpy as np

def main():
    # Parameters for range and matrices
    range_start, range_end = 1, 10
    matrix_size = 1000
    block_size = 100

    # Prime Generation
    primes = parallel_prime_gen(range_start, range_end)
    
    # Matrix Multiplication (simplified example)
    A, B = np.random.rand(matrix_size, matrix_size), np.random.rand(matrix_size, matrix_size)
    C = parallel_matrix_mult(A, B, block_size)
    print(C)

if __name__ == "__main__":
    main()
