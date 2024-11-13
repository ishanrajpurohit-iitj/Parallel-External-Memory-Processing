import numpy as np
import time
from prime_number_generation.parallel_prime_gen import parallel_prime_gen
from matrix_multiplication.parallel_matrix_mult import parallel_matrix_mult

def normal_matrix_multiply(A, B):
    """Perform standard matrix multiplication"""
    return np.dot(A, B)

def main():
    # Parameters for range and matrices
    range_start, range_end = 1, 10
    matrix_size = 1000
    block_size = 100

    # Prime Generation (keeping this as is)
    primes = parallel_prime_gen(range_start, range_end)
    
    # Generate random matrices
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)

    # Normal Matrix Multiplication
    start_time = time.time()
    C_normal = normal_matrix_multiply(A, B)
    normal_time = time.time() - start_time
    print(f"Normal Matrix Multiplication Time: {normal_time:.4f} seconds")

    # Parallel Matrix Multiplication
    start_time = time.time()
    C_parallel = parallel_matrix_mult(A, B, block_size)
    parallel_time = time.time() - start_time
    print(f"Parallel Matrix Multiplication Time: {parallel_time:.4f} seconds")

    # Optional: Verify results are approximately the same
    print("Results are equivalent:", np.allclose(C_normal, C_parallel))

    # Calculate speedup
    speedup = normal_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()