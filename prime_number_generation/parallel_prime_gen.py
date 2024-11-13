# Parallel prime generation logic
import multiprocessing as mp
from sieve_of_eratosthenes import sieve_of_eratosthenes

def parallel_prime_gen(range_start, range_end):
    # Splitting work for parallel execution (simplified example)
    num_range = range(range_start, range_end)
    with mp.Pool() as pool:
        results = pool.map(sieve_of_eratosthenes, num_range)
    return results
