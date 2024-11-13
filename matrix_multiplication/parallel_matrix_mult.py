import numpy as np
import multiprocessing as mp

def block_multiply(A_block, B_block):
    """
    Multiplies two individual blocks of matrices A and B.

    Args:
        A_block: A block of the first matrix (2D NumPy array).
        B_block: A block of the second matrix (2D NumPy array).

    Returns:
        The resulting product block (2D NumPy array).
    """
    return np.dot(A_block, B_block)

def compute_block(i, j, A, B, block_size, num_blocks_common):
    """
    Computes a block for the result matrix C.

    Args:
        i, j: Indices of the block in the result matrix C.
        A, B: The matrices to be multiplied.
        block_size: The size of each block.
        num_blocks_common: The number of blocks in the common dimension.

    Returns:
        The resulting block for position (i, j) in the matrix C.
    """
    C_block = np.zeros((block_size, block_size))
    for k in range(num_blocks_common):
        A_block = A[i * block_size:(i + 1) * block_size, k * block_size:(k + 1) * block_size]
        B_block = B[k * block_size:(k + 1) * block_size, j * block_size:(j + 1) * block_size]
        C_block += block_multiply(A_block, B_block)
    return (i, j, C_block)

def parallel_matrix_mult(A, B, block_size):
    """
    Performs parallel matrix multiplication using a block-based approach.

    Args:
        A: The first matrix (2D NumPy array).
        B: The second matrix (2D NumPy array).
        block_size: The size of the blocks for matrix partitioning (integer).

    Returns:
        The resulting product matrix (2D NumPy array).
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible matrix dimensions for multiplication")

    num_blocks_A_row = (A.shape[0] + block_size - 1) // block_size
    num_blocks_B_col = (B.shape[1] + block_size - 1) // block_size
    num_blocks_common = (A.shape[1] + block_size - 1) // block_size

    C_blocks = [[None] * num_blocks_B_col for _ in range(num_blocks_A_row)]

    # Use multiprocessing pool to compute each block in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = [pool.apply_async(compute_block, args=(i, j, A, B, block_size, num_blocks_common))
                   for i in range(num_blocks_A_row) for j in range(num_blocks_B_col)]
        
        # Collect the results and place them in C_blocks
        for result in results:
            i, j, C_block = result.get()
            C_blocks[i][j] = C_block

    # Reassemble the final matrix from the blocks
    C = np.vstack([np.hstack(row) for row in C_blocks])

    return C
