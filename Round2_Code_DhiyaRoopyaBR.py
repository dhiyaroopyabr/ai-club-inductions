"""
Memory-Efficient Exact Attention using Chunked Computation

Description:
- Computes exact attention without storing the full n×n matrix
- Uses chunked processing to reduce memory usage
- Uses online softmax for correct normalization
- Memory complexity reduced from O(n^2) to O(n * chunk_size)

Backward Pass Strategy:
- Uses gradient checkpointing
- Intermediate activations are NOT stored during forward pass
- During backward pass, required values are recomputed using the same chunked approach
- This reduces memory significantly at the cost of slight recomputation
"""

import numpy as np


def chunked_attention(X, chunk_size):
    """
    Forward pass: Computes attention outputs using chunked processing.

    Args:
        X: Input matrix of shape (n, d)
        chunk_size: Size of chunks for processing

    Returns:
        Y: Output matrix of shape (n, d)
    """
    n, d = X.shape
    Y = np.zeros((n, d))

    # Iterate over each query vector
    for i in range(n):
        xi = X[i]

        # Online softmax variables
        max_score = -np.inf
        sum_exp = 0.0
        weighted_sum = np.zeros(d)

        # Process keys in chunks (avoid n×n matrix)
        for j_start in range(0, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            X_chunk = X[j_start:j_end]

            # Compute similarity scores
            scores = X_chunk @ xi

            # Update running maximum (for numerical stability)
            new_max = max(max_score, np.max(scores))

            # Update exponential sum (online softmax)
            sum_exp = np.exp(max_score - new_max) * sum_exp + \
                      np.sum(np.exp(scores - new_max))

            # Update weighted sum of values
            weighted_sum = np.exp(max_score - new_max) * weighted_sum + \
                           (np.exp(scores - new_max)[:, None] * X_chunk).sum(axis=0)

            max_score = new_max

        # Final normalized output
        Y[i] = weighted_sum / sum_exp

    return Y


# ---------------------------------------------------
# Backward Pass (Explanation Only — As Required)
# ---------------------------------------------------
"""
Backward Pass Strategy:

- Standard attention requires storing intermediate attention scores (O(n^2) memory)

- To avoid this, we use Gradient Checkpointing:
    • Do NOT store intermediate values during forward pass
    • During backward pass, recompute required values using the same chunked attention logic

- This ensures:
    Memory complexity → O(n * chunk_size)
    instead of → O(n^2)

- Tradeoff:
    Slight increase in computation due to recomputation
"""


# -------------------------------
# Working Example (REQUIRED)
# -------------------------------
if __name__ == "__main__":
    # Problem requirement: n=128, d=32
    n, d = 128, 32

    # Generate random input
    X = np.random.rand(n, d)

    # Define chunk size
    chunk_size = 32

    # Run forward pass
    Y = chunked_attention(X, chunk_size)

    # Verify output
    print("Input shape:", X.shape)
    print("Output shape:", Y.shape)