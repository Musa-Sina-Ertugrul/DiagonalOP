"""Diagonal matrix operations library with PyTorch autograd support.

This package provides efficient diagonal matrix operations using custom CUDA kernels
with automatic differentiation support through PyTorch's autograd system.

Available functions:
    - diagonal_add: Add value to diagonal elements
    - diagonal_sub: Subtract value from diagonal elements
    - diagonal_mul: Multiply diagonal elements by value
    - diagonal_div: Divide diagonal elements by value
    - diagonal_sum: Sum diagonal elements

Requirements:
    - PyTorch
    - CUDA toolkit (for GPU acceleration)
"""

from .diagonal_functions import (
    diagonal_add,
    diagonal_mul,
    diagonal_div,
    diagonal_sub,
    diagonal_sum,
)
