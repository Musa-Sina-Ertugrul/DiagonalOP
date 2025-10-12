"""
PyTorch Custom Operator Registration for torch.compile compatibility

This module registers custom CUDA diagonal operations with PyTorch's operator
registry, enabling torch.compile (including max-autotune mode) to properly
trace and optimize these operations.

Mathematical Background:
-----------------------
For a square matrix A ∈ ℝⁿˣⁿ, the diagonal operator D extracts elements:
    D(A) = {a₁₁, a₂₂, ..., aₙₙ}

Our operations support element-wise transformations on diagonals:
    - add_diagonal: D(A) ← D(A) + v
    - sub_diagonal: D(A) ← D(A) - v
    - mul_diagonal: D(A) ← D(A) ⊙ v
    - div_diagonal: D(A) ← D(A) ⊘ v
    - sum_diagonal: s = Σᵢ aᵢᵢ

Where v can be:
    - Scalar: v ∈ ℝ (broadcast to all diagonal elements)
    - Vector: v ∈ ℝⁿ (element-wise operation)
"""

import torch
import diagonal_cuda

# Define the library namespace for our custom operations
# This creates a namespace "diagonal_ops" that torch.compile can understand
torch.library.define(
    "diagonal_ops::add_diagonal",
    "(Tensor input, Tensor value) -> Tensor",
    tags=torch.Tag.pt2_compliant_tag,
)

torch.library.define(
    "diagonal_ops::sub_diagonal",
    "(Tensor input, Tensor value) -> Tensor",
    tags=torch.Tag.pt2_compliant_tag,
)

torch.library.define(
    "diagonal_ops::mul_diagonal",
    "(Tensor input, Tensor value) -> Tensor",
    tags=torch.Tag.pt2_compliant_tag,
)

torch.library.define(
    "diagonal_ops::div_diagonal",
    "(Tensor input, Tensor value) -> Tensor",
    tags=torch.Tag.pt2_compliant_tag,
)

torch.library.define(
    "diagonal_ops::sum_diagonal",
    "(Tensor input) -> Tensor",
    tags=torch.Tag.pt2_compliant_tag,
)


# Implement the actual operations by delegating to CUDA kernels
@torch.library.impl("diagonal_ops::add_diagonal", "cuda")
def add_diagonal_impl(input: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Add value to diagonal elements of input tensor.

    Forward operation: out[i,i] = input[i,i] + value[i] (or + value if scalar)

    Args:
        input: Square matrices [..., n, n]
        value: Scalar or vector of length n

    Returns:
        Modified tensor with updated diagonal
    """
    return diagonal_cuda.add_diagonal(input, value)


@torch.library.impl("diagonal_ops::sub_diagonal", "cuda")
def sub_diagonal_impl(input: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Subtract value from diagonal elements of input tensor.

    Forward operation: out[i,i] = input[i,i] - value[i] (or - value if scalar)

    Args:
        input: Square matrices [..., n, n]
        value: Scalar or vector of length n

    Returns:
        Modified tensor with updated diagonal
    """
    return diagonal_cuda.sub_diagonal(input, value)


@torch.library.impl("diagonal_ops::mul_diagonal", "cuda")
def mul_diagonal_impl(input: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Multiply diagonal elements by value.

    Forward operation: out[i,i] = input[i,i] × value[i] (or × value if scalar)

    This is the problematic operation causing illegal memory access with torch.compile.

    Args:
        input: Square matrices [..., n, n]
        value: Scalar or vector of length n

    Returns:
        Modified tensor with updated diagonal
    """
    return diagonal_cuda.mul_diagonal(input, value)


@torch.library.impl("diagonal_ops::div_diagonal", "cuda")
def div_diagonal_impl(input: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Divide diagonal elements by value.

    Forward operation: out[i,i] = input[i,i] / value[i] (or / value if scalar)

    Args:
        input: Square matrices [..., n, n]
        value: Scalar or vector of length n

    Returns:
        Modified tensor with updated diagonal
    """
    return diagonal_cuda.div_diagonal(input, value)


@torch.library.impl("diagonal_ops::sum_diagonal", "cuda")
def sum_diagonal_impl(input: torch.Tensor) -> torch.Tensor:
    """
    Sum all diagonal elements.

    Forward operation: s = Σᵢ input[i,i]

    Args:
        input: Square matrices [..., n, n]

    Returns:
        Tensor containing sum of diagonal elements for each matrix in batch
    """
    return diagonal_cuda.sum_diagonal(input)


# Register fake implementations for meta device (shape inference)
# This allows torch.compile to perform shape analysis without running CUDA code
@torch.library.register_fake("diagonal_ops::add_diagonal")
def add_diagonal_abstract(input: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Shape inference for add_diagonal - output shape same as input"""
    return torch.empty_like(input)


@torch.library.register_fake("diagonal_ops::sub_diagonal")
def sub_diagonal_abstract(input: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Shape inference for sub_diagonal - output shape same as input"""
    return torch.empty_like(input)


@torch.library.register_fake("diagonal_ops::mul_diagonal")
def mul_diagonal_abstract(input: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Shape inference for mul_diagonal - output shape same as input"""
    return torch.empty_like(input)


@torch.library.register_fake("diagonal_ops::div_diagonal")
def div_diagonal_abstract(input: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Shape inference for div_diagonal - output shape same as input"""
    return torch.empty_like(input)


@torch.library.register_fake("diagonal_ops::sum_diagonal")
def sum_diagonal_abstract(input: torch.Tensor) -> torch.Tensor:
    """
    Shape inference for sum_diagonal.

    Output shape: input.shape[:-2] (removes last two matrix dimensions)
    Example: [B, C, H, W, n, n] → [B, C, H, W]
    """
    return torch.empty(input.shape[:-2], dtype=input.dtype, device=input.device)


# Setup backward pass using autograd.Function setup
def setup_autograd(op_name: str):
    """
    Register backward pass for custom operator.

    This uses torch.library.register_autograd to tell PyTorch how to
    compute gradients for our custom operations.
    """

    def add_diagonal_backward(ctx, grad_output):
        """
        Backward for add_diagonal:
        ∂L/∂input = ∂L/∂output (identity)
        ∂L/∂value = Σ_{batches} ∂L/∂output[i,i]
        """
        value = ctx.saved_tensors[0] if ctx.saved_tensors else None
        grad_input = grad_output

        grad_value = None
        if ctx.needs_input_grad[1]:
            # Sum gradients from all diagonal elements
            from functools import reduce
            from operator import mul
            grad_value = value.mul(reduce(mul, list(grad_output.shape[:-1])))
            if value.dim() == 0:
                grad_value = grad_value.sum()

        return grad_input, grad_value

    def mul_diagonal_backward(ctx, grad_output):
        """
        Backward for mul_diagonal:
        ∂L/∂input[i,j] = ∂L/∂output[i,j] × value[i] if i==j else ∂L/∂output[i,j]
        ∂L/∂value[i] = Σ_{batches} ∂L/∂output[i,i] × input[i,i]
        """
        input_tensor, value = ctx.saved_tensors
        grad_output = torch.ops.diagonal_ops.mul_diagonal(grad_output, value)

        grad_value = None
        if ctx.needs_input_grad[1]:
            grad_value = grad_output.mul(input_tensor)
            if value.dim() == 0:
                grad_value = torch.ops.diagonal_ops.sum_diagonal(grad_value)

        return grad_output, grad_value

    # Register the backward functions
    # Note: This requires torch >= 2.0
    # For now, we'll handle backward in the autograd.Function wrapper
    pass


# Export the registered operations
# These can be called as torch.ops.diagonal_ops.add_diagonal(...)
__all__ = [
    "add_diagonal_impl",
    "sub_diagonal_impl",
    "mul_diagonal_impl",
    "div_diagonal_impl",
    "sum_diagonal_impl",
]
