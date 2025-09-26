"""Diagonal matrix operations with automatic differentiation support.

This module provides PyTorch autograd functions for efficient diagonal matrix operations.
All operations are performed using custom kernels for better performance.
"""

from functools import reduce
from operator import mul
import torch
from .load import _diagonal_kernel


class AddDiagonal(torch.autograd.Function):
    """Autograd function for adding a value to the diagonal elements of a matrix."""

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, value: torch.Tensor | torch.NumberType | float | int
    ):
        """Forward pass: adds value to diagonal elements.

        Args:
            ctx: Context object for storing information for backward pass
            input: Input tensor (must be square matrix or batch of square matrices)
            value: Value to add to diagonal elements (scalar or tensor)

        Returns:
            Tensor with value added to diagonal elements
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=input.device, dtype=input.dtype)
        ctx.value = value
        return _diagonal_kernel.add_diagonal(input, value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_value = None
        if ctx.needs_input_grad[1]:
            grad_value = ctx.value.mul(reduce(mul, list(grad_output.shape[:-1])))
            if ctx.value.dim() == 0:
                grad_value = grad_value.sum()
        return grad_output, grad_value


def diagonal_add(input, value):
    """Add a value to the diagonal elements of a matrix.

    Args:
        input (torch.Tensor): Input tensor (square matrix or batch of square matrices)
        value (torch.Tensor|float|int): Value to add to diagonal elements

    Returns:
        torch.Tensor: Tensor with value added to diagonal elements

    Example:
        >>> x = torch.randn(3, 3)
        >>> result = diagonal_add(x, 2.0)
    """
    return AddDiagonal.apply(input, value)


class SubDiagonal(torch.autograd.Function):
    """Autograd function for subtracting a value from the diagonal elements of a matrix."""

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, value: torch.Tensor | torch.NumberType | float | int
    ):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=input.device, dtype=input.dtype)
        ctx.value = value
        return _diagonal_kernel.sub_diagonal(input, value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_value = None
        if ctx.needs_input_grad[1]:
            grad_value = ctx.value.mul(reduce(mul, list(grad_output.shape[:-1])) * -1)
            if ctx.value.dim() == 0:
                grad_value = grad_value.sum()
        return grad_output, grad_value


def diagonal_sub(input, value):
    """Subtract a value from the diagonal elements of a matrix.

    Args:
        input (torch.Tensor): Input tensor (square matrix or batch of square matrices)
        value (torch.Tensor|float|int): Value to subtract from diagonal elements

    Returns:
        torch.Tensor: Tensor with value subtracted from diagonal elements

    Example:
        >>> x = torch.randn(3, 3)
        >>> result = diagonal_sub(x, 1.5)
    """
    return SubDiagonal.apply(input, value)


class MulDiagonal(torch.autograd.Function):
    """Autograd function for multiplying diagonal elements of a matrix by a value."""

    @staticmethod
    def forward(ctx, input, value: torch.Tensor):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=input.device, dtype=input.dtype)
        ctx.value = value
        ctx.input = input
        return _diagonal_kernel.mul_diagonal(input, value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_value = None
        grad_output = _diagonal_kernel.mul_diagonal(grad_output, ctx.value)
        if ctx.needs_input_grad[1]:
            grad_value = grad_output.mul(ctx.input)
            if ctx.value.dim() == 0:
                grad_value = _diagonal_kernel.sum_diagonal(grad_value)
        return grad_output, grad_value


def diagonal_mul(input, value):
    """Multiply the diagonal elements of a matrix by a value.

    Args:
        input (torch.Tensor): Input tensor (square matrix or batch of square matrices)
        value (torch.Tensor|float|int): Value to multiply diagonal elements by

    Returns:
        torch.Tensor: Tensor with diagonal elements multiplied by value

    Example:
        >>> x = torch.randn(3, 3)
        >>> result = diagonal_mul(x, 0.5)
    """
    return MulDiagonal.apply(input, value)


class DivDiagonal(torch.autograd.Function):
    """Autograd function for dividing diagonal elements of a matrix by a value."""

    @staticmethod
    def forward(ctx, input, value: torch.Tensor):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=input.device, dtype=input.dtype)
        ctx.value = value
        ctx.input = input
        return _diagonal_kernel.mul_diagonal(input, value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_value = None
        grad_output = _diagonal_kernel.div_diagonal(grad_output, ctx.value)
        if ctx.needs_input_grad[1]:
            grad_value = _diagonal_kernel.div_diagonal(
                grad_output.mul(ctx.input), ctx.value
            ).sub(ctx.input)
            grad_value = _diagonal_kernel.div_diagonal(grad_value, ctx.value.pow(2))
            if ctx.value.dim() == 0:
                grad_value = _diagonal_kernel.sum_diagonal(grad_value)
        return grad_output, grad_value


def diagonal_div(input, value):
    """Divide the diagonal elements of a matrix by a value.

    Args:
        input (torch.Tensor): Input tensor (square matrix or batch of square matrices)
        value (torch.Tensor|float|int): Value to divide diagonal elements by

    Returns:
        torch.Tensor: Tensor with diagonal elements divided by value

    Example:
        >>> x = torch.randn(3, 3)
        >>> result = diagonal_div(x, 2.0)
    """
    return DivDiagonal.apply(input, value)


class SumDiagonal(torch.autograd.Function):
    """Autograd function for summing the diagonal elements of a matrix."""

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        ctx.input_shape = input.shape

        return _diagonal_kernel.sum_diagonal(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        identity_matrix = torch.eye(
            ctx.input_shape[-1], device=grad_output.device, dtype=grad_output.dtype
        )
        repeat_dims = [1 for _ in range(len(ctx.input_shape[:-2]))]
        if len(repeat_dims) == 0:
            grad_output = (
                grad_output.unsqueeze(dim=-1)
                .unsqueeze(dim=-1)
                .repeat(ctx.input_shape[-1], ctx.input_shape[-1])
                .matmul(identity_matrix)
            )
        else:
            grad_output = (
                grad_output.unsqueeze(dim=-1)
                .unsqueeze(dim=-1)
                .repeat(*repeat_dims, ctx.input_shape[-1], ctx.input_shape[-1])
                .matmul(identity_matrix)
            )
        return grad_output


def diagonal_sum(input):
    """Sum the diagonal elements of a matrix.

    Args:
        input (torch.Tensor): Input tensor (square matrix or batch of square matrices)

    Returns:
        torch.Tensor: Sum of diagonal elements

    Example:
        >>> x = torch.randn(3, 3)
        >>> result = diagonal_sum(x)
    """
    return SumDiagonal.apply(input)
