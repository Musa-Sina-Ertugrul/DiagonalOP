"""Diagonal matrix operations with automatic differentiation support.

This module provides PyTorch autograd functions for efficient diagonal matrix operations.
All operations are performed using custom kernels for better performance.
"""

from functools import reduce
from operator import mul
import torch
import diagonal_cuda


class _AddDiagonal(torch.autograd.Function):
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
        return diagonal_cuda.add_diagonal(input, value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: computes gradients for input and value.

        Args:
            ctx: Context object containing saved information from forward pass
            grad_output: Gradient of the loss with respect to the output

        Returns:
            Tuple of (grad_input, grad_value) gradients
        """
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
    return _AddDiagonal.apply(input, value)


class _SubDiagonal(torch.autograd.Function):
    """Autograd function for subtracting a value from the diagonal elements of a matrix."""

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, value: torch.Tensor | torch.NumberType | float | int
    ):
        """Forward pass: subtracts value from diagonal elements.

        Args:
            ctx: Context object for storing information for backward pass
            input: Input tensor (must be square matrix or batch of square matrices)
            value: Value to subtract from diagonal elements (scalar or tensor)

        Returns:
            Tensor with value subtracted from diagonal elements
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=input.device, dtype=input.dtype)
        ctx.value = value
        return diagonal_cuda.sub_diagonal(input, value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: computes gradients for input and value.

        Args:
            ctx: Context object containing saved information from forward pass
            grad_output: Gradient of the loss with respect to the output

        Returns:
            Tuple of (grad_input, grad_value) gradients
        """
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
    return _SubDiagonal.apply(input, value)


class _MulDiagonal(torch.autograd.Function):
    """Autograd function for multiplying diagonal elements of a matrix by a value."""

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, value: torch.Tensor | torch.NumberType | float | int
    ):
        """Forward pass: multiplies diagonal elements by value.

        Args:
            ctx: Context object for storing information for backward pass
            input: Input tensor (must be square matrix or batch of square matrices)
            value: Value to multiply diagonal elements by (scalar or tensor)

        Returns:
            Tensor with diagonal elements multiplied by value
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=input.device, dtype=input.dtype)
        ctx.value = value
        ctx.save_for_backward(input)
        return diagonal_cuda.mul_diagonal(input, value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: computes gradients for input and value.

        Args:
            ctx: Context object containing saved information from forward pass
            grad_output: Gradient of the loss with respect to the output

        Returns:
            Tuple of (grad_input, grad_value) gradients
        """
        grad_value = None
        grad_output = diagonal_cuda.mul_diagonal(grad_output, ctx.value)
        if ctx.needs_input_grad[1]:
            (input,) = ctx.saved_tensors
            grad_value = grad_output.mul(input)
            if ctx.value.dim() == 0:
                grad_value = diagonal_cuda.sum_diagonal(grad_value)
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
    return _MulDiagonal.apply(input, value)


class _DivDiagonal(torch.autograd.Function):
    """Autograd function for dividing diagonal elements of a matrix by a value."""

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, value: torch.Tensor | torch.NumberType | float | int
    ):
        """Forward pass: divides diagonal elements by value.

        Args:
            ctx: Context object for storing information for backward pass
            input: Input tensor (must be square matrix or batch of square matrices)
            value: Value to divide diagonal elements by (scalar or tensor)

        Returns:
            Tensor with diagonal elements divided by value
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=input.device, dtype=input.dtype)
        ctx.value = value
        ctx.save_for_backward(input)
        return diagonal_cuda.div_diagonal(input, value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: computes gradients for input and value.

        Args:
            ctx: Context object containing saved information from forward pass
            grad_output: Gradient of the loss with respect to the output

        Returns:
            Tuple of (grad_input, grad_value) gradients
        """
        grad_value = None
        grad_output = diagonal_cuda.div_diagonal(grad_output, ctx.value)
        if ctx.needs_input_grad[1]:
            (input,) = ctx.saved_tensors
            grad_value = diagonal_cuda.div_diagonal(
                grad_output.mul(input), ctx.value
            ).sub(input)
            grad_value = diagonal_cuda.div_diagonal(grad_value, ctx.value.pow(2))
            if ctx.value.dim() == 0:
                grad_value = diagonal_cuda.sum_diagonal(grad_value)
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
    return _DivDiagonal.apply(input, value)


class _SumDiagonal(torch.autograd.Function):
    """Autograd function for summing the diagonal elements of a matrix."""

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        """Forward pass: sums diagonal elements.

        Args:
            ctx: Context object for storing information for backward pass
            input: Input tensor (must be square matrix or batch of square matrices)

        Returns:
            Tensor containing the sum of diagonal elements
        """
        ctx.input_shape = input.shape

        return diagonal_cuda.sum_diagonal(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: computes gradient for input.

        Args:
            ctx: Context object containing saved information from forward pass
            grad_output: Gradient of the loss with respect to the output

        Returns:
            Gradient with respect to the input tensor
        """
        # Optimized backward: gradient is only non-zero on diagonal elements
        # Instead of creating full matrices and using matmul, directly create the result
        matrix_size = ctx.input_shape[-1]

        # Create output gradient tensor filled with zeros
        grad_input = torch.zeros(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)

        # Set diagonal elements to grad_output values
        # This is much faster than creating identity matrix and doing matmul
        if grad_output.dim() == 0:
            # Scalar gradient - single matrix case
            grad_input.diagonal(dim1=-2, dim2=-1).fill_(grad_output.item())
        else:
            # Batch gradient case
            # Reshape grad_output to broadcast correctly
            grad_output_expanded = grad_output.view(*grad_output.shape, 1).expand(*grad_output.shape, matrix_size)
            grad_input.diagonal(dim1=-2, dim2=-1).copy_(grad_output_expanded)

        return grad_input


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
    return _SumDiagonal.apply(input)
