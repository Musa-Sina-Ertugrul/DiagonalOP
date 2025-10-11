"""VRAM benchmark runner for diagonal matrix operations.

This module benchmarks VRAM usage for all diagonal operations against native
PyTorch implementations to measure memory efficiency of custom CUDA kernels.

Run with: python -m vram_benchmark
"""

import argparse
import gc
from typing import Callable, Tuple
import torch
from diagonal import diagonal_add, diagonal_sub, diagonal_mul, diagonal_div, diagonal_sum


def get_memory_stats() -> Tuple[float, float, float]:
    """Get current CUDA memory statistics.

    Returns:
        Tuple of (allocated_mb, reserved_mb, max_allocated_mb)
    """
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return allocated, reserved, max_allocated


def reset_memory_stats():
    """Reset CUDA memory statistics and clear cache."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def benchmark_vram_operation(
    operation: Callable,
    input_tensor: torch.Tensor,
    value: float | torch.Tensor = None,
    warmup_runs: int = 3,
) -> Tuple[float, float]:
    """Benchmark VRAM usage for a single operation.

    Args:
        operation: The operation function to benchmark
        input_tensor: Input tensor for the operation
        value: Value parameter for the operation (if needed)
        warmup_runs: Number of warmup iterations

    Returns:
        Tuple of (peak_memory_mb, allocated_memory_mb)
    """
    # Warmup
    for _ in range(warmup_runs):
        reset_memory_stats()
        if value is not None:
            _ = operation(input_tensor, value)
        else:
            _ = operation(input_tensor)
        torch.cuda.synchronize()

    # Actual benchmark
    reset_memory_stats()

    # Record memory before operation
    mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

    if value is not None:
        result = operation(input_tensor, value)
    else:
        result = operation(input_tensor)

    torch.cuda.synchronize()

    # Record peak memory during operation
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)

    # Memory used by the operation
    memory_used = peak_memory - mem_before

    # Clean up
    del result

    return peak_memory, memory_used


def pytorch_diagonal_add(input_tensor: torch.Tensor, value: float) -> torch.Tensor:
    """Native PyTorch implementation of diagonal_add."""
    result = input_tensor.clone()
    result.diagonal(dim1=-2, dim2=-1).add_(value)
    return result


def pytorch_diagonal_sub(input_tensor: torch.Tensor, value: float) -> torch.Tensor:
    """Native PyTorch implementation of diagonal_sub."""
    result = input_tensor.clone()
    result.diagonal(dim1=-2, dim2=-1).sub_(value)
    return result


def pytorch_diagonal_mul(input_tensor: torch.Tensor, value: float) -> torch.Tensor:
    """Native PyTorch implementation of diagonal_mul."""
    result = input_tensor.clone()
    result.diagonal(dim1=-2, dim2=-1).mul_(value)
    return result


def pytorch_diagonal_div(input_tensor: torch.Tensor, value: float) -> torch.Tensor:
    """Native PyTorch implementation of diagonal_div."""
    result = input_tensor.clone()
    result.diagonal(dim1=-2, dim2=-1).div_(value)
    return result


def pytorch_diagonal_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """Native PyTorch implementation of diagonal_sum."""
    return input_tensor.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def run_vram_benchmarks(
    batch_size: int,
    matrix_size: int,
    dtype: torch.dtype,
    warmup_runs: int,
):
    """Run all VRAM benchmarks and display results.

    Args:
        batch_size: Batch size for matrices
        matrix_size: Size of square matrices
        dtype: Data type for tensors
        warmup_runs: Number of warmup iterations
    """
    print(f"\n{'='*90}")
    print(f"Diagonal Operations VRAM Benchmark")
    print(f"{'='*90}")
    print(f"Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Matrix Size: {matrix_size}x{matrix_size}")
    print(f"  Data Type: {dtype}")
    print(f"  Warmup Runs: {warmup_runs}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")

    # Calculate input tensor size
    if batch_size > 1:
        input_shape = (batch_size, matrix_size, matrix_size)
    else:
        input_shape = (matrix_size, matrix_size)

    element_count = batch_size * matrix_size * matrix_size if batch_size > 1 else matrix_size * matrix_size
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    input_size_mb = (element_count * bytes_per_element) / (1024 ** 2)

    print(f"  Input Tensor Size: {input_size_mb:.2f} MB")
    print(f"{'='*90}\n")

    # Create test data
    reset_memory_stats()
    input_tensor = torch.randn(*input_shape, device="cuda", dtype=dtype)
    value = 2.5

    operations = [
        ("diagonal_add", diagonal_add, pytorch_diagonal_add, value),
        ("diagonal_sub", diagonal_sub, pytorch_diagonal_sub, value),
        ("diagonal_mul", diagonal_mul, pytorch_diagonal_mul, value),
        ("diagonal_div", diagonal_div, pytorch_diagonal_div, value),
        ("diagonal_sum", diagonal_sum, pytorch_diagonal_sum, None),
    ]

    print(f"{'Operation':<20} {'Custom Peak (MB)':<20} {'PyTorch Peak (MB)':<20} {'Memory Saved':<15}")
    print(f"{'-'*90}")

    for op_name, custom_op, pytorch_op, op_value in operations:
        # Benchmark custom operation VRAM
        custom_peak, custom_used = benchmark_vram_operation(
            custom_op, input_tensor, op_value, warmup_runs
        )

        # Benchmark PyTorch operation VRAM
        pytorch_peak, pytorch_used = benchmark_vram_operation(
            pytorch_op, input_tensor, op_value, warmup_runs
        )

        memory_saved = pytorch_peak - custom_peak
        memory_saved_pct = (memory_saved / pytorch_peak * 100) if pytorch_peak > 0 else 0

        print(
            f"{op_name:<20} {custom_peak:>10.2f} MB        "
            f"{pytorch_peak:>10.2f} MB        "
            f"{memory_saved:>6.2f} MB ({memory_saved_pct:>5.1f}%)"
        )

    print(f"{'-'*90}\n")


def run_gradient_vram_benchmarks(
    batch_size: int,
    matrix_size: int,
    dtype: torch.dtype,
    warmup_runs: int,
):
    """Run gradient computation VRAM benchmarks.

    Args:
        batch_size: Batch size for matrices
        matrix_size: Size of square matrices
        dtype: Data type for tensors
        warmup_runs: Number of warmup iterations
    """
    print(f"\n{'='*90}")
    print(f"Gradient Computation VRAM Benchmark")
    print(f"{'='*90}")
    print(f"Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Matrix Size: {matrix_size}x{matrix_size}")
    print(f"  Data Type: {dtype}")
    print(f"  Warmup Runs: {warmup_runs}")
    print(f"{'='*90}\n")

    def forward_backward_custom(input_tensor, value):
        """Forward and backward pass with custom operation."""
        result = diagonal_add(input_tensor, value)
        loss = result.sum()
        loss.backward()
        return loss

    def forward_backward_pytorch(input_tensor, value):
        """Forward and backward pass with PyTorch operation."""
        result = pytorch_diagonal_add(input_tensor, value)
        loss = result.sum()
        loss.backward()
        return loss

    # Create test data shape
    if batch_size > 1:
        shape = (batch_size, matrix_size, matrix_size)
    else:
        shape = (matrix_size, matrix_size)

    value_tensor = torch.tensor(2.5, device="cuda", dtype=dtype, requires_grad=True)

    # Warmup custom
    for _ in range(warmup_runs):
        reset_memory_stats()
        input_tensor = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
        _ = forward_backward_custom(input_tensor, value_tensor)
        del input_tensor

    # Benchmark custom gradient VRAM
    reset_memory_stats()
    input_tensor = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
    mem_before_custom = torch.cuda.memory_allocated() / (1024 ** 2)
    _ = forward_backward_custom(input_tensor, value_tensor)
    torch.cuda.synchronize()
    custom_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    custom_used = custom_peak - mem_before_custom
    del input_tensor

    # Warmup PyTorch
    for _ in range(warmup_runs):
        reset_memory_stats()
        input_tensor = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
        _ = forward_backward_pytorch(input_tensor, value_tensor)
        del input_tensor

    # Benchmark PyTorch gradient VRAM
    reset_memory_stats()
    input_tensor = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
    mem_before_pytorch = torch.cuda.memory_allocated() / (1024 ** 2)
    _ = forward_backward_pytorch(input_tensor, value_tensor)
    torch.cuda.synchronize()
    pytorch_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    pytorch_used = pytorch_peak - mem_before_pytorch
    del input_tensor

    memory_saved = pytorch_peak - custom_peak
    memory_saved_pct = (memory_saved / pytorch_peak * 100) if pytorch_peak > 0 else 0

    print(f"{'Method':<30} {'Peak Memory (MB)':<20}")
    print(f"{'-'*90}")
    print(f"{'Custom (fwd+bwd)':<30} {custom_peak:>10.2f} MB")
    print(f"{'PyTorch (fwd+bwd)':<30} {pytorch_peak:>10.2f} MB")
    print(f"{'Memory Saved':<30} {memory_saved:>10.2f} MB ({memory_saved_pct:>5.1f}%)")
    print(f"{'-'*90}\n")


def main():
    """Main VRAM benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark VRAM usage for diagonal matrix operations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for matrices"
    )
    parser.add_argument(
        "--matrix-size", type=int, default=2048, help="Size of square matrices"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type for tensors",
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=3, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--no-gradient",
        action="store_true",
        help="Skip gradient benchmarks",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. VRAM benchmarks require GPU support.")
        return

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[args.dtype]

    # Run forward pass VRAM benchmarks
    run_vram_benchmarks(
        batch_size=args.batch_size,
        matrix_size=args.matrix_size,
        dtype=dtype,
        warmup_runs=args.warmup_runs,
    )

    # Run gradient VRAM benchmarks
    if not args.no_gradient:
        run_gradient_vram_benchmarks(
            batch_size=args.batch_size,
            matrix_size=args.matrix_size,
            dtype=dtype,
            warmup_runs=args.warmup_runs,
        )


if __name__ == "__main__":
    main()
