"""Benchmark runner for diagonal matrix operations.

This module benchmarks all diagonal operations against native PyTorch implementations
to measure performance improvements from custom CUDA kernels.

Run with: python -m benchmark
"""

import argparse
import time
from typing import Callable, Tuple
import torch
from diagonal import diagonal_add, diagonal_sub, diagonal_mul, diagonal_div, diagonal_sum


def benchmark_operation(
    operation: Callable,
    input_tensor: torch.Tensor,
    value: float | torch.Tensor = None,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> Tuple[float, float]:
    """Benchmark a single operation.

    Args:
        operation: The operation function to benchmark
        input_tensor: Input tensor for the operation
        value: Value parameter for the operation (if needed)
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of benchmark iterations

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(warmup_runs):
        if value is not None:
            _ = operation(input_tensor, value)
        else:
            _ = operation(input_tensor)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        if value is not None:
            result = operation(input_tensor, value)
        else:
            result = operation(input_tensor)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)

    times_tensor = torch.tensor(times)
    mean_time = times_tensor.mean().item()
    std_time = times_tensor.std().item()

    return mean_time, std_time


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


def run_benchmarks(
    batch_size: int,
    matrix_size: int,
    dtype: torch.dtype,
    warmup_runs: int,
    benchmark_runs: int,
):
    """Run all benchmarks and display results.

    Args:
        batch_size: Batch size for matrices
        matrix_size: Size of square matrices
        dtype: Data type for tensors
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of benchmark iterations
    """
    print(f"\n{'='*80}")
    print(f"Diagonal Operations Benchmark")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Matrix Size: {matrix_size}x{matrix_size}")
    print(f"  Data Type: {dtype}")
    print(f"  Warmup Runs: {warmup_runs}")
    print(f"  Benchmark Runs: {benchmark_runs}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}\n")

    # Create test data
    if batch_size > 1:
        input_tensor = torch.randn(
            batch_size, matrix_size, matrix_size, device="cuda", dtype=dtype
        )
    else:
        input_tensor = torch.randn(matrix_size, matrix_size, device="cuda", dtype=dtype)

    value = 2.5

    operations = [
        ("diagonal_add", diagonal_add, pytorch_diagonal_add, value),
        ("diagonal_sub", diagonal_sub, pytorch_diagonal_sub, value),
        ("diagonal_mul", diagonal_mul, pytorch_diagonal_mul, value),
        ("diagonal_div", diagonal_div, pytorch_diagonal_div, value),
        ("diagonal_sum", diagonal_sum, pytorch_diagonal_sum, None),
    ]

    print(f"{'Operation':<20} {'Custom (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print(f"{'-'*80}")

    for op_name, custom_op, pytorch_op, op_value in operations:
        # Benchmark custom operation
        custom_mean, custom_std = benchmark_operation(
            custom_op, input_tensor, op_value, warmup_runs, benchmark_runs
        )

        # Benchmark PyTorch operation
        pytorch_mean, pytorch_std = benchmark_operation(
            pytorch_op, input_tensor, op_value, warmup_runs, benchmark_runs
        )

        speedup = pytorch_mean / custom_mean

        print(
            f"{op_name:<20} {custom_mean:>6.4f}±{custom_std:>5.4f}   "
            f"{pytorch_mean:>6.4f}±{pytorch_std:>5.4f}   {speedup:>6.2f}x"
        )

    print(f"{'-'*80}\n")


def run_gradient_benchmarks(
    batch_size: int,
    matrix_size: int,
    dtype: torch.dtype,
    warmup_runs: int,
    benchmark_runs: int,
):
    """Run gradient computation benchmarks.

    Args:
        batch_size: Batch size for matrices
        matrix_size: Size of square matrices
        dtype: Data type for tensors
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of benchmark iterations
    """
    print(f"\n{'='*80}")
    print(f"Gradient Computation Benchmark")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Matrix Size: {matrix_size}x{matrix_size}")
    print(f"  Data Type: {dtype}")
    print(f"  Warmup Runs: {warmup_runs}")
    print(f"  Benchmark Runs: {benchmark_runs}")
    print(f"{'='*80}\n")

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

    # Create test data
    if batch_size > 1:
        shape = (batch_size, matrix_size, matrix_size)
    else:
        shape = (matrix_size, matrix_size)

    value_tensor = torch.tensor(2.5, device="cuda", dtype=dtype, requires_grad=True)

    # Benchmark custom gradient
    times_custom = []
    for _ in range(warmup_runs):
        input_tensor = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
        _ = forward_backward_custom(input_tensor, value_tensor)

    for _ in range(benchmark_runs):
        input_tensor = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = forward_backward_custom(input_tensor, value_tensor)
        end_event.record()

        torch.cuda.synchronize()
        times_custom.append(start_event.elapsed_time(end_event))

    # Benchmark PyTorch gradient
    times_pytorch = []
    for _ in range(warmup_runs):
        input_tensor = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
        _ = forward_backward_pytorch(input_tensor, value_tensor)

    for _ in range(benchmark_runs):
        input_tensor = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = forward_backward_pytorch(input_tensor, value_tensor)
        end_event.record()

        torch.cuda.synchronize()
        times_pytorch.append(start_event.elapsed_time(end_event))

    custom_mean = torch.tensor(times_custom).mean().item()
    custom_std = torch.tensor(times_custom).std().item()
    pytorch_mean = torch.tensor(times_pytorch).mean().item()
    pytorch_std = torch.tensor(times_pytorch).std().item()
    speedup = pytorch_mean / custom_mean

    print(f"{'Method':<20} {'Time (ms)':<20}")
    print(f"{'-'*80}")
    print(f"{'Custom (fwd+bwd)':<20} {custom_mean:>6.4f}±{custom_std:>5.4f}")
    print(f"{'PyTorch (fwd+bwd)':<20} {pytorch_mean:>6.4f}±{pytorch_std:>5.4f}")
    print(f"{'Speedup':<20} {speedup:>6.2f}x")
    print(f"{'-'*80}\n")


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark diagonal matrix operations",
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
        "--warmup-runs", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--no-gradient",
        action="store_true",
        help="Skip gradient benchmarks",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Benchmarks require GPU support.")
        return

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[args.dtype]

    # Run forward pass benchmarks
    run_benchmarks(
        batch_size=args.batch_size,
        matrix_size=args.matrix_size,
        dtype=dtype,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )

    # Run gradient benchmarks
    if not args.no_gradient:
        run_gradient_benchmarks(
            batch_size=args.batch_size,
            matrix_size=args.matrix_size,
            dtype=dtype,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )


if __name__ == "__main__":
    main()
