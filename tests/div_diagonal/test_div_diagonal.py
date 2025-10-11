"""Unit tests for diagonal_div operation.

This module contains comprehensive tests for the diagonal_div function including:
- Basic functionality tests
- Shape preservation tests
- Gradient computation tests
- Compilation tests
- Training integration tests
- Mixed precision tests
- Distributed data parallel tests
"""

import sys

sys.path.append("./")

import os
import unittest
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing import Std, start_processes
from torch.utils.data import DataLoader, TensorDataset
from diagonal import *


class TestAdd(unittest.TestCase):
    """Test suite for diagonal_div operation."""

    def test_check_cuda(self):
        """Test that CUDA is available for testing."""
        self.assertTrue(torch.cuda.is_available(), "Is cuda available for tests")

    def test_div_diagonal(self):
        """Test basic diagonal_div functionality."""
        identity_matrix = torch.eye(5, dtype=torch.bfloat16, device="cuda").add(1)
        test_tensor = diagonal_div(identity_matrix.clone(), 1.0)
        identity_matrix.diagonal(dim1=-1, dim2=-2).div_(1.0)
        div_check = (identity_matrix == test_tensor).sum().to("cpu").item()
        self.assertEqual(div_check, 25, "Is all values same")

    def test_shape(self):
        """Test that diagonal_div preserves tensor shape."""
        identity_matrix = (
            torch.eye(5, dtype=torch.bfloat16, device="cuda")
            .view(1, 1, 1, 5, 5)
            .repeat(3, 3, 3, 1, 1)
        )
        old_shape = identity_matrix.shape
        identity_matrix = diagonal_div(identity_matrix, 1.0)
        for shape_1, shape_2 in zip(old_shape, identity_matrix.shape):
            self.assertEqual(shape_1, shape_2)

    def test_grad(self):
        """Test gradient computation for diagonal_div."""
        identity_matrix_1 = torch.eye(
            5, dtype=torch.bfloat16, device="cuda", requires_grad=True
        ).add(1)
        identity_matrix_2 = torch.eye(
            5, dtype=torch.bfloat16, device="cuda", requires_grad=True
        ).add(1)
        identity_matrix_1.diagonal(dim1=-1, dim2=-2).div_(1.0)
        identity_matrix_2 = diagonal_div(identity_matrix_2, 1.0)
        identity_matrix_1.sum().backward()
        identity_matrix_2.sum().backward()
        self.assertTrue(torch.allclose(identity_matrix_1, identity_matrix_2))

    def test_compile(self):
        """Test that diagonal_div works with torch.compile."""
        identity_matrix = torch.eye(
            5, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        add_diagonal_compiled = torch.compile(diagonal_div)
        matrix = add_diagonal_compiled(identity_matrix, 1.0)

    def _create_model(self):
        """Create a simple test model that uses diagonal_div."""

        class Model(torch.nn.Module):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.linear_1 = torch.nn.Linear(10, 10)
                self.linear_2 = torch.nn.Linear(10, 1)

            def forward(self, x):
                x = F.relu(self.linear_1(x))
                x = diagonal_div(x, 1.0)
                return F.sigmoid(self.linear_2(x))

        return Model().to(torch.bfloat16)

    def test_training(self):
        """Test diagonal_div in a training loop."""
        model = self._create_model().to("cuda")
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters())
        x = torch.randn((10, 10), dtype=torch.bfloat16, device="cuda")
        one = torch.tensor(
            [0.5 for _ in range(10)], dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(dim=-1)

        for _ in range(2):
            output = model(x)
            loss = loss_fn(one, output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_training_mixed_precision(self):
        """Test diagonal_div with mixed precision training."""
        model = self._create_model().to("cuda")
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters())
        x = torch.randn((10, 10), dtype=torch.bfloat16, device="cuda")
        one = torch.tensor(
            [0.5 for _ in range(10)], dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(dim=-1)

        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(x)
                loss = loss_fn(one, output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_ddp(self):
        """Test diagonal_div with distributed data parallel."""
        error_code = subprocess.call(["python", "./tests/div_diagonal/_ddp.py"])
        if error_code:
            raise RuntimeError(f"ERROR: ddp exited with error code {error_code}")

    def test_array_tensor(self):
        """Test diagonal_div with tensor values instead of scalar."""
        identity_matrix = torch.eye(5, dtype=torch.bfloat16, device="cuda").add(1)
        test_tensor = diagonal_div(
            identity_matrix.clone(),
            torch.tensor([1.0 for _ in range(5)], dtype=torch.bfloat16, device="cuda"),
        )
        identity_matrix.diagonal(dim1=-1, dim2=-2).div_(1.0)
        div_check = (identity_matrix == test_tensor).sum().to("cpu").item()
        self.assertEqual(div_check, 25, "Is all values same")

    def test_grad_value(self):
        """Test gradient computation for value parameter in diagonal_div."""
        identity_matrix_1 = torch.eye(5, dtype=torch.bfloat16, device="cuda")
        identity_matrix_2 = torch.eye(5, dtype=torch.bfloat16, device="cuda")
        ones_1 = torch.ones(
            (5,), device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        ones_2 = torch.ones(
            (5,), device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        identity_matrix_1.diagonal(dim1=-1, dim2=-2).div_(ones_1)
        identity_matrix_2 = diagonal_div(identity_matrix_2, ones_2)
        identity_matrix_1.sum().backward()
        identity_matrix_2.sum().backward()
        self.assertTrue(torch.allclose(ones_1, ones_2))

    def test_edge_case_scalar_values(self):
        """Test diagonal_div with edge case scalar values: 2, -2, -1.

        Mathematical properties tested:
        - Division by 2: halves diagonal elements (1/2 = 0.5)
        - Division by -2: halves and negates (1/(-2) = -0.5)
        - Division by -1: negates diagonal elements (1/(-1) = -1)

        Note: Division by 0 is undefined and not tested here.
        """
        base_matrix = torch.eye(5, dtype=torch.bfloat16, device="cuda")

        # Test with value 2: diagonal becomes 0.5
        result = diagonal_div(base_matrix.clone(), 2.0)
        expected_diag = torch.full((5,), 0.5, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(torch.allclose(result.diagonal(), expected_diag),
                       f"Failed for scalar 2: got {result.diagonal()}, expected {expected_diag}")

        # Test with value -2: diagonal becomes -0.5
        result = diagonal_div(base_matrix.clone(), -2.0)
        expected_diag = torch.full((5,), -0.5, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(torch.allclose(result.diagonal(), expected_diag),
                       f"Failed for scalar -2: got {result.diagonal()}, expected {expected_diag}")

        # Test with value -1: diagonal becomes -1
        result = diagonal_div(base_matrix.clone(), -1.0)
        expected_diag = torch.full((5,), -1.0, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(torch.allclose(result.diagonal(), expected_diag),
                       f"Failed for scalar -1: got {result.diagonal()}, expected {expected_diag}")

    def test_edge_case_array_values(self):
        """Test diagonal_div with edge case array values: [2, 1, -2, -1, 0.5].

        Tests element-wise division with different edge cases on each diagonal element.
        Note: Avoiding division by zero.
        """
        base_matrix = torch.eye(5, dtype=torch.bfloat16, device="cuda")
        divisors = torch.tensor([2.0, 1.0, -2.0, -1.0, 0.5], dtype=torch.bfloat16, device="cuda")

        result = diagonal_div(base_matrix.clone(), divisors)

        # Expected: [1/2, 1/1, 1/(-2), 1/(-1), 1/0.5] = [0.5, 1.0, -0.5, -1.0, 2.0]
        expected_diag = torch.tensor([0.5, 1.0, -0.5, -1.0, 2.0], dtype=torch.bfloat16, device="cuda")

        self.assertTrue(torch.allclose(result.diagonal(), expected_diag),
                       f"Diagonal mismatch: got {result.diagonal()}, expected {expected_diag}")

        # Verify off-diagonal elements remain zero
        mask = ~torch.eye(5, dtype=torch.bool, device="cuda")
        self.assertTrue(torch.allclose(result[mask], torch.zeros(20, device="cuda", dtype=torch.bfloat16)),
                       "Off-diagonal elements should remain zero")


if __name__ == "__main__":
    unittest.main()
