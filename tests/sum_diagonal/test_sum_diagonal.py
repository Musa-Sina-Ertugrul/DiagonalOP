"""Unit tests for diagonal_sum operation.

This module contains comprehensive tests for the diagonal_sum function including:
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


class Testsub(unittest.TestCase):
    """Test suite for diagonal_sum operation."""

    def test_check_cuda(self):
        """Test that CUDA is available for testing."""
        self.assertTrue(torch.cuda.is_available(), "Is cuda available for tests")

    def test_sub_diagonal(self):
        """Test basic diagonal_sum functionality."""
        identity_matrix = torch.eye(5, dtype=torch.bfloat16, device="cuda")
        test_tensor = diagonal_sum(identity_matrix.clone())
        identity_matrix = identity_matrix.sum(dim=-1).sum(dim=-1)
        sum_check = (identity_matrix == test_tensor).sum().to("cpu").item()
        self.assertEqual(sum_check, 1, "Is all values same")

    def test_shape(self):
        """Test that diagonal_sum reduces tensor dimensions correctly."""
        identity_matrix = (
            torch.eye(5, dtype=torch.bfloat16, device="cuda")
            .view(1, 1, 1, 5, 5)
            .repeat(3, 3, 3, 1, 1)
        )
        old_shape = identity_matrix.shape
        identity_matrix = diagonal_sum(identity_matrix).squeeze()
        for shape_1, shape_2 in zip(old_shape, identity_matrix.shape):
            self.assertEqual(shape_1, shape_2)

    def test_grad(self):
        """Test gradient computation for diagonal_sum."""
        identity_matrix_1 = torch.eye(
            5, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        identity_matrix_2 = torch.eye(
            5, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        identity_matrix_1 = identity_matrix_1.sum(dim=-1).sum(dim=-1)
        identity_matrix_2 = diagonal_sum(identity_matrix_2)
        identity_matrix_1.backward()
        identity_matrix_2.backward()
        self.assertTrue(torch.allclose(identity_matrix_1, identity_matrix_2))

    def test_compile(self):
        """Test that diagonal_sum works with torch.compile."""
        identity_matrix = torch.eye(
            5, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        sum_diagonal_compiled = torch.compile(diagonal_sum)
        matrix = sum_diagonal_compiled(identity_matrix)

    def _create_model(self):
        """Create a simple test model that uses diagonal_sum."""

        class Model(torch.nn.Module):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.linear_1 = torch.nn.Linear(10, 10)

            def forward(self, x):
                x = F.relu(self.linear_1(x))
                x = diagonal_sum(x)
                return F.sigmoid(x)

        return Model().to(torch.bfloat16)

    def test_training(self):
        """Test diagonal_sum in a training loop."""
        model = self._create_model().to("cuda")
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters())
        x = torch.randn((10, 10), dtype=torch.bfloat16, device="cuda")
        one = torch.tensor([0.5], dtype=torch.bfloat16, device="cuda").squeeze()

        for _ in range(2):
            output = model(x)
            loss = loss_fn(one, output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_training_mixed_precision(self):
        """Test diagonal_sum with mixed precision training."""
        model = self._create_model().to("cuda")
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters())
        x = torch.randn((10, 10), dtype=torch.bfloat16, device="cuda")
        one = torch.tensor([0.5], dtype=torch.bfloat16, device="cuda").squeeze()

        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(x)
                loss = loss_fn(one, output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_ddp(self):
        """Test diagonal_sum with distributed data parallel."""
        error_code = subprocess.call(["python", "./tests/sum_diagonal/_ddp.py"])
        if error_code:
            raise RuntimeError(f"ERROR: ddp exited with error code {error_code}")

    def test_ddp_compiled(self):
        """Test diagonal_sum with distributed data parallel and torch.compile."""
        error_code = subprocess.call(["python", "./tests/sum_diagonal/_ddp.py", "compiled"])
        if error_code:
            raise RuntimeError(f"ERROR: ddp compiled exited with error code {error_code}")

    def test_edge_case_values(self):
        """Test diagonal_sum with edge case values: 2, 0, -2, -1."""
        # Test with value 2
        matrix = torch.full((5, 5), 2.0, dtype=torch.bfloat16, device="cuda")
        result = diagonal_sum(matrix)
        expected = torch.tensor(10.0, dtype=torch.bfloat16, device="cuda")  # 5 * 2
        self.assertTrue(torch.allclose(result, expected), f"Failed for value 2: got {result}, expected {expected}")

        # Test with value 0
        matrix = torch.zeros((5, 5), dtype=torch.bfloat16, device="cuda")
        result = diagonal_sum(matrix)
        expected = torch.tensor(0.0, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(torch.allclose(result, expected), f"Failed for value 0: got {result}, expected {expected}")

        # Test with value -2
        matrix = torch.full((5, 5), -2.0, dtype=torch.bfloat16, device="cuda")
        result = diagonal_sum(matrix)
        expected = torch.tensor(-10.0, dtype=torch.bfloat16, device="cuda")  # 5 * -2
        self.assertTrue(torch.allclose(result, expected), f"Failed for value -2: got {result}, expected {expected}")

        # Test with value -1
        matrix = torch.full((5, 5), -1.0, dtype=torch.bfloat16, device="cuda")
        result = diagonal_sum(matrix)
        expected = torch.tensor(-5.0, dtype=torch.bfloat16, device="cuda")  # 5 * -1
        self.assertTrue(torch.allclose(result, expected), f"Failed for value -1: got {result}, expected {expected}")

    def test_edge_case_values_batched(self):
        """Test diagonal_sum with edge case values in batched tensors."""
        # Create a batched tensor with different edge case values
        batch_size = 4
        matrix_size = 3

        # Batch contains matrices with diagonal values: 2, 0, -2, -1
        matrices = torch.zeros((batch_size, matrix_size, matrix_size), dtype=torch.bfloat16, device="cuda")
        matrices[0] = torch.eye(matrix_size, device="cuda") * 2.0
        matrices[1] = torch.eye(matrix_size, device="cuda") * 0.0
        matrices[2] = torch.eye(matrix_size, device="cuda") * -2.0
        matrices[3] = torch.eye(matrix_size, device="cuda") * -1.0

        result = diagonal_sum(matrices)
        expected = torch.tensor([6.0, 0.0, -6.0, -3.0], dtype=torch.bfloat16, device="cuda")

        self.assertTrue(torch.allclose(result, expected),
                       f"Failed for batched edge cases: got {result}, expected {expected}")


if __name__ == "__main__":
    unittest.main()
