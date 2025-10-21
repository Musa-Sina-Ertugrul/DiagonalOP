"""Unit tests for torch.compile compatibility with diagonal operations.

This module contains comprehensive tests for torch.compile integration including:
- Basic compilation tests
- max-autotune mode tests
- Gradient computation with compilation
- All diagonal operations with compilation
- Shape handling with repeat operations
"""

import sys

sys.path.append("./")

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from diagonal import *


class TestTorchCompile(unittest.TestCase):
    """Test suite for torch.compile compatibility with diagonal operations."""

    def test_check_cuda(self):
        """Verify CUDA is available for testing."""
        self.assertTrue(torch.cuda.is_available(), "CUDA must be available for tests")

    def test_compile_basic(self):
        """Test basic torch.compile without mode specified."""
        @torch.compile
        def forward(x, value):
            return diagonal_mul(x, value)

        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16)
        value = -1.0

        result = forward(x, value)
        self.assertEqual(result.shape, x.shape, "Output shape must match input shape")

    def test_compile_max_autotune(self):
        """Test torch.compile with max-autotune mode."""
        @torch.compile(mode="max-autotune")
        def forward(x, value):
            return diagonal_mul(x, value)

        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16)
        value = -1.0

        result = forward(x, value)
        self.assertEqual(result.shape, x.shape, "Output shape must match input shape")

    def test_compile_with_gradients(self):
        """Test gradient computation with compiled function."""
        @torch.compile(mode="max-autotune")
        def forward(x, value):
            return diagonal_mul(x, value)

        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        value = torch.tensor(-1.0, device='cuda', dtype=torch.bfloat16, requires_grad=True)

        result = forward(x, value)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "Input gradients must be computed")
        self.assertIsNotNone(value.grad, "Value gradients must be computed")

    def test_compile_all_operations(self):
        """Test all diagonal operations with torch.compile."""
        @torch.compile(mode="max-autotune")
        def all_ops(x, value):
            x = diagonal_add(x, value)
            x = diagonal_sub(x, value)
            x = diagonal_mul(x, value)
            x = diagonal_div(x, value)
            s = diagonal_sum(x)
            return x, s

        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        value = torch.tensor(2.0, device='cuda', dtype=torch.bfloat16, requires_grad=True)

        result, sum_diag = all_ops(x, value)

        self.assertEqual(result.shape, x.shape, "Result shape must match input shape")
        self.assertEqual(sum_diag.shape, torch.Size([2, 3]), "Sum diagonal shape must be correct")

        # Test gradients
        loss = result.sum() + sum_diag.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "Input gradients must be computed")
        self.assertIsNotNone(value.grad, "Value gradients must be computed")

    def test_repeat_pattern_square(self):
        """Test correct repeat pattern that maintains square matrices."""
        @torch.compile(mode="max-autotune")
        def forward(x, dim):
            # CORRECT: repeat both dimensions to maintain square
            x = x.repeat(1, 1, dim, dim)
            x = diagonal_mul(x, -1.0)
            return x

        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16)
        dim = 5

        result = forward(x, dim)
        expected_shape = torch.Size([2, 3, 20, 20])
        self.assertEqual(result.shape, expected_shape, f"Result shape must be {expected_shape}")

    def test_non_square_error(self):
        """Test that non-square matrices correctly raise error."""
        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16)

        # Create non-square by repeating only one dimension
        x_non_square = x.repeat(1, 1, 5, 1)  # Shape: [2, 3, 20, 4]

        with self.assertRaises(RuntimeError) as context:
            diagonal_mul(x_non_square, -1.0)

        self.assertIn("same shape", str(context.exception), "Error message must mention shape requirement")

    def test_compile_with_tensor_value(self):
        """Test compilation with tensor values instead of scalars."""
        @torch.compile(mode="max-autotune")
        def forward(x, value):
            return diagonal_mul(x, value)

        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16)
        value = torch.randn(4, device='cuda', dtype=torch.bfloat16)  # Vector value

        result = forward(x, value)
        self.assertEqual(result.shape, x.shape, "Output shape must match input shape")

    def test_compile_backward_all_ops(self):
        """Test backward pass for all operations with compilation."""
        @torch.compile(mode="max-autotune")
        def forward(x, v_add, v_sub, v_mul, v_div):
            x = diagonal_add(x, v_add)
            x = diagonal_sub(x, v_sub)
            x = diagonal_mul(x, v_mul)
            x = diagonal_div(x, v_div)
            return x

        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v_add = torch.tensor(1.0, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v_sub = torch.tensor(0.5, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v_mul = torch.tensor(2.0, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v_div = torch.tensor(1.5, device='cuda', dtype=torch.bfloat16, requires_grad=True)

        result = forward(x, v_add, v_sub, v_mul, v_div)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "Input gradients must be computed")
        self.assertIsNotNone(v_add.grad, "Add value gradients must be computed")
        self.assertIsNotNone(v_sub.grad, "Sub value gradients must be computed")
        self.assertIsNotNone(v_mul.grad, "Mul value gradients must be computed")
        self.assertIsNotNone(v_div.grad, "Div value gradients must be computed")

    def test_compile_with_model(self):
        """Test torch.compile with a full neural network model."""
        class DiagonalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 4)
                self.linear2 = nn.Linear(4, 4)

            def forward(self, x):
                # x shape: [batch, 4, 4]
                x = self.linear1(x)
                x = diagonal_mul(x, 0.5)
                x = F.relu(x)
                x = self.linear2(x)
                x = diagonal_add(x, 1.0)
                return x

        model = DiagonalModel().to('cuda').to(torch.bfloat16)
        compiled_model = torch.compile(model, mode="max-autotune")

        x = torch.randn(8, 4, 4, device='cuda', dtype=torch.bfloat16)
        result = compiled_model(x)

        self.assertEqual(result.shape, x.shape, "Model output shape must match input")

    def test_compile_sum_diagonal(self):
        """Test sum_diagonal operation with torch.compile."""
        @torch.compile(mode="max-autotune")
        def forward(x):
            return diagonal_sum(x)

        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16)
        result = forward(x)

        expected_shape = torch.Size([2, 3])
        self.assertEqual(result.shape, expected_shape, f"Sum result shape must be {expected_shape}")

    def test_compile_mixed_ops(self):
        """Test mixing diagonal operations with standard PyTorch operations."""
        @torch.compile(mode="max-autotune")
        def forward(x, value):
            x = diagonal_mul(x, value)
            x = torch.relu(x)
            x = diagonal_add(x, 1.0)
            x = torch.sigmoid(x)
            return x

        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16)
        value = 2.0

        result = forward(x, value)
        self.assertEqual(result.shape, x.shape, "Output shape must match input shape")

    def test_compile_with_autocast(self):
        """Test torch.compile with autocast mixed precision.

        Note: Custom CUDA ops preserve input dtype. To use mixed precision,
        pass bfloat16 tensors directly instead of relying on autocast conversion.
        """
        @torch.compile(mode="max-autotune")
        def forward(x, value):
            return diagonal_mul(x, value)

        # Use bfloat16 directly since custom ops don't auto-convert
        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16)
        value = 2.0

        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = forward(x, value)

        self.assertEqual(result.shape, x.shape, "Output shape must match input shape")
        self.assertEqual(result.dtype, torch.bfloat16, "Output dtype must match input dtype")

    def test_compile_autocast_all_ops(self):
        """Test all diagonal operations with torch.compile + autocast.

        Note: Custom CUDA ops preserve input dtype. To use mixed precision,
        pass bfloat16 tensors directly.
        """
        @torch.compile(mode="max-autotune")
        def all_ops(x, value):
            x = diagonal_add(x, value)
            x = diagonal_sub(x, value)
            x = diagonal_mul(x, value)
            x = diagonal_div(x, value)
            s = diagonal_sum(x)
            return x, s

        # Use bfloat16 directly
        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16)
        value = 2.0

        with torch.autocast("cuda", dtype=torch.bfloat16):
            result, sum_diag = all_ops(x, value)

        self.assertEqual(result.shape, x.shape, "Result shape must match input shape")
        self.assertEqual(sum_diag.shape, torch.Size([2, 3]), "Sum diagonal shape must be correct")
        self.assertEqual(result.dtype, torch.bfloat16, "Result dtype must match input dtype")

    def test_compile_autocast_with_gradients(self):
        """Test gradient computation with torch.compile + autocast."""
        @torch.compile(mode="max-autotune")
        def forward(x, value):
            return diagonal_mul(x, value)

        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        value = torch.tensor(2.0, device='cuda', dtype=torch.bfloat16, requires_grad=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = forward(x, value)
            loss = result.sum()

        loss.backward()

        self.assertIsNotNone(x.grad, "Input gradients must be computed")
        self.assertIsNotNone(value.grad, "Value gradients must be computed")

    def test_compile_autocast_model(self):
        """Test full model with torch.compile + autocast mixed precision.

        Note: Linear layers are autocast-aware and will use bfloat16 automatically.
        Diagonal ops preserve the dtype of the input they receive from the linear layers.
        """
        class DiagonalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 4)
                self.linear2 = nn.Linear(4, 4)

            def forward(self, x):
                x = self.linear1(x)  # Linear converts to bfloat16 under autocast
                x = diagonal_mul(x, 0.5)  # Diagonal ops preserve bfloat16
                x = F.relu(x)
                x = self.linear2(x)
                x = diagonal_add(x, 1.0)
                return x

        model = DiagonalModel().to('cuda')
        compiled_model = torch.compile(model, mode="max-autotune")

        x = torch.randn(8, 4, 4, device='cuda', dtype=torch.float32)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = compiled_model(x)

        self.assertEqual(result.shape, x.shape, "Model output shape must match input")
        # Linear layers use bfloat16 under autocast, diagonal ops preserve it
        self.assertEqual(result.dtype, torch.bfloat16, "Model output must be bfloat16 under autocast")

    def test_training_mixed_precision_compiled(self):
        """Test training loop with torch.compile + autocast (matching style from other tests)."""
        def _create_model():
            return nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
            )

        model = _create_model().to("cuda")
        compiled_model = torch.compile(model, mode="max-autotune")
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(compiled_model.parameters())

        x = torch.randn((10, 10), dtype=torch.bfloat16, device="cuda")
        target = torch.ones(
            (10, 10), device="cuda", dtype=torch.bfloat16, requires_grad=False
        )

        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = compiled_model(x)
                output = diagonal_mul(output, 2.0)
                loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_compile_autocast_non_contiguous(self):
        """Test torch.compile + autocast with non-contiguous tensors.

        This test specifically validates the .reshape() fix for the illegal memory access bug.
        """
        @torch.compile(mode="max-autotune")
        def forward(x, value):
            return diagonal_mul(x, value)

        # Create non-contiguous tensor via transpose
        x = torch.randn(3, 2, 4, 4, device='cuda', dtype=torch.bfloat16)
        x_transposed = x.transpose(0, 1)  # Non-contiguous

        self.assertFalse(x_transposed.is_contiguous(), "Tensor must be non-contiguous for this test")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = forward(x_transposed, 2.0)

        self.assertEqual(result.shape, x_transposed.shape, "Output shape must match input shape")

    def test_compile_autocast_backward_all_ops(self):
        """Test backward pass for all operations with torch.compile + autocast."""
        @torch.compile(mode="max-autotune")
        def forward(x, v_add, v_sub, v_mul, v_div):
            x = diagonal_add(x, v_add)
            x = diagonal_sub(x, v_sub)
            x = diagonal_mul(x, v_mul)
            x = diagonal_div(x, v_div)
            return x

        x = torch.randn(2, 3, 4, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v_add = torch.tensor(1.0, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v_sub = torch.tensor(0.5, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v_mul = torch.tensor(2.0, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        v_div = torch.tensor(1.5, device='cuda', dtype=torch.bfloat16, requires_grad=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = forward(x, v_add, v_sub, v_mul, v_div)
            loss = result.sum()

        loss.backward()

        self.assertIsNotNone(x.grad, "Input gradients must be computed")
        self.assertIsNotNone(v_add.grad, "Add value gradients must be computed")
        self.assertIsNotNone(v_sub.grad, "Sub value gradients must be computed")
        self.assertIsNotNone(v_mul.grad, "Mul value gradients must be computed")
        self.assertIsNotNone(v_div.grad, "Div value gradients must be computed")


def suite():
    """Create test suite for torch.compile compatibility tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTorchCompile))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
