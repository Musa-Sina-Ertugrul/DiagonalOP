# DiagonalOP

High-performance diagonal matrix operations with PyTorch autograd support.

## Features

- **Custom CUDA Kernels** - Blazing fast GPU operations
- **Multi-GPU Support** - Works seamlessly with DistributedDataParallel (DDP)
- **Torch Compile Ready** - Full compatibility with `torch.compile()`
- **Mixed Precision** - FP16/BF16 support for faster training
- **Tensor Values** - Support for both scalar and tensor values
- **Full Autograd** - Complete gradient computation support
- **Batch Operations** - Efficient batch processing

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

```python
import torch
from diagonal import diagonal_add, diagonal_mul, diagonal_sum

# Basic usage
x = torch.randn(3, 3, device='cuda')
result = diagonal_add(x, 2.0)  # Add scalar to diagonal

# Tensor values support
values = torch.tensor([1.0, 2.0, 3.0], device='cuda')
result = diagonal_add(x, values)  # Add different values to each diagonal element

# Mixed precision
with torch.autocast('cuda', dtype=torch.bfloat16):
    result = diagonal_mul(x.half(), 0.5)

# Torch compile support
compiled_add = torch.compile(diagonal_add)
result = compiled_add(x, 2.0)  # Faster execution
```

## Functions

- `diagonal_add(input, value)` - Add value to diagonal elements
- `diagonal_sub(input, value)` - Subtract value from diagonal elements  
- `diagonal_mul(input, value)` - Multiply diagonal elements by value
- `diagonal_div(input, value)` - Divide diagonal elements by value
- `diagonal_sum(input)` - Sum diagonal elements

## Advanced Usage

### Multi-GPU Training (DDP)
```python
from torch.nn.parallel import DistributedDataParallel as DDP
from diagonal import diagonal_add

# Works seamlessly with DDP
model = DDP(your_model, device_ids=[rank])
x = torch.randn(batch_size, dim, dim, device=f'cuda:{rank}')
result = diagonal_add(x, 1.0)  # Distributed across GPUs
```

### Batch Processing
```python
# Process multiple matrices at once
batch_matrices = torch.randn(32, 128, 128, device='cuda')  # 32 matrices of 128x128
result = diagonal_mul(batch_matrices, 0.5)  # Applied to all 32 matrices
```

### Gradient Computation
```python
x = torch.randn(3, 3, device='cuda', requires_grad=True)
values = torch.tensor([1.0, 2.0, 3.0], device='cuda', requires_grad=True)

result = diagonal_add(x, values)
loss = result.sum()
loss.backward()  # Gradients computed for both x and values

print(x.grad)      # Gradients w.r.t input matrix
print(values.grad) # Gradients w.r.t diagonal values
```

## Development

### Setup
```bash
export CUDA_HOME=/usr/local/cuda
pip install -e .[dev]
```

### Testing
```bash
python -m tests                  # Run all tests
python -m unittest tests.add_diagonal.test_add_diagonal  # Specific test
```

### Benchmarking
```bash
python -m benchmark              # Run all benchmarks (default: 128x2048x2048)
python -m benchmark --batch-size 64 --matrix-size 1024  # Custom config
python -m benchmark --dtype float16 --no-gradient       # FP16, skip gradient
python -m benchmark --help       # See all options
```

### Formatting
```bash
black .                          # Format code
black --check .                  # Check formatting
```

### Documentation
```bash
cd docs && make html             # Build docs
cd docs && make clean && make html  # Rebuild docs
xdg-open docs/_build/html/index.html  # View docs
```

### Common Commands
```bash
# Git workflow
git status
git add .
git commit -m "message"
git push origin main

# Package building
python setup.py sdist
python setup.py bdist_wheel

# CUDA check
nvidia-smi
nvcc --version
```

## Troubleshooting

```bash
# CUDA issues
export CUDA_HOME=/usr/local/cuda
echo $CUDA_HOME

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Reinstall
pip uninstall diagonal
pip install -e .
```

## Author

Musa Sina ERTUGRUL (m.s.ertugrul@gmail.com)