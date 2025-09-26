.. Diagonal Operations documentation master file, created by
   sphinx-quickstart on Fri Sep 26 13:57:54 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Diagonal Operations Documentation
==================================

Welcome to the Diagonal Operations library documentation! This library provides efficient diagonal matrix operations using custom CUDA kernels with automatic differentiation support through PyTorch's autograd system.

🚀 Key Features
---------------

* **🔥 Custom CUDA Kernels**: Blazing fast GPU-accelerated operations
* **🧠 Full Autograd Support**: Complete automatic differentiation integration
* **⚙️ Torch Compile Ready**: Full compatibility with PyTorch 2.0 compilation
* **🎯 Mixed Precision**: FP16/BF16 support for faster training
* **📊 Tensor Values**: Support for both scalar and tensor diagonal values
* **🔥 Multi-GPU Support**: Seamless DistributedDataParallel (DDP) integration
* **📦 Batch Processing**: Efficient batch operations on multiple matrices
* **🧪 Comprehensive Testing**: Extensive test suite including distributed training

Quick Start
-----------

Install the package and start using diagonal operations:

.. code-block:: python

   import torch
   from diagonal import diagonal_add, diagonal_mul, diagonal_sum

   # Basic usage
   x = torch.randn(3, 3, device='cuda')
   result = diagonal_add(x, 2.0)  # Add scalar to diagonal

   # Tensor values support - different value for each diagonal element
   values = torch.tensor([1.0, 2.0, 3.0], device='cuda')
   result = diagonal_add(x, values)

   # Mixed precision support
   with torch.autocast('cuda', dtype=torch.bfloat16):
       result = diagonal_mul(x.half(), 0.5)

   # Torch compile support for maximum performance
   compiled_add = torch.compile(diagonal_add)
   result = compiled_add(x, 2.0)

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   api
   modules

