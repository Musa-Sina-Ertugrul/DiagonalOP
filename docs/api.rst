API Reference
=============

This section provides detailed documentation of all functions available in the Diagonal Operations library.

Main Functions
--------------

diagonal_add
~~~~~~~~~~~~

.. code-block:: python

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

Add a value to the diagonal elements of a matrix. This function performs efficient diagonal addition using custom CUDA kernels with full autograd support.

diagonal_sub
~~~~~~~~~~~~

.. code-block:: python

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

Subtract a value from the diagonal elements of a matrix using optimized CUDA operations.

diagonal_mul
~~~~~~~~~~~~

.. code-block:: python

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

Multiply the diagonal elements of a matrix by a value with efficient GPU computation.

diagonal_div
~~~~~~~~~~~~

.. code-block:: python

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

Divide the diagonal elements of a matrix by a value using custom CUDA kernels.

diagonal_sum
~~~~~~~~~~~~

.. code-block:: python

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

Sum the diagonal elements of a matrix with automatic gradient computation support.

Advanced Features
-----------------

🔥 Multi-GPU Support (DDP)
~~~~~~~~~~~~~~~~~~~~~~~~~~

DiagonalOP works seamlessly with PyTorch's DistributedDataParallel for multi-GPU training:

.. code-block:: python

   import torch
   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel as DDP
   from diagonal import diagonal_add
   
   # Initialize distributed training
   dist.init_process_group("nccl", rank=rank, world_size=world_size)
   
   # Create model and wrap with DDP
   model = DDP(your_model, device_ids=[rank])
   
   # Use diagonal operations in distributed setting
   x = torch.randn(batch_size, dim, dim, device=f'cuda:{rank}')
   result = diagonal_add(x, 1.0)  # Works across all GPUs

⚙️ Torch Compile Support
~~~~~~~~~~~~~~~~~~~~~~~~

All functions are fully compatible with PyTorch 2.0's ``torch.compile()`` for maximum performance:

.. code-block:: python

   import torch
   from diagonal import diagonal_add, diagonal_mul
   
   # Compile individual functions
   compiled_add = torch.compile(diagonal_add)
   compiled_mul = torch.compile(diagonal_mul)
   
   # Or compile entire models that use diagonal operations
   @torch.compile
   def my_model(x):
       x = diagonal_add(x, 1.0)
       x = diagonal_mul(x, 0.5)
       return x
   
   x = torch.randn(1000, 1000, device='cuda')
   result = compiled_add(x, 2.0)  # Faster execution

🎯 Mixed Precision Support
~~~~~~~~~~~~~~~~~~~~~~~~~~

Full support for FP16 and BF16 mixed precision training:

.. code-block:: python

   import torch
   from diagonal import diagonal_add, diagonal_mul
   
   x = torch.randn(512, 512, device='cuda')
   
   # Automatic mixed precision
   with torch.autocast('cuda', dtype=torch.bfloat16):
       result = diagonal_add(x, 2.0)
       result = diagonal_mul(result, 0.5)
   
   # Manual half precision
   x_half = x.half()
   result = diagonal_add(x_half, 2.0)

📊 Tensor Value Support
~~~~~~~~~~~~~~~~~~~~~~

Support for both scalar and tensor values - apply different values to each diagonal element:

.. code-block:: python

   import torch
   from diagonal import diagonal_add, diagonal_mul
   
   x = torch.randn(4, 4, device='cuda')
   
   # Scalar value (applied to all diagonal elements)
   result1 = diagonal_add(x, 2.0)
   
   # Tensor values (different value for each diagonal element)
   values = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
   result2 = diagonal_add(x, values)
   
   # Works with gradients too!
   values.requires_grad_(True)
   result3 = diagonal_mul(x, values)
   result3.sum().backward()
   print(values.grad)  # Gradients computed for each diagonal value

🧠 Full Autograd Support
~~~~~~~~~~~~~~~~~~~~~~~~

Complete automatic differentiation support with gradients for both inputs and values:

.. code-block:: python

   import torch
   from diagonal import diagonal_add
   
   # Both input matrix and values can have gradients
   x = torch.randn(3, 3, device='cuda', requires_grad=True)
   values = torch.tensor([1.0, 2.0, 3.0], device='cuda', requires_grad=True)
   
   result = diagonal_add(x, values)
   loss = result.sum()
   loss.backward()
   
   print("Matrix gradients:", x.grad)
   print("Value gradients:", values.grad)

📦 Batch Processing
~~~~~~~~~~~~~~~~~~

Efficient processing of batched matrices:

.. code-block:: python

   # Process multiple matrices simultaneously
   batch_matrices = torch.randn(32, 128, 128, device='cuda')  # 32 matrices
   batch_values = torch.randn(32, 128, device='cuda')         # Different values per matrix
   
   # Apply to entire batch efficiently
   result = diagonal_add(batch_matrices, batch_values)
   
   # Higher-dimensional batches also supported
   nested_batch = torch.randn(16, 8, 64, 64, device='cuda')   # 16x8 = 128 matrices
   result = diagonal_mul(nested_batch, 0.5)

🚀 Performance Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DiagonalOP is built for maximum performance:

.. code-block:: python

   import torch
   from diagonal import diagonal_add
   
   # Large matrix performance test
   large_matrix = torch.randn(4096, 4096, device='cuda')
   
   # Time the operation
   start_event = torch.cuda.Event(enable_timing=True)
   end_event = torch.cuda.Event(enable_timing=True)
   
   start_event.record()
   result = diagonal_add(large_matrix, 1.0)
   end_event.record()
   
   torch.cuda.synchronize()
   elapsed_time = start_event.elapsed_time(end_event)
   print(f"Diagonal add took {elapsed_time:.2f} ms")
   
   # Memory efficient - operations are in-place when possible
   print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")