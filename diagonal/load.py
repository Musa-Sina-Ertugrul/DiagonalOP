"""Module for loading custom CUDA kernels for diagonal operations.

This module dynamically compiles and loads CUDA kernels for efficient diagonal matrix operations.
Requires CUDA toolkit to be installed and CUDA_HOME environment variable to be set.

Note:
    Make sure to set CUDA_HOME environment variable:
    export CUDA_HOME=/usr/local/cuda
"""

import torch
from torch.utils.cpp_extension import load

_diagonal_kernel = load(
    "_diagonal_kernel",
    sources=[
        "./diagonal/add_diagonal.cu",
        "./diagonal/mul_diagonal.cu",
        "./diagonal/sub_diagonal.cu",
        "./diagonal/div_diagonal.cu",
        "./diagonal/sum_diagonal.cu",
        "./diagonal/utils.cu",
        "./diagonal/diagonal.cu",
    ],
    extra_cuda_cflags=["-O2", "--use_fast_math"],
    verbose=True,
)
