import os

# Force use of system CUDA toolkit BEFORE importing torch
os.environ['CUDA_HOME'] = '/usr/local/cuda/'
os.environ['CUDA_PATH'] = '/usr/local/cuda/'
os.environ['CUDACXX'] = '/usr/bin/nvcc'

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Get the directory where setup.py is located
setup_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='diagonal_cuda',
    ext_modules=[
        CUDAExtension(
            name='diagonal_cuda',
            sources=[
                os.path.join(setup_dir, './diagonal_add.cu'),
                os.path.join(setup_dir, './diagonal_sum.cu'),
                os.path.join(setup_dir, './diagonal_sub.cu'),
                os.path.join(setup_dir, './diagonal_mul.cu'),
                os.path.join(setup_dir, './diagonal_div.cu'),
                os.path.join(setup_dir, './diagonal_utils.cu'),
            ],
            include_dirs=[
                setup_dir,
                os.path.join(setup_dir, 'include'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '-lineinfo',
                    '-Xptxas=-allow-expensive-optimizations=true',
                    '-gencode=arch=compute_89,code=sm_89',
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ]
            },
            extra_link_args=[
                '-lcudadevrt',
                '-lcudart',
                '-lcudnn',
                '-lcublas',
                '-lcublasLt',
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
