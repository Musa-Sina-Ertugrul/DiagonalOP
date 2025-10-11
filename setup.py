import os
from setuptools import setup, find_packages

setup(
    name="diagonal",
    version="0.1.0",
    author="Musa Sina ERTUGRUL",
    author_email="m.s.ertugrul@gmail.com",
    description="High-performance diagonal matrix operations with PyTorch autograd support",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/musasinaertugrul/DiagonalOP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
    ],
    extras_require={
        "dev": [
            "black",
            "sphinx",
            "sphinx-rtd-theme",
            "pytest",
            "pytest-benchmark",
        ],
    },
)
