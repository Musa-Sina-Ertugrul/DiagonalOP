import os
from setuptools import setup, find_packages

setup(
    name="diagonal",
    version="0.1.0",
    author="Musa Sina ERTUGRUL",
    author_email="m.s.ertugrul@gmail.com",
    description="Diagonal operations library with autograd support",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DiagonalOP",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch",
    ],
    extras_require={
        "dev": [
            "black",
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
)
