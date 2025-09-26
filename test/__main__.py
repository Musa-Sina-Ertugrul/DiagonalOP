"""Test runner for diagonal operations library.

This module automatically discovers and runs all unit tests in the test directory.
Run with: python -m test
"""

import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    start_dir = "./test"
    suite = loader.discover(start_dir)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
