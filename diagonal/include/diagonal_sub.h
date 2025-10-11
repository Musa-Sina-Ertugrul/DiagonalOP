/**
 * @file diagonal_sub.h
 * @brief Header for diagonal subtraction operation
 *
 * Provides function declaration for subtracting values from diagonal elements of tensors.
 */

#ifndef DIAGONAL_SUB_H
#define DIAGONAL_SUB_H

#include <torch/extension.h>

/**
 * @brief Subtract a value from the diagonal elements of a tensor
 *
 * This function performs an in-place subtraction of a scalar or vector value from the
 * diagonal elements of a square matrix or batch of square matrices.
 *
 * @param input Input tensor (square matrix or batch of square matrices)
 * @param value Scalar tensor or 1D tensor with length equal to matrix dimension
 * @return torch::Tensor Modified tensor with value subtracted from diagonal elements
 *
 * @note The input tensor is modified in-place for efficiency
 * @note Supports batched operations for tensors with shape (..., N, N)
 */
torch::Tensor sub_diagonal(torch::Tensor input, torch::Tensor value);

#endif // DIAGONAL_SUB_H
