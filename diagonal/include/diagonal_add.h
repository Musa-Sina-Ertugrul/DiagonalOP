/**
 * @file diagonal_add.h
 * @brief Header for diagonal addition operation
 *
 * Provides function declaration for adding values to diagonal elements of tensors.
 */

#ifndef DIAGONAL_ADD_H
#define DIAGONAL_ADD_H

#include <torch/extension.h>

/**
 * @brief Add a value to the diagonal elements of a tensor
 *
 * This function performs an in-place addition of a scalar or vector value to the
 * diagonal elements of a square matrix or batch of square matrices.
 *
 * @param input Input tensor (square matrix or batch of square matrices)
 * @param value Scalar tensor or 1D tensor with length equal to matrix dimension
 * @return torch::Tensor Modified tensor with value added to diagonal elements
 *
 * @note The input tensor is modified in-place for efficiency
 * @note Supports batched operations for tensors with shape (..., N, N)
 */
torch::Tensor add_diagonal(torch::Tensor input, torch::Tensor value);

#endif // DIAGONAL_ADD_H
