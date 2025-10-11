/**
 * @file diagonal_mul.h
 * @brief Header for diagonal multiplication operation
 *
 * Provides function declaration for multiplying diagonal elements of tensors by values.
 */

#ifndef DIAGONAL_MUL_H
#define DIAGONAL_MUL_H

#include <torch/extension.h>

/**
 * @brief Multiply the diagonal elements of a tensor by a value
 *
 * This function performs an in-place multiplication of the diagonal elements of a
 * square matrix or batch of square matrices by a scalar or vector value.
 *
 * @param input Input tensor (square matrix or batch of square matrices)
 * @param value Scalar tensor or 1D tensor with length equal to matrix dimension
 * @return torch::Tensor Modified tensor with diagonal elements multiplied by value
 *
 * @note The input tensor is modified in-place for efficiency
 * @note Supports batched operations for tensors with shape (..., N, N)
 */
torch::Tensor mul_diagonal(torch::Tensor input, torch::Tensor value);

#endif // DIAGONAL_MUL_H
