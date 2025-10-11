/**
 * @file diagonal_div.h
 * @brief Header for diagonal division operation
 *
 * Provides function declaration for dividing diagonal elements of tensors by values.
 */

#ifndef DIAGONAL_DIV_H
#define DIAGONAL_DIV_H

#include <torch/extension.h>

/**
 * @brief Divide the diagonal elements of a tensor by a value
 *
 * This function performs an in-place division of the diagonal elements of a
 * square matrix or batch of square matrices by a scalar or vector value.
 *
 * @param input Input tensor (square matrix or batch of square matrices)
 * @param value Scalar tensor or 1D tensor with length equal to matrix dimension
 * @return torch::Tensor Modified tensor with diagonal elements divided by value
 *
 * @note The input tensor is modified in-place for efficiency
 * @note Supports batched operations for tensors with shape (..., N, N)
 * @warning Division by zero will result in inf/nan values
 */
torch::Tensor div_diagonal(torch::Tensor input, torch::Tensor value);

#endif // DIAGONAL_DIV_H
