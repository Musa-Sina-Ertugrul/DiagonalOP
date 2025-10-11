/**
 * @file diagonal_sum.h
 * @brief Header for diagonal sum operation
 *
 * Provides function declaration for computing the sum of diagonal elements of tensors.
 */

#ifndef DIAGONAL_SUM_H
#define DIAGONAL_SUM_H

#include <torch/extension.h>

/**
 * @brief Sum the diagonal elements of a tensor
 *
 * This function computes the sum of diagonal elements (trace) of a square matrix
 * or batch of square matrices.
 *
 * @param input Input tensor (square matrix or batch of square matrices)
 * @return torch::Tensor Scalar tensor containing the sum of diagonal elements,
 *                       or tensor with shape (...,) for batched inputs
 *
 * @note For batched inputs with shape (..., N, N), returns tensor with shape (...)
 * @note This operation is equivalent to computing the trace of the matrix
 */
torch::Tensor sum_diagonal(torch::Tensor input);

#endif // DIAGONAL_SUM_H
