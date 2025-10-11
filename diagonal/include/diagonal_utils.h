#ifndef DIAGONAL_UTILS_H
#define DIAGONAL_UTILS_H

#include <torch/extension.h>

// Check if tensor is valid for diagonal operations
void check_tensor(const torch::Tensor& input);

#endif // DIAGONAL_UTILS_H
