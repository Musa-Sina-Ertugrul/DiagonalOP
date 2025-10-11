#include <torch/extension.h>

torch::Tensor add_diagonal(torch::Tensor input, torch::Tensor value);
torch::Tensor mul_diagonal(torch::Tensor input, torch::Tensor value);
torch::Tensor sub_diagonal(torch::Tensor input, torch::Tensor value);
torch::Tensor div_diagonal(torch::Tensor input, torch::Tensor value);
torch::Tensor sum_diagonal(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("add_diagonal", &add_diagonal, "Adds a value to the diagonal");
    m.def("mul_diagonal", &mul_diagonal, "Multiplies the diagonal by a value");
    m.def("sub_diagonal", &sub_diagonal, "Subtracts a value from the diagonal");
    m.def("div_diagonal", &div_diagonal, "Divides the diagonal by a value");
    m.def("sum_diagonal",&sum_diagonal,"Sums diagonal");
}