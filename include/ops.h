#pragma once
#include "tensor.h"


std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> transpose_view(std::shared_ptr<Tensor> a);