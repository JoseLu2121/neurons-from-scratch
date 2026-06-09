#pragma once
#include "tensor.h"
#include "unit.h"

std::shared_ptr<Unit> operator+(const std::shared_ptr<Unit>& a, const std::shared_ptr<Unit>& b);
std::shared_ptr<Unit> operator-(const std::shared_ptr<Unit>& a, const std::shared_ptr<Unit>& b);
std::shared_ptr<Unit> operator*(const std::shared_ptr<Unit>& a, const std::shared_ptr<Unit>& b);
std::shared_ptr<Unit> operator/(const std::shared_ptr<Unit>& a, const std::shared_ptr<Unit>& b);
std::shared_ptr<Unit> operatorpow(const std::shared_ptr<Unit>& a, const std::shared_ptr<Unit>& b);
std::shared_ptr<Unit> relu(const std::shared_ptr<Unit>& a);
std::shared_ptr<Unit> operator_tanh(const std::shared_ptr<Unit>& a);
std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, bool require_grad=true);
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> transpose_view(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> linear_activation(std::shared_ptr<Tensor> x);
std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, float scalar);
std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, float scalar);
std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, float scalar);
std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> sqrt(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a, int dim);
std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a, int dim);
std::shared_ptr<Tensor> argmax(std::shared_ptr<Tensor> a, int dim);
std::shared_ptr<Tensor> matmul_no_grad(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> gather(std::shared_ptr<Tensor> w, std::shared_ptr<Tensor> ind);
std::shared_ptr<Tensor> conv2d(std::shared_ptr<Tensor>  input, std::shared_ptr<Tensor>  w, int stride, int padding);
std::shared_ptr<Tensor> flatten(std::shared_ptr<Tensor> a);