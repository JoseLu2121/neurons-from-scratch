#pragma once
#include "tensor.h"



// Main backend class
class Backend {
public:
    Backend() = default;
    virtual ~Backend() = default;

    // Memory management
    virtual float* alloc(size_t size) = 0;
    virtual void free(float* ptr) = 0;
    virtual void set(float* ptr, float value, size_t size) = 0;
    
    // Operations
    virtual void binary(const TensorInfo& a, const TensorInfo& b, TensorInfo& out, BinaryOp op) = 0;
    virtual void unary(const TensorInfo& a, TensorInfo& out, UnaryOp op) = 0;
    virtual void reduce(const TensorInfo& in, TensorInfo& out, int dim, ReduceOp op) = 0;

    // Operation for matmul
    virtual void gemm(const TensorInfo& a, const TensorInfo& b, TensorInfo& out) = 0;

    virtual void accumulate_grad(std::shared_ptr<Tensor> param, std::shared_ptr<Tensor> incoming_grad) = 0;

    virtual void gather(const TensorInfo& w, const TensorInfo& indexes, TensorInfo& out) = 0;
    virtual void scatter_add(const TensorInfo& indexes, const TensorInfo& incoming_grad, const TensorInfo& w_grad) = 0;

};


struct CPUBackend : public Backend {

    CPUBackend() = default;

    float* alloc(size_t size) override;
    void free(float* ptr) override;
    void set(float* ptr, float value, size_t size) override;

    void binary(const TensorInfo& a, const TensorInfo& b, TensorInfo& out, BinaryOp op) override;
    void unary(const TensorInfo& a, TensorInfo& out, UnaryOp op) override;    
    void reduce(const TensorInfo& in, TensorInfo& out, int dim, ReduceOp op) override;
    void gemm(const TensorInfo& a, const TensorInfo& b, TensorInfo& out) override;

    void accumulate_grad(std::shared_ptr<Tensor> param, std::shared_ptr<Tensor> incoming_grad) override;
    void gather(const TensorInfo& w, const TensorInfo& indexes, TensorInfo& out);
    void scatter_add(const TensorInfo& indexes, const TensorInfo& incoming_grad, const TensorInfo& w_grad);
};


struct CPUBackendOptimized : public Backend {

    CPUBackendOptimized() = default;

    float* alloc(size_t size) override;
    void free(float* ptr) override;
    void set(float* ptr, float value, size_t size) override;

    void binary(const TensorInfo& a, const TensorInfo& b, TensorInfo& out, BinaryOp op) override;
    void unary(const TensorInfo& a, TensorInfo& out, UnaryOp op) override;    
    // Missing overrides
    void reduce(const TensorInfo& in, TensorInfo& out, int dim, ReduceOp op) override;
    void gemm(const TensorInfo& a, const TensorInfo& b, TensorInfo& out) override;

    void accumulate_grad(std::shared_ptr<Tensor> param, std::shared_ptr<Tensor> incoming_grad) override;
    void gather(const TensorInfo& w, const TensorInfo& indexes, TensorInfo& out);
    void scatter_add(const TensorInfo& indexes, const TensorInfo& incoming_grad, const TensorInfo& w_grad);

};

struct GEMMOptimizedBackend : public Backend {

    GEMMOptimizedBackend() = default;

    float* alloc(size_t size) override;
    void free(float* ptr) override;
    void set(float* ptr, float value, size_t size) override;

    void binary(const TensorInfo& a, const TensorInfo& b, TensorInfo& out, BinaryOp op) override;
    void unary(const TensorInfo& a, TensorInfo& out, UnaryOp op) override;    
    // Missing overrides
    void reduce(const TensorInfo& in, TensorInfo& out, int dim, ReduceOp op) override;
    void gemm(const TensorInfo& a, const TensorInfo& b, TensorInfo& out) override;

    void accumulate_grad(std::shared_ptr<Tensor> param, std::shared_ptr<Tensor> incoming_grad) override;
    void gather(const TensorInfo& w, const TensorInfo& indexes, TensorInfo& out);
    void scatter_add(const TensorInfo& indexes, const TensorInfo& incoming_grad, const TensorInfo& w_grad);

};