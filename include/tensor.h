#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <iostream>
#include <cassert>
#include <unordered_set>
#include <random>
#include "types.h"

// tensor class 
class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    // data storage 
    std::shared_ptr<float[]> data;
    // number of elements of the tensor
    size_t total_size;
    // shape of the tensor (batch,row,col)
    std::vector<int> shape;
    // strides for each dimension
    std::vector<int> strides;
    
    // parent nodes of the tensor in the computation graph
    std::vector<std::shared_ptr<Tensor>> parents;
    // gradient tensor
    std::shared_ptr<Tensor> grad;
    // backward function
    std::function<void()> _backward;

    // constructor
    Tensor(const std::vector<int>& shape_param, const std::vector<float>& data_param = {},
           const std::vector<std::shared_ptr<Tensor>>& parents_param = {});
    
    // copy constructor
    Tensor(const Tensor& other);

public:
    // creates a tensor safely managed by a shared_ptr
    static std::shared_ptr<Tensor> create(const std::vector<int>& shape_param, 
                                          const std::vector<float>& data_param = {},
                                          const std::vector<std::shared_ptr<Tensor>>& parents_param = {});

    // all elements are 0
    static std::shared_ptr<Tensor> zeros(const std::vector<int>& shape);
    
    // all elements are 1
    static std::shared_ptr<Tensor> ones(const std::vector<int>& shape);
    
    // elements are random values between min_val and max_val
    static std::shared_ptr<Tensor> random(const std::vector<int>& shape, float min_val = -1.0f, float max_val = 1.0f);

    // get the size of the tensor
    size_t getSize() const { return total_size; }
    
    // get the dimension of the tensor
    int getDimension() const { return shape.size(); }
    
    // get the pointer to the data of the tensor
    float* getData() const { return data.get(); }
    
    // get the gradient tensor
    std::shared_ptr<Tensor> getGrad() const { return grad; }
    
    // get the vector of strides of the tensor
    const std::vector<int>& getStrides() const { return strides; }
    
    // get the vector of the shape of the tensor
    const std::vector<int>& getShape() const { return shape; }
    
    // get the parents of the tensor
    const std::vector<std::shared_ptr<Tensor>>& getParents() const { return parents; }

    void set_backward(std::function<void()> bw) { _backward = bw; }
    
    void set_parents(const std::vector<std::shared_ptr<Tensor>>& p) { parents = p; }

    void set_shape(const std::vector<int>& s) { shape = s; }

    void set_strides(const std::vector<int>& s) { strides = s; }

    void init_grad() { if (!grad) grad = Tensor::zeros(shape); }

    // compute backpropagation
    void backward();

    // prints the first n elements of the tensor
    void printElements(int count = 1) const;

    // prints the shape of the tensor
    void printShape() const;

    // prints the strides of the tensor
    void printStrides() const;

    // prints all the info of the tensor with a max number of data elements
    void info(int max_size = 20) const;
    
    // reshape a tensor
    std::shared_ptr<Tensor> reshape(const std::vector<int>& new_shape);

    // construct a batch view
    std::shared_ptr<Tensor> batch_view(int index, bool keep_dim);

    // slice a tensor by a start and end index
    std::shared_ptr<Tensor> slice(int start_idx, int end_idx);

    // build a 3d view of a tensor
    std::shared_ptr<Tensor> view_to_3d(); 

    // build a view for gemm
    std::shared_ptr<Tensor> view_to_gemm(bool as_b_term);
    
    // creates a broadcasted view to match target_shape
    std::shared_ptr<Tensor> broadcast_to(const std::vector<int>& target_shape);
    
    // calculates the resulting shape from broadcasting two shapes
    static std::vector<int> broadcast_shapes(const std::vector<int>& shape_a, const std::vector<int>& shape_b);

    // build an optimized version of the tensor
    TensorInfo getInfo();

    // compute a binary operation
    std::shared_ptr<Tensor> compute_binary_op(std::shared_ptr<Tensor> b, BinaryOp op);

    // compute a unary operation
    std::shared_ptr<Tensor> compute_unary_op(UnaryOp op);

    // compute matmul
    std::shared_ptr<Tensor> compute_matmul(std::shared_ptr<Tensor> b);

    // compute a reduce operation
    std::shared_ptr<Tensor> compute_reduce_op(int dim, ReduceOp op);

    // compute a flatten operation
    std::shared_ptr<Tensor> compute_flatten();
    
    // compute a gather operation
    std::shared_ptr<Tensor> compute_gather(std::shared_ptr<Tensor> ind);
    
    // compute a scatter add operation
    void compute_scatter_add(std::shared_ptr<Tensor> ind, std::shared_ptr<Tensor> incoming_grad);
    
    // compute conv2d
    std::shared_ptr<Tensor> compute_conv2d(std::shared_ptr<Tensor> w, int stride, int padding);

    // build topological sort of the graph
    void build_topo(std::shared_ptr<Tensor> v,
                std::vector<std::shared_ptr<Tensor>>& topo,
                std::unordered_set<Tensor*>& visited);

    // serialize tensor
    void serialize(std::ofstream& out) const;

    // deserialize tensor
    void deserialize(std::ifstream& in);

    // im2col operation
    void im2col(float* workspace, int out_height, int out_width, int kernel_height, int kernel_width, int stride, int padding);

    // col2im operation
    void col2im(const float* data_col, int out_height, int out_width, int kernel_height, int kernel_width, int stride, int padding);
};