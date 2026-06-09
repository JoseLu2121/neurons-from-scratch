#include "ops.h"
#include "types.h"
#include "utils.h"
#include "backend.h"
#include "device.h"
#include <algorithm>
#include <stdexcept>
#include <cmath> 
#include <iostream> 

using namespace std;

// Add operation
shared_ptr<Unit> operator+(const shared_ptr<Unit>& a, const shared_ptr<Unit>& b) {
    // The output is a new Unit with a value of a + b
    auto out = make_shared<Unit>(a->data + b->data, vector<shared_ptr<Unit>>{a, b}); // We add a and b as out childrens

    // The local derivative of (a + b) with respect to 'a' or 'b' are both 1, so we just apply
    // the chain rule -> 1 * out.grad
    out->backward = [a,b,out] () mutable {
        a->grad += out->grad;
        b->grad += out->grad;
    };
    return out;
};

// Mult operation
shared_ptr<Unit> operator*(const shared_ptr<Unit>& a, const shared_ptr<Unit>& b) {
    // The output is simply a*b
    auto out = make_shared<Unit>(a->data * b->data, vector<shared_ptr<Unit>>{a,b}); // We add a and b as out childrens

    // The local derivative of (a * b) with respect to 'a' is b, and with respect to 'b' is a
    // So we apply the chain rule a.grad = b * out.grad and viceversa 
    out->backward = [a,b,out] () mutable {
        a->grad += out->grad * b->data;
        b->grad += out->grad * a->data;
    };
    return out;
};

// Sub operation
shared_ptr<Unit> operator-(const shared_ptr<Unit>& a, const shared_ptr<Unit>& b) {
    // The output is a - b
    auto out = make_shared<Unit>(a->data - b->data, vector<shared_ptr<Unit>>{a,b});

    // The derivate is the same as add, but the second operand local derivate is negative
    out->backward = [a,b,out] () mutable {
        a->grad += out->grad;
        b->grad += -out->grad;
    };
    return out;
};

// Pow operation
shared_ptr<Unit> operatorpow(const shared_ptr<Unit>& a, double b){
    // We use the pow cmath C++ function 
    auto out = make_shared<Unit>(pow(a->data, b), vector<shared_ptr<Unit>>{a});

    // The derivative of a^b, with respect to a is just b * a^(b-1)
    out->backward = [a,b,out] () mutable {
        a->grad += b * pow(a->data,(b - 1)) * out->grad;
    };
    return out;
}

// Div operation
shared_ptr<Unit> operator/(const shared_ptr<Unit>& a, const shared_ptr<Unit>& b) {
    // We can make a div operation using mult and pow -> (a/b) = a * b^-1
    return a * operatorpow(b, -1);
}


// Relu operation
shared_ptr<Unit> relu(const shared_ptr<Unit>& a) {
    // If the value is less than zero, we set it to zero
    double t = a->data < 0 ? 0.0 : a->data;
    auto out = make_shared<Unit>(t, vector<shared_ptr<Unit>>{a});

    // The derivative of relu is 0 or 1 depending if the value is greater or lower than 0,
    // so out.data > 0 ? 1 * out.grad : 0 * out.grad (chain rule)
    out->backward = [a,out] () mutable {
        a->grad += (out->data > 0) ? out->grad : 0.0;
    };
    return out;
};

// Tanh operator
shared_ptr<Unit> operator_tanh(const shared_ptr<Unit>& a) {
    // we use the cmath C++ library
    auto output_data = tanh(a->data);
    auto out = make_shared<Unit>(output_data, vector<shared_ptr<Unit>>{a});

    // the derivative of tanh(x) = 1 - tanh²(x)
    out->backward = [a,out] () mutable {
        double t2 = tanh(a->data);
        a->grad += (1 - t2*t2) * out->grad;
    };
    return out;
}

std::shared_ptr<Tensor> linear_activation(std::shared_ptr<Tensor> x) {
    return x; 
}

shared_ptr<Tensor> operator+(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    auto out = a->compute_binary_op(b, BinaryOp::ADD);

    out->set_parents({a, b});
    Tensor* raw_out = out.get();

    out->set_backward([a, b, raw_out]() {
        a->init_grad();
        b->init_grad();
        Device::get()->accumulate_grad(a, raw_out->getGrad());
        Device::get()->accumulate_grad(b, raw_out->getGrad());
    });

    return out;
}

shared_ptr<Tensor> matmul(shared_ptr<Tensor> a, shared_ptr<Tensor> b, bool require_grad) {

    auto out = a->compute_matmul(b);
    out->set_parents({a, b}); 

    Tensor* raw_out = out.get();

    out->set_backward([a, b, raw_out]() {
        a->init_grad();
        b->init_grad();
        auto grad_output = raw_out->getGrad();
        
        auto b_T = transpose_view(b);
        auto da = matmul(grad_output, b_T, false);
        Device::get()->accumulate_grad(a, da);

        auto a_T = transpose_view(a);
        auto db = matmul(a_T, grad_output, false);
        
        Device::get()->accumulate_grad(b, db);
    });

    return out;
}


std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { 
    
    auto out = a->compute_binary_op(b, BinaryOp::MUL);

    out->set_parents({a, b});
    Tensor* raw_out = out.get();

    out->set_backward([a, b, raw_out]() { 
        a->init_grad();
        b->init_grad();

        auto grad_a_part = raw_out->getGrad() * b; 
        Device::get()->accumulate_grad(a, grad_a_part);

        auto grad_b_part = raw_out->getGrad() * a;
        Device::get()->accumulate_grad(b, grad_b_part);
    });

    return out;
}

shared_ptr<Tensor> operator*(shared_ptr<Tensor> a, float scalar) {
    vector<float> output_data(element_vector_product(a->getShape()));
    float* input_data = a->getData();
    size_t size = a->getSize();

    for (size_t i = 0; i < size; i++) {
        output_data[i] = input_data[i] * scalar;
    }

    auto result = Tensor::create(a->getShape(), output_data, {a});
    Tensor* raw_result = result.get();

    result->set_backward([a, raw_result, scalar]() {
        a->init_grad();

        float* grad_input = a->getGrad()->getData();
        float* grad_output = raw_result->getGrad()->getData();
        size_t size = a->getSize();

        for (size_t i = 0; i < size; i++) {
            grad_input[i] += grad_output[i] * scalar;
        }
    });

    return result;
}

shared_ptr<Tensor> operator+(shared_ptr<Tensor> a, float scalar) {
    vector<float> output_data(a->getSize());
    float* input_data = a->getData();
    size_t size = a->getSize();

    for (size_t i = 0; i < size; i++) {
        output_data[i] = input_data[i] + scalar;
    }

    auto result = Tensor::create(a->getShape(), output_data, {a});
    Tensor* raw_result = result.get();

    result->set_backward([a, raw_result]() {
        a->init_grad();

        float* grad_input = a->getGrad()->getData();
        float* grad_output = raw_result->getGrad()->getData();
        size_t size = a->getSize();

        for (size_t i = 0; i < size; i++) {
            grad_input[i] += grad_output[i];
        }
    });

    return result;
}

shared_ptr<Tensor> operator-(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    auto out = a->compute_binary_op(b, BinaryOp::SUB);
    out->set_parents({a, b});
    Tensor* raw_out = out.get();

    out->set_backward([a, b, raw_out]() {
        a->init_grad();
        b->init_grad();
        Device::get()->accumulate_grad(a, raw_out->getGrad());
        Device::get()->accumulate_grad(b, raw_out->getGrad() * -1.0f);
    });

    return out;
}

std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, float scalar) {
    return a * (1.0f / scalar);
}

std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b){
    auto out = a->compute_binary_op(b, BinaryOp::DIV);
    out->set_parents({a, b});
    Tensor* raw_out = out.get();

    out->set_backward([a, b, raw_out]() {
        a->init_grad();
        b->init_grad();

        auto da = raw_out->getGrad() / b;
        Device::get()->accumulate_grad(a, da);

        auto b_sq = b * b;
        auto neg_a = a * -1.0f;
        auto db = (raw_out->getGrad() * neg_a) / b_sq;
        Device::get()->accumulate_grad(b, db);
    });

    return out;
}

shared_ptr<Tensor> sum(shared_ptr<Tensor> a, int dim = 0) {
    auto out = a->compute_reduce_op(dim, ReduceOp::SUM);
    out->set_parents({a});
    Tensor* raw_out = out.get();

    out->set_backward([a, raw_out, dim]() {
        a->init_grad();

        float* grad_input = a->getGrad()->getData();
        float* grad_output = raw_out->getGrad()->getData(); 
        
        const auto& in_shape = a->getShape();
        const auto& out_strides = raw_out->getStrides();
        size_t total_elements = a->getSize();

        for (size_t i = 0; i < total_elements; i++) {
            int current_idx = i;
            int out_idx = 0;
            
            for (int d = in_shape.size() - 1; d >= 0; --d) {
                int coord = current_idx % in_shape[d];
                current_idx /= in_shape[d];
                
                if (d != dim) {
                    out_idx += coord * out_strides[d];
                }
            }
            
            grad_input[i] += grad_output[out_idx];
        }
    });

    return out;
}

shared_ptr<Tensor> max(shared_ptr<Tensor> a, int dim = 0) {
    auto out = a->compute_reduce_op(dim, ReduceOp::MAX);
    out->set_parents({a});
    Tensor* raw_out = out.get();

    out->set_backward([a, raw_out, dim]() {
        a->init_grad();

        float* grad_in = a->getGrad()->getData();
        float* grad_out = raw_out->getGrad()->getData(); 
        float* val_in = a->getData();
        float* val_out = raw_out->getData();
        
        const auto& in_shape = a->getShape();
        const auto& out_strides = raw_out->getStrides();
        size_t total_elements = a->getSize();

        for (size_t i = 0; i < total_elements; i++) {
            int current_idx = i;
            int out_idx = 0;
            
            for (int d = in_shape.size() - 1; d >= 0; --d) {
                int coord = current_idx % in_shape[d];
                current_idx /= in_shape[d];
                
                if (d != dim) {
                    out_idx += coord * out_strides[d];
                }
            }
            
            if (val_in[i] == val_out[out_idx]) {
                grad_in[i] += grad_out[out_idx];
            }
        }
    });

    return out;
}

shared_ptr<Tensor> argmax(shared_ptr<Tensor> a, int dim = 0){
    auto out = a->compute_reduce_op(dim, ReduceOp::ARGMAX);
    out->set_parents({a});
    Tensor* raw_out = out.get();
    
    out->set_backward([a,raw_out]() {
        (void) a; (void) raw_out;
    });

    return out;
}

shared_ptr<Tensor> transpose_view(shared_ptr<Tensor> a) {
    // we use a view copy of the tensor passing its data directly
    auto result = Tensor::create(a->getShape(), {}, {a}); 
    // manually mapping the shared data pointer
    result = a->view_to_gemm(false); // safe fallback or adjust memory manually
    
    // override shape and strides for transposition
    std::vector<int> new_shape = a->getShape();
    std::vector<int> new_strides = a->getStrides();
    int dims = a->getDimension();

    if (dims == 2) {
        std::swap(new_shape[0], new_shape[1]);
        std::swap(new_strides[0], new_strides[1]);
    } else if (dims == 3) {
        std::swap(new_shape[1], new_shape[2]);
        std::swap(new_strides[1], new_strides[2]);
    } else {
        throw std::runtime_error("transpose_view: dimensiones no soportadas");
    }

    result->set_shape(new_shape);
    result->set_strides(new_strides);

    Tensor* raw_result = result.get();

    result->set_backward([a, raw_result]() {
        a->init_grad();
        auto grad_T = transpose_view(raw_result->getGrad());
        Device::get()->accumulate_grad(a, grad_T);
    });

    return result;
}

shared_ptr<Tensor> relu(shared_ptr<Tensor> a){
    
    auto out = a->compute_unary_op(UnaryOp::RELU);
    out->set_parents({a});
    Tensor* raw_out = out.get();
    
    out->set_backward([a, raw_out]() {
        a->init_grad();

        float* grad_input_data = a->getGrad()->getData(); 
        float* grad_output_data = raw_out->getGrad()->getData();
        float* input_val = a->getData();

        size_t size = a->getSize();
        for(size_t i=0; i<size; i++) {
            float local_deriv = (input_val[i] > 0) ? 1.0f : 0.0f;
            grad_input_data[i] += grad_output_data[i] * local_deriv;
        }
    });

    return out;
}

shared_ptr<Tensor> exp(shared_ptr<Tensor> a){
    auto out = a->compute_unary_op(UnaryOp::EXP);
    out->set_parents({a});
    Tensor* raw_out = out.get();

    out->set_backward([a,raw_out]() {
        a->init_grad();

        float* grad_input_data = a->getGrad()->getData();
        float* grad_output_data = raw_out->getGrad()->getData();
        float* out_data = raw_out->getData();
        size_t size = raw_out->getSize();
        for(size_t i=0; i<size;i++){
            grad_input_data[i]  += grad_output_data[i] * out_data[i];
        }
    });
    
    return out;
}

shared_ptr<Tensor> sqrt(shared_ptr<Tensor> a) {
    auto out = a->compute_unary_op(UnaryOp::SQRT);
    out->set_parents({a});
    Tensor* raw_out = out.get();

    out->set_backward([a, raw_out]() {
        a->init_grad();
        
        auto out_shared = raw_out->shared_from_this();
        auto grad_sqrt = (raw_out->getGrad() * 0.5f) / out_shared;
        Device::get()->accumulate_grad(a, grad_sqrt);
    });

    return out;
}

shared_ptr<Tensor> log(shared_ptr<Tensor> a){
    
    auto out = a->compute_unary_op(UnaryOp::LOG);
    out->set_parents({a});
    Tensor* raw_out = out.get();

    out->set_backward([a,raw_out]() {
        a->init_grad();

        float* grad_input_data = a->getGrad()->getData();
        float* grad_output_data = raw_out->getGrad()->getData();
        float* out_data = raw_out->getData();
        float* in_data = a->getData();
        size_t size = raw_out->getSize();
        for(size_t i=0; i<size;i++){
            grad_input_data[i]  += grad_output_data[i] * (1.0f / (in_data[i] + 1e-8f));
        }
    });
    
    return out;
}

shared_ptr<Tensor> sigmoid(shared_ptr<Tensor> a){
    auto out = a->compute_unary_op(UnaryOp::SIGMOID);
    out->set_parents({a});
    Tensor* raw_out = out.get();

    out->set_backward([a, raw_out]() {
        a->init_grad();

        float* grad_input_data = a->getGrad()->getData(); 
        float* grad_output_data = raw_out->getGrad()->getData();
        float* output_val = raw_out->getData();

        size_t size = a->getSize();
        for(size_t i=0; i<size; i++) {
            float local_deriv = output_val[i] * (1.0f - output_val[i]);
            grad_input_data[i] += grad_output_data[i] * local_deriv;
        }
    });

    return out;
}

shared_ptr<Tensor> tanh(shared_ptr<Tensor> a){
    auto out = a->compute_unary_op(UnaryOp::TANH);
    out->set_parents({a});
    Tensor* raw_out = out.get();
        
    out->set_backward([a, raw_out]() {
        a->init_grad();

        float* grad_input_data = a->getGrad()->getData(); 
        float* grad_output_data = raw_out->getGrad()->getData();
        float* output_val = raw_out->getData();

        size_t size = a->getSize();
        for(size_t i=0; i<size; i++) {
            float local_deriv = 1.0f - (output_val[i] * output_val[i]);
            grad_input_data[i] += grad_output_data[i] * local_deriv;
        }
    });

    return out;
}

shared_ptr<Tensor> gather(shared_ptr<Tensor> w, shared_ptr<Tensor> ind) {
    auto out = w->compute_gather(ind);

    out->set_parents({w});
    Tensor* raw_out = out.get();

    out->set_backward([w, ind, raw_out] {
        w->init_grad();
        w->getGrad()->compute_scatter_add(ind, raw_out->getGrad());
    });

    return out;
}

std::shared_ptr<Tensor> conv2d(std::shared_ptr<Tensor>  input, std::shared_ptr<Tensor>  w, int stride, int padding) {
    auto out = input->compute_conv2d(w,stride,padding);
    out->set_parents({input, w});
    Tensor* raw_out = out.get();

    out->set_backward([input, w, raw_out, stride, padding]() {
        input->init_grad();
        w->init_grad();

        int n_batch = input->getShape()[0];
        int c_in = input->getShape()[1];
        int h_in = input->getShape()[2];
        int w_in = input->getShape()[3];

        int c_out = w->getShape()[0];
        int k_h = w->getShape()[2];
        int k_w = w->getShape()[3];

        int h_out = raw_out->getShape()[2];
        int w_out = raw_out->getShape()[3];

        int workspace_size = c_in * k_h * k_w * h_out * w_out;
        float* col_buffer = new float[workspace_size];
        
        std::vector<float> w_data(w->getData(), w->getData() + w->getSize());
        auto w_2d = Tensor::create({c_out, c_in * k_h * k_w}, w_data);
        auto w_T = transpose_view(w_2d);
                
        for(int b = 0; b < n_batch; b++) {
            auto x_b = input->batch_view(b, true);
            auto dY_b = raw_out->getGrad()->batch_view(b, true);
            auto dx_b = input->getGrad()->batch_view(b,true);

            std::vector<float> dy_data(dY_b->getData(), dY_b->getData() + dY_b->getSize());
            auto dY_b_2d = Tensor::create({c_out, h_out * w_out}, dy_data);

            x_b->im2col(col_buffer, h_out, w_out, k_h, k_w, stride, padding);

            std::vector<float> col_data(col_buffer, col_buffer + workspace_size);
            auto X_col = Tensor::create({c_in * k_h * k_w, h_out * w_out}, col_data);
            X_col->set_strides({1, c_in * k_h * k_w});

            auto X_col_T = transpose_view(X_col);
            auto dW_b = matmul(dY_b_2d, X_col_T, false);

            dW_b->set_shape(w->getShape());
            dW_b->set_strides(w->getStrides());
            Device::get()->accumulate_grad(w, dW_b);

            auto dX_col = matmul(w_T, dY_b_2d, false);
            dx_b->col2im(dX_col->getData(), h_out, w_out, k_h, k_w, stride, padding);
        }
        
        delete[] col_buffer;
    });

    return out;
}

shared_ptr<Tensor> flatten(shared_ptr<Tensor> a) {
    auto out = a->compute_flatten();
    
    out->set_parents({a});
    Tensor* raw_out = out.get();
    
    out->set_backward([a, raw_out]() {
        a->init_grad();
        
        float* grad_in = a->getGrad()->getData();
        float* grad_out = raw_out->getGrad()->getData();
        
        for (size_t i = 0; i < a->getSize(); i++) {
            grad_in[i] += grad_out[i];
        }
    });
    
    return out;
}