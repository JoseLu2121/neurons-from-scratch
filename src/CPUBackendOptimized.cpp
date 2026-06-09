#include "backend.h"
#include "types.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "utils.h"

using namespace std;

// Get memory index given one
inline int getIndex(int index, const TensorInfo& tensor){
    int offset = 0;
    int current_index = index;
    // We iterate each dimension from the right
    for(int d = tensor.dim - 1; d >= 0 ; --d){
        int relative_position = current_index % tensor.shape[d]; // Get the relative position from the actual dim
        offset += relative_position * tensor.strides[d]; // Apply stride in case of broadcast
        current_index /= tensor.shape[d]; // We establish the current index of the next dimension
    }

    return offset;

}

// A basic binary operation of two tensors
void CPUBackendOptimized::binary(const TensorInfo& a, const TensorInfo& b, TensorInfo& out, BinaryOp op){

    size_t out_size = out.size;
    
    // Pointers
    float* out_data = out.data;
    const float* a_data = a.data;
    const float* b_data = b.data;

    // optimized way checking if tensors are contiguous
    if(is_contiguous(a) && is_contiguous(b) && is_contiguous(out)) {

        switch(op)
            {
                case BinaryOp::ADD :
                #pragma omp simd
                for(size_t i = 0; i < out_size; i++) out_data[i] = a_data[i] + b_data[i]; break;

                case BinaryOp::MUL :
                #pragma omp simd
                for(size_t i = 0; i < out_size; i++) out_data[i] = a_data[i] * b_data[i]; break;

                case BinaryOp::SUB :
                #pragma omp simd
                for(size_t i = 0; i < out_size; i++) out_data[i] = a_data[i] - b_data[i]; break;

                case BinaryOp::DIV :
                #pragma omp simd
                for(size_t i = 0; i < out_size; i++) out_data[i] = a_data[i] / b_data[i]; break;

                case BinaryOp::POW : 
                #pragma omp simd
                for(size_t i = 0; i < out_size; i++) out_data[i] = std::pow(a_data[i],b_data[i]); break;

            }
        return;
    }


    // we just need to iterate index-by-index mapping to coordinates.
    #pragma omp parallel for
    for(size_t i = 0; i < out_size; i++){ 
        int a_index = getIndex(i, a);
        int b_index = getIndex(i, b);
        
        // Get values
        float a_val = a_data[a_index];
        float b_val = b_data[b_index];
        float out_val = 0.0f;

        // Switch for each case of operation
        switch (op)
        {
            case BinaryOp::ADD : out_val = a_val + b_val; break;
            case BinaryOp::MUL : out_val = a_val * b_val; break;
            case BinaryOp::SUB : out_val = a_val - b_val; break;
            case BinaryOp::DIV : out_val = a_val / b_val; break;
            case BinaryOp::POW : out_val = std::pow(a_val,b_val); break;
        }
        
        out_data[i] = out_val;
    };

};

// A basic unary operation of a tensor
void CPUBackendOptimized::unary(const TensorInfo& a, TensorInfo& out, UnaryOp op){
    
    // Pointers
    float* out_data = out.data;
    const float* a_data = a.data;

    // optimized way checking if tensors are contiguous
    if(is_contiguous(a) && is_contiguous(out)) {
    switch (op) {
        case UnaryOp::RELU:
            #pragma omp parallel for simd
            for(size_t i = 0; i < out.size; i++) out_data[i] = (a_data[i] > 0.0f) ? a_data[i] : 0.0f;
            break;
        case UnaryOp::SIGMOID:
            #pragma omp parallel for simd
            for(size_t i = 0; i < out.size; i++) out_data[i] = 1.0f / (1.0f + std::exp(-a_data[i]));
            break;
        case UnaryOp::TANH:
            #pragma omp parallel for simd
            for(size_t i = 0; i < out.size; i++) out_data[i] = std::tanh(a_data[i]);
            break;
        case UnaryOp::EXP:
            #pragma omp parallel for simd
            for(size_t i = 0; i < out.size; i++) out_data[i] = std::exp(a_data[i]);
            break;
        case UnaryOp::LOG:
            #pragma omp parallel for simd
            for(size_t i = 0; i < out.size; i++) out_data[i] = std::log(a_data[i]);
            break;
        case UnaryOp::NEG:
            #pragma omp parallel for simd
            for(size_t i = 0; i < out.size; i++) out_data[i] = -a_data[i];
            break;
        case UnaryOp::SQRT:
            #pragma omp parallel for simd
            for(size_t i = 0; i < out.size; i++) out_data[i] = std::sqrt(a_data[i]);
            break;
    }
    return;
}
        
    #pragma omp parallel for
    for(size_t i = 0; i < out.size; i++){

        int a_index = getIndex(i,a);
        int out_index = getIndex(i,out);
        
        float a_val = a.data[a_index];
        float out_val = 0.0f;
        
        switch (op)
        {
            case UnaryOp::RELU : out_val = (a_val > 0.0f) ? a_val : 0.0f; break;
            case UnaryOp::SIGMOID : out_val = 1.0f / (1.0f + std::exp(-a_val)); break;
            case UnaryOp::TANH: out_val = std::tanh(a_val); break;
            case UnaryOp::EXP : out_val = std::exp(a_val); break;
            case UnaryOp::LOG : out_val = std::log(a_val); break;
            case UnaryOp::NEG : out_val = -a_val; break;
            case UnaryOp::SQRT: out_val = std::sqrt(a_val); break;
        }
        
        out.data[out_index] = out_val;
    }
}



void CPUBackendOptimized::gemm(const TensorInfo& a, const TensorInfo& b, TensorInfo& out) {
    auto n_batch = out.shape[0];
    auto M = out.shape[1];
    auto N = out.shape[2];
    auto K = a.shape[2];

    if (is_contiguous(a) && is_contiguous(b) && is_contiguous(out)) {
        for(int b_idx = 0; b_idx < n_batch; b_idx++) {
            const float* batch_a = a.data + b_idx * (M * K);
            const float* batch_b = b.data + b_idx * (K * N);
            float* batch_out = out.data + b_idx * (M * N);

            #pragma omp parallel for
            for(int m = 0; m < M; m++) {
                for(int n = 0; n < N; n++) {
                    batch_out[m * N + n] = 0.0f;
                }
                
                for(int k = 0; k < K; k++) {
                    float val_a = batch_a[m * K + k];
                
                    #pragma omp simd
                    for(int n = 0; n < N; n++) {
                        batch_out[m * N + n] += val_a * batch_b[k * N + n];
                    }
                }
            }
        }
        return;
    }

    // if b is transposed
    if (is_contiguous(a) && is_contiguous(out) && b.strides[1] == 1 && b.strides[2] == K) {
        for(int b_idx = 0; b_idx < n_batch; b_idx++) {
            const float* batch_a = a.data + b_idx * (M * K);
            const float* batch_b = b.data + b_idx * (K * N);
            float* batch_out = out.data + b_idx * (M * N);

            #pragma omp parallel for
            for(int m = 0; m < M; m++) {
                for(int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    #pragma omp simd reduction(+:sum)
                    for(int k = 0; k < K; k++) {
                        sum += batch_a[m * K + k] * batch_b[n * K + k];
                    }
                    batch_out[m * N + n] = sum;
                }
            }
        }
        return;
    }

    auto a_col_strides = a.strides[2];
    auto b_row_strides = b.strides[1];
    auto a_row_strides = a.strides[1];
    auto b_col_strides = b.strides[2];
    
    for(int b_idx = 0; b_idx < n_batch; b_idx++) {
        const float* batch_a = a.data + b_idx * a.strides[0];
        const float* batch_b = b.data + b_idx * b.strides[0];
        float* batch_out = out.data + b_idx * out.strides[0];

        #pragma omp parallel for
        for(int m = 0; m < M; m++) {
            for(int k = 0; k < K; k++) {
                float val_a = batch_a[m * a_row_strides + k * a_col_strides];
                
                for(int n = 0; n < N; n++) {
                    float val_b = batch_b[k * b_row_strides + n * b_col_strides];
                    batch_out[m * out.strides[1] + n * out.strides[2]] += val_a * val_b;
                }
            }
        }
    }
}

float* CPUBackendOptimized::alloc(size_t size) { return new float[size]; }
void CPUBackendOptimized::free(float* ptr) { delete[] ptr; }
void CPUBackendOptimized::set(float* ptr, float value, size_t size) { 
   for(size_t i=0; i<size; i++) ptr[i] = value;
}

void CPUBackendOptimized::reduce(const TensorInfo& in, TensorInfo& out, int dim, ReduceOp op) {
 
    size_t out_size = out.size;
    int reduction_size = in.shape[dim]; // number of elements to reduce


    if (is_contiguous(in) && is_contiguous(out) && dim == in.dim - 1) {
        switch (op) {
            case ReduceOp::SUM:
                #pragma omp parallel for
                for (size_t i = 0; i < out_size; i++) {
                    float acc = 0.0f;
                    const float* in_row = in.data + i * reduction_size;
                    #pragma omp simd reduction(+:acc)
                    for (int j = 0; j < reduction_size; j++) acc += in_row[j];
                    out.data[i] = acc;
                }
                break;
            case ReduceOp::MAX:
                #pragma omp parallel for
                for (size_t i = 0; i < out_size; i++) {
                    const float* in_row = in.data + i * reduction_size;
                    float acc = reduction_size > 0 ? in_row[0] : -1e9f;
                    #pragma omp simd reduction(max:acc)
                    for (int j = 0; j < reduction_size; j++) {
                        if (in_row[j] > acc) acc = in_row[j];
                    }
                    out.data[i] = acc;
                }
                break;
            case ReduceOp::MIN:
                #pragma omp parallel for
                for (size_t i = 0; i < out_size; i++) {
                    const float* in_row = in.data + i * reduction_size;
                    float acc = reduction_size > 0 ? in_row[0] : 1e9f;
                    #pragma omp simd reduction(min:acc)
                    for (int j = 0; j < reduction_size; j++) {
                        if (in_row[j] < acc) acc = in_row[j];
                    }
                    out.data[i] = acc;
                }
                break;
            case ReduceOp::ARGMAX:
                #pragma omp parallel for
                for (size_t i = 0; i < out_size; i++) {
                    const float* in_row = in.data + i * reduction_size;
                    float max_val = -1e9f;
                    int best_idx = 0;
                    for (int j = 0; j < reduction_size; j++) {
                        if (in_row[j] > max_val) {
                            max_val = in_row[j];
                            best_idx = j;
                        }
                    }
                    out.data[i] = (float)best_idx;
                }
                break;
        }
        return;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < out_size; i++) {
        
        int current_idx = i;
        int in_offset_base = 0;

        for (int d = out.dim - 1; d >= 0; --d) {
            int coord = current_idx % out.shape[d];
            current_idx /= out.shape[d];
            in_offset_base += coord * in.strides[d];
        }
        float acc = 0.0f;
        float max_val = -1e-9f;
        int best_idx = 0;
        switch (op) { // initial op default values
            case ReduceOp::SUM: acc = 0.0f; break;
            case ReduceOp::MAX: acc = -1e9f; break; 
            case ReduceOp::MIN: acc = 1e9f; break;
            case ReduceOp::ARGMAX: max_val = -1e9f; 
                                   best_idx = 0; break;
            default: break;
        }
        // we prefer adding the first value of the dimension as the default 
        if (reduction_size > 0 && op == ReduceOp::MAX) acc = in.data[in_offset_base]; 
        if (reduction_size > 0 && op == ReduceOp::MIN) acc = in.data[in_offset_base];
        if (op == ReduceOp::ARGMAX) max_val = in.data[in_offset_base];

        // the stride of the dimension we reduce
        int stride_dim = in.strides[dim];

        switch (op) {
            case ReduceOp::SUM:
                for (int j = 0; j < reduction_size; j++) acc += in.data[in_offset_base + j * stride_dim];
                break;
            case ReduceOp::MAX:
                for (int j = 0; j < reduction_size; j++) {
                    float val = in.data[in_offset_base + j * stride_dim];
                    if (val > acc) acc = val;
                }
                break;
            case ReduceOp::MIN:
                for (int j = 0; j < reduction_size; j++) {
                    float val = in.data[in_offset_base + j * stride_dim];
                    if (val < acc) acc = val;
                }
                break;
            case ReduceOp::ARGMAX:
                for (int j = 0; j < reduction_size; j++) {
                    float val = in.data[in_offset_base + j * stride_dim];
                    if (val > max_val) {
                        max_val = val;
                        best_idx = j;
                    }
                }
                acc = (float)best_idx;
                break;
        }
        
        // set the value in the real memory data index
        int out_real_index = getIndex(i, out);
        out.data[out_real_index] = acc;
    }
}



// Unbroadcast function to accumulate gradients correctly
void CPUBackendOptimized::accumulate_grad(shared_ptr<Tensor> param, shared_ptr<Tensor> incoming_grad) {

    float* p_data = param->getGrad()->getData(); // Buffer to accumulate gradients
    float* g_data = incoming_grad->getData();  // Incoming gradient data that needs to be unbroadcasted

    const auto& p_shape = param->getShape(); 
    const auto& p_strides = param->getStrides(); 
    const auto& g_shape = incoming_grad->getShape(); 
   
    int g_dims = g_shape.size();
    int p_dims = p_shape.size();
    int dim_diff = g_dims - p_dims; 

    // Coordinates of the incoming gradient
    vector<int> g_coords(g_dims, 0); // ex: if g_dims = 3, g_coords = {0,0,0}

    size_t total = incoming_grad->getSize();

    for (size_t idx = 0; idx < total; idx++) {

        // we calculate the offset
        int p_offset = 0;
        for (int j = 0; j < p_dims; j++) { // we iterate each dimension of the original gradient
            int g_j = j + dim_diff;        // we look up the corresponding dimension in the incoming gradient
            // if the parameter shape in this dimension is 1, we always take coordinate 0 (broadcasted)
            int coord = (p_shape[j] == 1) ? 0 : g_coords[g_j];
            p_offset += coord * p_strides[j];
        }

        // we accumulate the gradient
        p_data[p_offset] += g_data[idx];

        // We increment like an odometer ex for 3D: (0,0,0) -> (0,0,1)
        for (int d = g_dims - 1; d >= 0; d--) {
            g_coords[d]++;
            if (g_coords[d] < g_shape[d]) break;
            g_coords[d] = 0;
        }
    }
}

// Acts like a look up table
void CPUBackendOptimized::gather(const TensorInfo& w, const TensorInfo& indexes, TensorInfo& out) {
    int embed_dim = w.shape[w.dim - 1];
    int vocab_size = w.shape[0]; 

    int w_stride_row = w.strides[0];
    int w_stride_col = w.strides[w.dim - 1];

    #pragma omp parallel for
    for(size_t i = 0; i < indexes.size; i++) {
        int idx = getIndex(i, indexes);
        int token_id = static_cast<int>(indexes.data[idx]);

        if(token_id >= 0 && token_id < vocab_size) {
            float* out_row = out.data + (i * embed_dim);

            for(int d = 0; d < embed_dim; d++) {
                int w_offset = (token_id * w_stride_row) + (d * w_stride_col);
                out_row[d] = w.data[w_offset];
            }
        }
    }
}

void CPUBackendOptimized::scatter_add(const TensorInfo& indexes, const TensorInfo& incoming_grad, const TensorInfo& w_grad) {
    int embed_dim = w_grad.shape[w_grad.dim - 1];
    int vocab_size = w_grad.shape[0];

    int w_stride_row = w_grad.strides[0];
    int w_stride_col = w_grad.strides[w_grad.dim - 1];
    #pragma omp parallel for
    for(size_t i = 0; i < indexes.size; i++) {
        int idx = getIndex(i, indexes);
        int token_id = static_cast<int>(indexes.data[idx]);

        if(token_id >= 0 && token_id < vocab_size) {
            const float* grad_row = incoming_grad.data + (i * embed_dim);

            for(int d = 0; d < embed_dim; d++) {
                int w_offset = (token_id * w_stride_row) + (d * w_stride_col);
                #pragma omp atomic
                w_grad.data[w_offset] += grad_row[d];
            }
        }
    }
}

