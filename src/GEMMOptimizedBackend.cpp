#include "backend.h"
#include "types.h"
#include <cmath>
#include <cstring>
#include <immintrin.h>
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
void GEMMOptimizedBackend::binary(const TensorInfo& a, const TensorInfo& b, TensorInfo& out, BinaryOp op){

    size_t out_size = out.size;
    
    // Pointers
    float* out_data = out.data;
    const float* a_data = a.data;
    const float* b_data = b.data;

    // we just need to iterate index-by-index mapping to coordinates.
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
void GEMMOptimizedBackend::unary(const TensorInfo& a, TensorInfo& out, UnaryOp op){
    
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




// This implementation is fully based on GOTO paper 
// ("Anatomy of High-Performance Matrix Multiplication") fig 10.
// is done on a AMD with AVX2 and cache size of
// L1 = 32KiB 
// l2 = 512KiB (512 * 1024) = 524.288 bytes
// l3 = 16MiB
// because LearnTorch is row-major, we will implement GEPP + GEPB

// B must occupy half of l2 -> 256 * 256 * 4 = 262.144
const int KC = 256;
const int NC = 256;

// A
const int MC = 1024;

// for avx2 registers (256 bits -> 8 floats)
const int MR = 4;
const int NR = 16;

// used to pack an B block
void pack_block(float* target, const float* source, int current_k, int current_n, int stride_k, int stride_n) {
    int idx = 0;
    for(int k = 0; k < current_k; k++) {
        for (int j = 0; j < current_n; j++) {
            target[idx++] = source[k * stride_k + j * stride_n];
        }
    }
};

// micro kernel for gemm
static void micro_kernel_avx2(const float* A, const float* B, float* C,
                               int k, int mr, int nr, int stride_b, int stride_out) {

    // in case mr and nr are smaller than the specified block size
    if (mr < MR || nr < NR) {
        for (int i = 0; i < mr; i++)
            for (int j = 0; j < nr; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++)
                    sum += A[i * k + p] * B[p * stride_b + j];
                C[i * stride_out + j] += sum;
            }
        return;
    }

    // _m256 references a YMM 256 bits register (8 floats)
    // we got a block of 4x16 floats, so we initiate 8 accumulators
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

    for (int p = 0; p < k; p++) {
        // load a B row (16 floats) in two registers
        __m256 b0 = _mm256_loadu_ps(B + p * stride_b);
        __m256 b1 = _mm256_loadu_ps(B + p * stride_b + 8);

        // replicate a single float in all the register
        __m256 a0 = _mm256_broadcast_ss(A + 0 * k + p);
        // we do the add and set it to the c tile accumulator
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        // same process
        __m256 a1 = _mm256_broadcast_ss(A + 1 * k + p);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        __m256 a2 = _mm256_broadcast_ss(A + 2 * k + p);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        __m256 a3 = _mm256_broadcast_ss(A + 3 * k + p);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }

    // loads the current C mem in YMM register, then we add the 
    // accumulators and write them back to mem
    _mm256_storeu_ps(C + 0*stride_out,     _mm256_add_ps(_mm256_loadu_ps(C + 0*stride_out),     c00));
    _mm256_storeu_ps(C + 0*stride_out + 8, _mm256_add_ps(_mm256_loadu_ps(C + 0*stride_out + 8), c01));
    _mm256_storeu_ps(C + 1*stride_out,     _mm256_add_ps(_mm256_loadu_ps(C + 1*stride_out),     c10));
    _mm256_storeu_ps(C + 1*stride_out + 8, _mm256_add_ps(_mm256_loadu_ps(C + 1*stride_out + 8), c11));
    _mm256_storeu_ps(C + 2*stride_out,     _mm256_add_ps(_mm256_loadu_ps(C + 2*stride_out),     c20));
    _mm256_storeu_ps(C + 2*stride_out + 8, _mm256_add_ps(_mm256_loadu_ps(C + 2*stride_out + 8), c21));
    _mm256_storeu_ps(C + 3*stride_out,     _mm256_add_ps(_mm256_loadu_ps(C + 3*stride_out),     c30));
    _mm256_storeu_ps(C + 3*stride_out + 8, _mm256_add_ps(_mm256_loadu_ps(C + 3*stride_out + 8), c31));
}




void gepb(float* packed_B, const float* packed_A,
          int current_k, int current_n, int M,
          float* out_panel_start, const float* b_block_start,
          int stride_k_b, int stride_n_b, int stride_out) {

    pack_block(packed_B, b_block_start, current_k, current_n, stride_k_b, stride_n_b);

    for (int i = 0; i < M; i += MR) {
        int current_m = std::min(MR, M - i); // last iteration
        // A packed is just an array with k_current columns
        const float* a_micro_start = packed_A + (i * current_k);

        for (int j = 0; j < current_n; j += NR) {
            int current_nr = std::min(NR, current_n - j); // last iteration
            
            // inside C n panel, we extract an NR block of a row
            float* out_micro_start = out_panel_start + (i * stride_out) + j;
            // we set the exact start of the NR block
            const float* b_micro_start = packed_B + j; 

            micro_kernel_avx2(
                a_micro_start, 
                b_micro_start, 
                out_micro_start, 
                current_k, current_m, current_nr, current_n, stride_out
            );
        }
    }
}

// used to pack an A panel of MC,KC dimensions into a contiguous memory
void pack_panel(float* target, const float* source, int current_k, int m, int stride_m, int stride_k) {
    int idx = 0;
    for(int i = 0; i < m; i++) {
        for (int k = 0; k < current_k; k++) {
            target[idx++] = source[i * stride_m + k * stride_k];
        }
    }
};

void GEMMOptimizedBackend::gemm(const TensorInfo& a, const TensorInfo& b, TensorInfo& out) {
    auto n_batch = out.shape[0];
    auto M = out.shape[1];
    auto N = out.shape[2];
    auto K = a.shape[2];
    float* packed_B = (float*) std::aligned_alloc(64, KC * NC * sizeof(float));
    float* packed_A = (float*) std::aligned_alloc(64, M * KC * sizeof(float));

    // We iterate per batch 
    for(int b_idx = 0; b_idx < n_batch; b_idx++){
        const float* batch_a = a.data + b_idx * a.strides[0];
        const float* batch_b = b.data + b_idx * b.strides[0];
        float* batch_out = out.data + b_idx * out.strides[0];
        std::memset(batch_out, 0, M * N * sizeof(float));

        // first loop por GEPP, we slice by K
        for(int k = 0; k < K; k += KC){
            int current_k = std::min(KC, K - k);
            const float* a_panel_start = batch_a + k * a.strides[2];
            // pack A: stride between rows = a.strides[1], stride along K = a.strides[2]
            pack_panel(packed_A, a_panel_start, current_k, M, a.strides[1], a.strides[2]);
            // GEPP inside loop
            for(int n = 0; n < N; n += NC){
                int current_n = std::min(NC, N - n);
                const float* b_block_start = batch_b + (k * b.strides[1]) + n * b.strides[2];
                float* out_panel_start = batch_out + n;

                gepb(packed_B, packed_A, current_k, current_n, M,
                     out_panel_start, b_block_start, b.strides[1], b.strides[2], out.strides[1]);
            };
        }
    }

    std::free(packed_A);
    std::free(packed_B);
}

float* GEMMOptimizedBackend::alloc(size_t size) { return new float[size]; }
void GEMMOptimizedBackend::free(float* ptr) { delete[] ptr; }
void GEMMOptimizedBackend::set(float* ptr, float value, size_t size) { 
   for(size_t i=0; i<size; i++) ptr[i] = value;
}

void GEMMOptimizedBackend::reduce(const TensorInfo& in, TensorInfo& out, int dim, ReduceOp op) {
 
    size_t out_size = out.size;
    int reduction_size = in.shape[dim]; // number of elements to reduce
    
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

        for (int j = 0; j < reduction_size; j++) { // iterate per element to reduce
            float val = in.data[in_offset_base + j * stride_dim];
            switch (op) {
                case ReduceOp::SUM: acc += val; break;
                case ReduceOp::MAX: if (val > acc) acc = val; break;
                case ReduceOp::MIN: if (val < acc) acc = val; break;
                case ReduceOp::ARGMAX: if (val > max_val) {
                        max_val = val;
                        best_idx = j;
                    }
                    break;
                default: break;
            }
        }

        if (op == ReduceOp::ARGMAX) {
            acc = (float)best_idx;
        }
        
        // set the value in the real memory data index
        int out_real_index = getIndex(i, out);
        out.data[out_real_index] = acc;
    }
}

// Unbroadcast function to accumulate gradients correctly
void GEMMOptimizedBackend::accumulate_grad(shared_ptr<Tensor> param, shared_ptr<Tensor> incoming_grad) {

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
void GEMMOptimizedBackend::gather(const TensorInfo& w, const TensorInfo& indexes, TensorInfo& out) {
    int embed_dim = w.shape[w.dim - 1];
    int vocab_size = w.shape[0];

    int w_stride_row = w.strides[0];
    int w_stride_col = w.strides[w.dim - 1];

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

void GEMMOptimizedBackend::scatter_add(const TensorInfo& indexes, const TensorInfo& incoming_grad, const TensorInfo& w_grad) {
    int embed_dim = w_grad.shape[w_grad.dim - 1];
    int vocab_size = w_grad.shape[0];

    int w_stride_row = w_grad.strides[0];
    int w_stride_col = w_grad.strides[w_grad.dim - 1];

    for(size_t i = 0; i < indexes.size; i++) {
        int idx = getIndex(i, indexes);
        int token_id = static_cast<int>(indexes.data[idx]);

        if(token_id >= 0 && token_id < vocab_size) {
            const float* grad_row = incoming_grad.data + (i * embed_dim);

            for(int d = 0; d < embed_dim; d++) {
                int w_offset = (token_id * w_stride_row) + (d * w_stride_col);
                w_grad.data[w_offset] += grad_row[d];
            }
        }
    }
}