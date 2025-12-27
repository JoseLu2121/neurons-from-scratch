#include "ops.h"
#include "utils.h"
#include <algorithm>
#include <stdexcept>
#include <cmath> 

using namespace std;


Tensor dot_scalar_product(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    int size_output = a->getShape().at(0);
    vector<float> output_data(size_output);
    

    int stride_a = a->strides[0];
    int stride_b = b->strides[0];

    for (int i = 0; i < size_output; i++) {

        float val_a = a->getData()[i * stride_a];
        float val_b = b->getData()[i * stride_b];
        output_data[i] = val_a * val_b;
    }
    return Tensor({size_output}, output_data.data(), {a,b});
}


Tensor vector_matrix_product(shared_ptr<Tensor> v, shared_ptr<Tensor> m) {
    int column_v = v->getShape().at(1);
    int column_m = m->getShape().at(1);
    

    int stride_v_col = v->strides[1]; 
    int stride_m_row = m->strides[0]; 
    int stride_m_col = m->strides[1]; 

    vector<float> output_data(column_m);
    
    for (int i = 0; i < column_m; i++) {
        float sum = 0;
        for (int j = 0; j < column_v; j++) {
            // v: índice j * su stride
            // m: fila j (j*stride_row) + columna i (i*stride_col)
            float val_v = v->getData()[j * stride_v_col];
            float val_m = m->getData()[j * stride_m_row + i * stride_m_col];
            sum += val_v * val_m;
        }
        output_data[i] = sum;
    }
    return Tensor({column_m}, output_data.data(), {v,m});
}

Tensor matrix_matrix_product(shared_ptr<Tensor> m, shared_ptr<Tensor> v) {
    int row_m = m->shape.at(0);
    int col_v = v->shape.at(1);
    int col_m = m->shape.at(1);
    
    vector<int> output_shape = {row_m, col_v};
    vector<float> output_data(product(output_shape));
    

    int m_stride_0 = m->strides[0];
    int m_stride_1 = m->strides[1];
    int v_stride_0 = v->strides[0];
    int v_stride_1 = v->strides[1];

    for (int i = 0; i < row_m; i++) {
        for (int j = 0; j < col_v; j++) {
            float sum = 0;
            for (int k = 0; k < col_m; k++) {
                // M[i, k] * V[k, j]
                float val_m = m->getData()[i * m_stride_0 + k * m_stride_1];
                float val_v = v->getData()[k * v_stride_0 + j * v_stride_1];
                sum += val_m * val_v;
            }
            output_data[i * col_v + j] = sum;
        }
    }
    return Tensor(output_shape, output_data.data(), {m,v});
}

Tensor batch_matrix_product(shared_ptr<Tensor> b, shared_ptr<Tensor> m) {
    int b_batch = b->getShape().at(0);
    int m_col = m->getShape().at(1);
    int b_row = b->getShape().at(1);
    vector<int> output_shape = {b_batch, b_row};
    if (m_col != 1) output_shape.push_back(m_col);
    vector<float> output_data(product(output_shape));
    for (int i = 0; i < b_batch; i++) {
        auto matrix_batch_i = b->getBatch(i);
        Tensor matrix_output = matrix_matrix_product(matrix_batch_i, m);
        float* matrix_data = matrix_output.getData();
        copy(matrix_data, matrix_data + b_row * m_col, output_data.begin() + b_row * m_col * i);
    }
    return Tensor(output_shape, output_data.data(), {b,m});
}

Tensor vector_batch_product(shared_ptr<Tensor> v, shared_ptr<Tensor> b){
    auto v_view = make_shared<Tensor>(*v);
    v_view->shape.insert(v_view->shape.begin(),1);
    v_view->strides.insert(v_view->strides.begin(),0);
    int b_batch = b->shape.at(0);
    int b_col = b->shape.at(2);
    vector<int> output_shape = {b_batch,b_col};
    vector<float> output_data(product(output_shape));
    for(int i = 0; i < b_batch; i++){
        auto matrix_batch_i = b->getBatch(i);
        Tensor matrix_output = vector_matrix_product(v_view, matrix_batch_i);
        float* matrix_data = matrix_output.getData();
        copy(matrix_data, matrix_data + b_col, output_data.begin() + i * b_col);
    }
    return Tensor(output_shape, output_data.data(), {v,b});
 }

Tensor matrix_batch_product(shared_ptr<Tensor> m, shared_ptr<Tensor> b) {
    int b_batch = b->getShape().at(0);
    int m_row = m->getShape().at(0);
    int b_col = b->getShape().at(2);
    vector<int> output_shape = {b_batch, m_row, b_col};
    vector<float> output_data(product(output_shape));
    for (int i = 0; i < b_batch; i++) {
        auto matrix_batch_i = b->getBatch(i);
        Tensor matrix_output = matrix_matrix_product(m, matrix_batch_i);
        float* matrix_data = matrix_output.getData();
        copy(matrix_data, matrix_data + m_row*b_col, output_data.begin() + m_row*b_col* i);
    }
    return Tensor(output_shape, output_data.data(), {m,b});
}


Tensor batch_batch_product(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    int batch = a->shape[0]; 

    int m = a->shape[1];
    int n = b->shape[2];
    
    vector<int> output_shape = {batch, m, n};
    vector<float> output_data(product(output_shape));

    for(int i=0; i<batch; i++){

        int idx_a = (i < a->shape[0]) ? i : 0;
        int idx_b = (i < b->shape[0]) ? i : 0;

        auto sub_a = a->getBatch(idx_a);
        auto sub_b = b->getBatch(idx_b);

        auto sub_res = matrix_matrix_product(sub_a, sub_b);
 
        float* src = sub_res.getData();
        float* dst = output_data.data() + i * (m*n);
        copy(src, src + (m*n), dst);
    }
    return Tensor(output_shape, output_data.data(), {a,b});
}

-
shared_ptr<Tensor> operator+(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {

    auto a_view = a->view_to_3d();
    auto b_view = b->view_to_3d();

    int out_batch = max(a_view->shape[0], b_view->shape[0]);
    int out_rows  = max(a_view->shape[1], b_view->shape[1]);
    int out_cols  = max(a_view->shape[2], b_view->shape[2]);
    vector<int> output_shape = {out_batch, out_rows, out_cols};
    
    vector<int> sA = a_view->strides;
    if (a_view->shape[0] == 1 && out_batch > 1) sA[0] = 0;
    if (a_view->shape[1] == 1 && out_rows  > 1) sA[1] = 0;
    if (a_view->shape[2] == 1 && out_cols  > 1) sA[2] = 0;

    vector<int> sB = b_view->strides;
    if (b_view->shape[0] == 1 && out_batch > 1) sB[0] = 0;
    if (b_view->shape[1] == 1 && out_rows  > 1) sB[1] = 0;
    if (b_view->shape[2] == 1 && out_cols  > 1) sB[2] = 0;

    vector<float> output_data(product(output_shape));
    float* data_a = a_view->getData();
    float* data_b = b_view->getData();

    for (int k = 0; k < out_batch; k++) {
        for (int i = 0; i < out_rows; i++) {
            for (int j = 0; j < out_cols; j++) {
                int idx_a = k * sA[0] + i * sA[1] + j * sA[2];
                int idx_b = k * sB[0] + i * sB[1] + j * sB[2];
                int idx_out = k * (out_rows * out_cols) + i * out_cols + j;
                output_data[idx_out] = data_a[idx_a] + data_b[idx_b];
            }
        }
    }

    auto result = make_shared<Tensor>(output_shape, output_data.data(), vector<shared_ptr<Tensor>>{a, b});


    result->_backward = [a, b, result]() {

        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);


        a->grad = a->grad + result->grad; 
        b->grad = b->grad + result->grad;
    };

    return result;
}


float relu_function(float x){
    if(x<0) x=0;
    return x;
}

shared_ptr<Tensor> relu(shared_ptr<Tensor> a){
    vector<float> output_data(product(a->getShape()));
    for(size_t i=0; i< a->getSize();i++){
        output_data[i] = relu_function(a->getData()[i]); 
    }
    
    auto result = make_shared<Tensor>(a->getShape(), output_data.data(), vector<shared_ptr<Tensor>>{a});

    result->_backward = [a, result]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData(); 
        float* grad_output_data = result->grad->getData();
        float* input_val = a->getData();

        size_t size = a->getSize();
        for(size_t i=0; i<size; i++) {
            float local_deriv = (input_val[i] > 0) ? 1.0f : 0.0f;
            grad_input_data[i] += grad_output_data[i] * local_deriv;
        }
    };

    return result;
}


shared_ptr<Tensor> matmul(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    shared_ptr<Tensor> result;

    if (a->getDimension() == 1 && b->getDimension() == 1) result = make_shared<Tensor> (dot_scalar_product(a, b));
    else if (a->getDimension() == 1 && b->getDimension() == 2) {
        auto a_view = make_shared<Tensor>(*a);
        a_view->shape.insert(a_view->shape.begin(), 1);
        a_view->strides.insert(a_view->strides.begin(), 0);
        result = make_shared<Tensor> (vector_matrix_product(a_view, b));
    }
    else if (a->getDimension() == 2 && b->getDimension() == 1) {
        auto b_view = make_shared<Tensor>(*b);
        b_view->shape.push_back(1);
        b_view->strides.push_back(0);
        result = make_shared<Tensor> (matrix_matrix_product(a, b_view));
    }
    else if (a->getDimension() == 2 && b->getDimension() == 2) result = make_shared<Tensor> (matrix_matrix_product(a, b));
    else if (a->getDimension() == 3 && b->getDimension() == 2) result = make_shared<Tensor> (batch_matrix_product(a, b));
    else if (a->getDimension() == 3 && b->getDimension() == 1) {
        auto b_view = make_shared<Tensor>(*b);
        b_view->shape.push_back(1);
        b_view->strides.push_back(0);
        result = make_shared<Tensor> (batch_matrix_product(a, b_view));
    }
    else if(a->getDimension() == 1 && b->getDimension() == 3){
        result = make_shared<Tensor> (vector_batch_product(a,b));
    }
    else if(a->getDimension()==2 && b->getDimension() == 3){
        result = make_shared<Tensor>(matrix_batch_product(a,b));
    }
    else if(a->getDimension() == 3 && b->getDimension() == 3) {
        result = make_shared<Tensor>(batch_batch_product(a, b));
    }
    else throw runtime_error("Dimensiones no soportadas");

    result->childs = {a, b}; 


    result->_backward = [a, b, result]() {

        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);

        auto grad_output = result->grad;
        
        // dA = grad_output @ B.T
        auto b_T = transpose_view(b);
        auto da = matmul(grad_output, b_T);
        a->grad = a->grad + da;

        // dB = A.T @ grad_output
        auto a_T = transpose_view(a);
        auto db = matmul(a_T, grad_output);
        b->grad = b->grad + db;
    };

    return result;
}

shared_ptr<Tensor> transpose_view(shared_ptr<Tensor> a) {

    auto result = make_shared<Tensor>(*a);
    

    if (a->getDimension() == 2) {
        std::swap(result->shape[0], result->shape[1]);
        std::swap(result->strides[0], result->strides[1]);
    } 
    else if (a->getDimension() == 3) {
        std::swap(result->shape[1], result->shape[2]);
        std::swap(result->strides[1], result->strides[2]);
    } 
    else {
        throw std::runtime_error("transpose_view: Dimensiones no soportadas");
    }


    result->childs = {a};

    // Y = X.T -> dL/dX = (dL/dY).T
    result->_backward = [a, result]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        

        auto grad_transposed = transpose_view(result->grad);
        
        a->grad = a->grad + grad_transposed;
    };
    
    return result;
}