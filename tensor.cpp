#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <functional>
#include <array>
#include <cassert>

using namespace std;

struct Tensor : std::enable_shared_from_this<Tensor> {
private:
    shared_ptr<float[]> data;
    size_t total_size;
    vector<shared_ptr<Tensor>> childs;
    shared_ptr<Tensor> grad;
    std::function<void()> _backward;
    size_t offset = 0;

public:
    vector<int> shape;
    vector<int> strides;

    Tensor(const vector<int>& str, float* data_param = nullptr,
           const vector<shared_ptr<Tensor>> childs_param = {}) : data(nullptr),
            total_size(1), shape(str), childs(childs_param) {
        for (const auto& ptr : str) total_size *= ptr;

        if (total_size == 0) return;
        data = shared_ptr<float[]>(new float[total_size]);
        if (data_param != nullptr) {
            for (size_t i = 0; i < total_size; i++) {
                data[i] = data_param[i];
            }
        }

        strides.resize(str.size());
        strides[str.size() - 1] = 1;
        for (int i = str.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * str[i + 1];
        }
    }

    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor(Tensor&& other) = default;
    Tensor& operator=(Tensor&& other) = default;

    void printElements(int count = 1) const {
        cout << "Elementos del tensor" << endl;
        for (int i = 0; i < count; i++) {
            cout << "Elemento " << i << ": " << data[i] << endl;
        }
    }

    size_t getSize() const { return total_size; }
    vector<int> getStrides() const { return strides; }
    float* getData() const { return data.get() + offset; }
    vector<int> getShape() const { return shape; }
    vector<shared_ptr<Tensor>> getChilds() const { return childs; }
    int getDimension() const { return shape.size(); }
    void setShape(vector<int> p) { shape = p; }

    void printShape() {
        cout << "Shape: ("; 
        for (size_t i = 0; i < shape.size(); i++) {
            cout << shape[i];
            if (i != shape.size() - 1) cout << ", ";
        }
        cout << ")" << endl;
    }

    shared_ptr<Tensor> getBatch(int index) {
        assert(getDimension() == 3 && "Tensor must have three dimensions");
        int batch_size = shape.at(0);
        int rows = shape.at(1);
        int cols = shape.at(2);
        if (index < 0 || index >= batch_size)
            throw std::out_of_range("Batch index out of range");

        auto tensor_view = make_shared<Tensor>(*this);
        tensor_view->shape = {rows, cols};
        tensor_view->offset = offset + index * strides[0];
        tensor_view->strides = {cols,1};
        tensor_view->childs = {shared_from_this()};
        return tensor_view;
    }
};

static size_t product(const vector<int>& v) {
    size_t p = 1;
    for (int x : v) p *= x;
    return p;
}

void broadcasting_inplace(Tensor& small, const Tensor& large) {
    for (size_t d = 0; d < small.shape.size(); ++d) {
        if (small.shape[d] == 1 && large.shape[d] != 1) {
            small.strides[d] = 0;
        }
    }
}


Tensor dot_scalar_product(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    int size_output = a->getShape().at(0);
    vector<float> output_data(size_output);
    for (int i = 0; i < size_output; i++)
        output_data[i] = a->getData()[i] * b->getData()[i];
    return Tensor({size_output}, output_data.data(), {});
}

Tensor vector_matrix_product(shared_ptr<Tensor> v, shared_ptr<Tensor> m) {
    int column_v = v->getShape().at(1);
    int column_m = m->getShape().at(1);
    vector<float> output_data(column_m);
    for (int i = 0; i < column_m; i++) {
        float sum = 0;
        for (int j = 0; j < column_v; j++) {
            sum += v->getData()[j] * m->getData()[i + j * m->strides.at(0)];
        }
        output_data[i] = sum;
    }
    return Tensor({column_m}, output_data.data(), {});
}

Tensor matrix_matrix_product(shared_ptr<Tensor> m, shared_ptr<Tensor> v) {
    int row_m = m->shape.at(0);
    int col_v = v->shape.at(1);
    int col_m = m->shape.at(1);
    vector<int> output_shape = {row_m, col_v};
    vector<float> output_data(product(output_shape));
    for (int i = 0; i < row_m; i++) {
        for (int j = 0; j < col_v; j++) {
            float sum = 0;
            for (int k = 0; k < col_m; k++)
                sum += m->getData()[k + i * m->strides.at(0)] * v->getData()[j + k * v->strides.at(0)];
            output_data[i * col_v + j] = sum;
        }
    }
    return Tensor(output_shape, output_data.data(), {});
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
        std::copy(matrix_data, matrix_data + b_row * m_col, output_data.begin() + b_row * m_col * i);
    }
    return Tensor(output_shape, output_data.data(), {});
}

Tensor vector_batch_product(shared_ptr<Tensor> v,shared_ptr<Tensor> b){
    v->shape.insert(v->shape.begin(),1);
    v->strides.insert(v->strides.begin(),0);

    int b_batch = b->shape.at(0);
    int b_col = b->shape.at(2);
    vector<int> output_shape = {b_batch,b_col};

    vector<float> output_data(product(output_shape));

    for(int i = 0; i < b_batch; i++){
        auto matrix_batch_i = b->getBatch(i);
        Tensor matrix_output = vector_matrix_product(v, matrix_batch_i);
        matrix_output.printShape();
        float* matrix_data = matrix_output.getData();
        std::copy(matrix_data, matrix_data + b_col, output_data.begin() + i * b_col);
    }
    return Tensor(output_shape, output_data.data(), {});
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
        std::copy(matrix_data, matrix_data + m_row*b_col, output_data.begin() + m_row*b_col* i);
    }
    return Tensor(output_shape, output_data.data(), {});
}



Tensor matmul(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {

    if (a->getDimension() == 1 && b->getDimension() == 1) return dot_scalar_product(a, b);
    if (a->getDimension() == 1 && b->getDimension() == 2) {
        a->shape.insert(a->shape.begin(), 1);
        a->strides.insert(a->strides.begin(), 0);
        return vector_matrix_product(a, b);
    }
    if (a->getDimension() == 2 && b->getDimension() == 1) {
        b->shape.push_back(1);
        b->strides.push_back(0);
        return matrix_matrix_product(a, b);
    }
    if (a->getDimension() == 2 && b->getDimension() == 2) return matrix_matrix_product(a, b);
    if (a->getDimension() == 3 && b->getDimension() == 2) return batch_matrix_product(a, b);
    if (a->getDimension() == 3 && b->getDimension() == 1) {
        b->shape.push_back(1);
        b->strides.push_back(0);
        return batch_matrix_product(a, b);
    }

    if(a->getDimension() == 1 && b->getDimension() == 3){
        return vector_batch_product(a,b);
    }
    if(a->getDimension()==2 && b->getDimension() == 3){
        return matrix_batch_product(a,b);
    }
    throw runtime_error("Dimensiones no soportadas");
}

int main() {
    auto tensor1 = make_shared<Tensor>(vector<int>{4, 3, 2}, new float[24]{1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1});
    auto tensor2 = make_shared<Tensor>(vector<int>{2,3}, new float[6]{6,5,4,3,2,1});
    tensor2->printShape();
    Tensor tensor3 = matmul(tensor1, tensor2);
 
    
    tensor3.printElements(16);
    tensor3.printShape();
}
